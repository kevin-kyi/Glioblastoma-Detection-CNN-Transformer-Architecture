import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_sampler import BraTSSliceDataset, SimpleAugment
from encoder import ModalityEncoder
from transformer import FusionTransformer
from hybridloss import HybridDiceFocalLoss

def save_checkpoints(epoch, encoders, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for mod, model in encoders.items():
        save_path = os.path.join(checkpoint_dir, f"{mod}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved for {mod} at epoch {epoch}: {save_path}")

def save_transformer(epoch, transformer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pth")
    torch.save(transformer.state_dict(), save_path)
    print(f"Checkpoint saved for transformer at epoch {epoch}: {save_path}")

def expand_channels(image):
    """
    Convert a single-channel image (H, W) into a 3-channel image (3, H, W)
    by replicating the image.
    """
    image = np.expand_dims(image, axis=0) 
    return np.repeat(image, 3, axis=0)     

def collate_fn(batch):
    # Prepare lists for each modality and segmentation masks.
    modalities = [[] for _ in range(4)]
    seg_masks = []
    for sample in batch:
        imgs = sample['image']
        for i in range(len(imgs)):
            modalities[i].append(expand_channels(imgs[i]))  # Expand each image to 3 channels.
        seg_masks.append(sample['seg'])
    modalities = [torch.tensor(np.stack(mod_list), dtype=torch.float32) for mod_list in modalities]
    seg_masks = torch.tensor(np.stack(seg_masks), dtype=torch.float32).unsqueeze(1)
    return {'image': modalities, 'seg': seg_masks}

def main():
    # Construct the path relative to this file.
    train_data_path = "/content/drive/MyDrive/final-project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"
    print(f"******************* Starting Sampling *******************")
    dataset = BraTSSliceDataset(train_data_path,
                                modalities=['t1c', 't1n', 't2f', 't2w'],
                                positive_label=3,  # Use label 3 based on your output.
                                transform=SimpleAugment())
    print(f"******************* Ending Sampling *******************")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create four modality-specific encoder models.
    modality_names = ['t1c', 't1n', 't2f', 't2w']
    encoders = {}
    optimizers = {}
    criterion = nn.BCEWithLogitsLoss()  # Binary segmentation loss
    for mod in modality_names:
        model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
        encoders[mod] = model
        optimizers[mod] = optim.Adam(model.parameters(), lr=1e-4)

    print(f"******************* TRAINING STARTED *******************")
    # checkpoint_dir = "/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints"

    
    # num_epochs = 5  # Change as needed.
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         imgs = batch['image']  # List of 4 tensors, one per modality: (B, 3, H, W)
    #         segs = batch['seg'].to(device)  # (B, 1, H, W)
    #         losses = {}
    #         # Train each modality's encoder separately.
    #         for i, mod in enumerate(modality_names):
    #             inputs = imgs[i].to(device)
    #             outputs, features = encoders[mod](inputs)
    #             # Upsample outputs if necessary.
    #             if outputs.shape[-2:] != segs.shape[-2:]:
    #                 outputs = nn.functional.interpolate(outputs, size=segs.shape[-2:], mode='bilinear', align_corners=False)
    #             loss = criterion(outputs, segs)
    #             optimizers[mod].zero_grad()
    #             loss.backward()
    #             optimizers[mod].step()
    #             losses[mod] = loss.item()
    #         print(f"Epoch {epoch+1}, Losses: {losses}")

    #     save_checkpoints(epoch, encoders, checkpoint_dir)
    #     print(f"Saved State Dict for Epoch {epoch} in /checkpoints folder")

    print(f"******************* TRANSFORMER TRAINING *******************")
    # Load encoders
    encoder_dir = "/content/drive/MyDrive/Glioblastoma-Detection-CNN-Transformer-Architecture/checkpoints"
    for mod in modality_names:
        encoders[mod].load_state_dict(torch.load(os.path.join(encoder_dir, f"{mod}_epoch_1.pth"), map_location=device))
        encoders[mod].to(device)
        encoders[mod].eval()
    transformer_dir = "/content/drive/MyDrive/Glioblastoma-Detection-CNN-Transformer-Architecture/transformer"
    
    # Train the transformer
    transformer = FusionTransformer(feature_dim=1280, num_layers=6, num_heads=8, num_classes=1).to(device)
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=1e-4)
    criterion = HybridDiceFocalLoss(dice_weight=0.7, focal_weight=0.3, alpha=0.25, gamma=2.0)
    transformer_epochs = 5
    for epoch in range(transformer_epochs):
        for batch in dataloader:
            imgs = batch['image']
            segs = batch['seg'].to(device)
            batch_features = []
            for i, mod in enumerate(modality_names):
                inputs = imgs[i].to(device)
                outputs, features = encoders[mod](inputs)
                # Save features for transformer training
                batch_features.append(features)
            batch_features = torch.cat(batch_features, dim=1).to(device)  # Concatenate features from all modalities
            transformer_optimizer.zero_grad()
            transformer_out = transformer(batch_features)
            if transformer_out.shape[-2:] != segs.shape[-2:]:
                transformer_out = nn.functional.interpolate(transformer_out, size=segs.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(transformer_out, segs)
            loss.backward()
            transformer_optimizer.step()
            print(f"Epoch {epoch+1}, Transformer Loss: {loss.item()}")
        save_transformer(epoch, transformer, transformer_dir)
    

    # -------------------------------------------------------------------
    # ***** WILLLLL ******** VISUALIZATION PORTION (For Just the "t1c" prediction and actual semgmentation):
    # -------------------------------------------------------------------
    sample = dataset[29]  
    mod_index = 0        
    mod_name = modality_names[mod_index]
    ground_truth = sample['seg']          

    # Prepare the image: expand to 3 channels, convert to tensor.
    
    
    # Set the corresponding model to evaluation mode and run inference.
    print(f"******************* Starting Inference *******************")
    # encoder_model = encoders[mod_name]
    # encoder_model.eval()
    # with torch.no_grad():
    #     out, _ = encoder_model(image_tensor)
    #     if out.shape[-2:] != image_np.shape:
    #         out = nn.functional.interpolate(out, size=image_np.shape, mode='bilinear', align_corners=False)
    #     pred_prob = torch.sigmoid(out)
    #     pred_mask = (pred_prob.squeeze().cpu().numpy() > 0.5).astype(np.float32)
    features_test = []
    for i, mod in enumerate(modality_names):
        image_np = sample['image'][i]
        image_expanded = expand_channels(image_np)
        inputs = torch.tensor(image_expanded, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, H, W)
        encoder_model = encoders[mod]
        encoder_model.eval()
        outputs, features = encoder_model(inputs)
        # Save features for transformer training
        features_test.append(features)
    features_test = torch.cat(features_test, dim=1).to(device)
    transformer.eval()
    with torch.no_grad():
        out = transformer(features_test)
        if out.shape[-2:] != image_np.shape:
            out = nn.functional.interpolate(out, size=image_np.shape, mode='bilinear', align_corners=False)
        pred_prob = torch.sigmoid(out)
        pred_mask = (pred_prob.squeeze().cpu().numpy() > 0.5).astype(np.float32)
    
    # Visualize the original image, predicted mask, and ground truth mask side by side.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np.T, cmap='gray', origin='lower')
    plt.title(f"{mod_name} Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.T, cmap='gray', origin='lower')
    plt.title("Predicted Segmentation")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth.T, cmap='gray', origin='lower')
    plt.title("Ground Truth Segmentation")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
