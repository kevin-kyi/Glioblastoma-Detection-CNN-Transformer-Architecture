import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_sampler import BraTSSliceDataset, SimpleAugment
from old_testing.encoder import ModalityEncoder

import time



def save_checkpoints(epoch, encoders, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for mod, model in encoders.items():
        save_path = os.path.join(checkpoint_dir, f"{mod}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved for {mod} at epoch {epoch}: {save_path}")




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
            modalities[i].append(expand_channels(imgs[i]))  
        seg_masks.append(sample['seg'])
    modalities = [torch.tensor(np.stack(mod_list), dtype=torch.float32) for mod_list in modalities]
    seg_masks = torch.tensor(np.stack(seg_masks), dtype=torch.float32).unsqueeze(1)
    return {'image': modalities, 'seg': seg_masks}

def main():
    # Construct the path relative to this file.
    train_data_path = "/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"
    print(f"******************* Starting Sampling *******************")
    dataset = BraTSSliceDataset(train_data_path,
                                modalities=['t1c', 't1n', 't2f', 't2w'],
                                positive_label=3,  # Use label 3 based on your output.
                                transform=SimpleAugment())
    print(f"******************* Ending Sampling *******************")
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create four modality-specific encoder models.
    modality_names = ['t1c', 't1n', 't2f', 't2w']
    encoders = {}
    optimizers = {}
    criterion = nn.BCEWithLogitsLoss()  # Binary segmentation loss

    # for mod in modality_names:
        # model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
        # encoders[mod] = model
        # optimizers[mod] = optim.Adam(model.parameters(), lr=4e-4)

    # t1c
    model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
    checkpoint_path = 'checkpoints/colab/t1c_epoch_1.pth'   
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    encoders['t1c'] = model
    optimizers['t1c'] = optim.Adam(model.parameters(), lr=4e-4)


    # t1n
    model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
    checkpoint_path = 'checkpoints/colab/t1n_epoch_1.pth'
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    encoders['t1n'] = model
    optimizers['t1n'] = optim.Adam(model.parameters(), lr=4e-4)


    # t2f
    model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
    checkpoint_path = 'checkpoints/colab/t2f_epoch_1.pth' 
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    encoders['t2f'] = model
    optimizers['t2f'] = optim.Adam(model.parameters(), lr=4e-4)


    # t2w
    model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
    checkpoint_path = 'checkpoints/colab/t2w_epoch_1.pth'
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    encoders['t2w'] = model
    optimizers['t2w'] = optim.Adam(model.parameters(), lr=4e-4)



    


    
    # print(f"******************* TRAINING STARTED *******************")
    # checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    
    # num_epochs = 5  # Change as needed.
    # for epoch in range(num_epochs):

    #     for batch in dataloader:
    #         start = time.time()

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

    
    print(f"******************* TRAINING STARTED *******************")
    checkpoint_dir = "/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints"

    
    num_epochs = 5  # Change as needed.
    for epoch in range(num_epochs):
        for batch in dataloader:
            imgs = batch['image']  # List of 4 tensors, one per modality: (B, 3, H, W)
            segs = batch['seg'].to(device)  # (B, 1, H, W)
            losses = {}
            # Train each modality's encoder separately.
            for i, mod in enumerate(modality_names):
                inputs = imgs[i].to(device)
                outputs, features = encoders[mod](inputs)
                # Upsample outputs if necessary.
                if outputs.shape[-2:] != segs.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=segs.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, segs)
                optimizers[mod].zero_grad()
                loss.backward()
                optimizers[mod].step()
                losses[mod] = loss.item()
            print(f"Epoch {epoch+1}, Losses: {losses}")

    #     save_checkpoints(epoch, encoders, checkpoint_dir)
    #     print(f"Saved State Dict for Epoch {epoch} in /checkpoints folder")







    # -------------------------------------------------------------------
    # ***** WILLLLL ******** VISUALIZATION PORTION (For Just the "t1c" prediction and actual semgmentation):
    # -------------------------------------------------------------------
    sample = dataset[29]  
    mod_index = 1      
    mod_name = modality_names[mod_index]
    print(f"Using Modality: {mod_name}")
    
    image_np = sample['image'][mod_index] 
    ground_truth = sample['seg']          

    # Prepare the image: expand to 3 channels, convert to tensor.
    image_expanded = expand_channels(image_np)
    image_tensor = torch.tensor(image_expanded, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Set the corresponding model to evaluation mode and run inference.
    print(f"******************* Starting Inference *******************")
    encoder_model = encoders[mod_name]
    encoder_model.eval()
    with torch.no_grad():
        out, _ = encoder_model(image_tensor)
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










# THIS IS TO SET DICT BACK TO EPOCH STATES

# for mod in modality_names:

#         optimizers[mod] = optim.Adam(model.parameters(), lr=4e-4)

#     # t1c
#     model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
#     checkpoint_path = '/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints/t1c_epoch_0.pth'   
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict)
#     encoders['t1c'] = model

#     # t1n
#     model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
#     checkpoint_path = '/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints/t1n_epoch_0.pth'   
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict)
#     encoders['t1n'] = model

#     # t2f
#     model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
#     checkpoint_path = '/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints/t2f_epoch_0.pth'   
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict)
#     encoders['t2f'] = model

#     # t2w
#     model = ModalityEncoder(in_channels=3, num_classes=1).to(device)
#     checkpoint_path = '/content/drive/MyDrive/Intro to Deep Learning 18-786/final project/checkpoints/t2w_epoch_0.pth'   
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict)
#     encoders['t2w'] = model