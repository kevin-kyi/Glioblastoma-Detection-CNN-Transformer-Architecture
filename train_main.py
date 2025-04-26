import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_sampler import BraTSSliceDataset, SimpleAugment
from unet import MultiModalSegModel
from hybridloss import HybridDiceFocalLoss
from transformer import FusionTransformer

# Collate that groups modalities and mask
def collate_fn(batch):
    modalities = [[] for _ in range(4)]
    masks = []
    for sample in batch:
        for i, img in enumerate(sample['image']):
            # img is (H, W) → expand to (1, H, W)
            modalities[i].append(np.expand_dims(img, 0))
        masks.append(np.expand_dims(sample['seg'], 0))  # (1, H, W)

    modalities = [
        torch.tensor(np.stack(m), dtype=torch.float32)  # now (B, 1, H, W)
        for m in modalities
    ]
    masks = torch.tensor(np.stack(masks), dtype=torch.float32)  # already (B,1,H,W)
    return {
        't1c': modalities[0],
        't1n': modalities[1],
        't2f': modalities[2],
        't2w': modalities[3],
        'seg': masks
    }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset + loader
    train_data_path = "/content/drive/MyDrive/final-project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"
    ds = BraTSSliceDataset(train_data_path,
                                modalities=['t1c', 't1n', 't2f', 't2w'],
                                positive_label=3,  # Use label 3 based on your output.
                                transform=SimpleAugment())
    # ds = BraTSSliceDataset(train_data_path,
    #                             modalities=['t1c', 't1n', 't2f', 't2w'],
    #                             positive_label=3)
    loader = DataLoader(
        ds, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )

    # Load UNet model
    model = MultiModalSegModel().to(device)
    checkpoint_path = ('/content/drive/MyDrive/Glioblastoma-Detection-CNN-Transformer-Architecture/checkpoints/transformer_testing1_epoch60.pth')
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Transformer, optimizer, scheduler and loss
    transformer = FusionTransformer(in_chans=4*112).to(device)
    optimizer = optim.AdamW(transformer.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


    sample = next(iter(loader))
    image = sample['t1c'][0,0].cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(4,4))

    # plt.subplot(1,1,1)
    # plt.imshow(image.T, cmap='gray', origin='lower')
    # plt.title("t1c Input")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.savefig("transTest1.png")
    # plt.show()
    epochs = 500
    for epoch in range(1, epochs+1):
        print(f"Starting Epoch: {epoch}/{epochs} — {len(loader)} batches")
        transformer.train()

        epoch_loss = 0.0

        batch = next(iter(loader))
        inputs = {m: batch[m].to(device) for m in ['t1c','t1n','t2f','t2w']}
        masks  = batch['seg'].to(device)
        features = {}
        for mod, i in inputs.items():
            mod_output = model.encoders[mod](i)
            features[mod] = mod_output[-2]
            
        # Concatenate features from all modalities
        fused = torch.cat([features[m] for m in ['t1c','t1n','t2f','t2w']], dim=1)
            
        optimizer.zero_grad()
        logits = transformer(fused)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # track for epoch
        epoch_loss += loss.item()
        
        # compute averages
        n_batches = len(loader)
        print(
            f"Epoch {epoch:3d}/{epochs} — "
            f"Total: {(epoch_loss/n_batches):.4f}"
        )

        # Save transformer
        if epoch == 1 or epoch % 50 == 0:
            print(f"Saving model at epoch {epoch}")
            os.makedirs('transformer', exist_ok=True)
            transformer_checkpoint = {
                'epoch': epoch,
                'transformer_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }
            torch.save(transformer_checkpoint, f'transformer/transformer_epoch{epoch}.pth')

    # ***********  VISUALIZATION PORTION  ************
    # eval_idx = 3
    # sample   = ds[eval_idx]
    # image_np = sample['image'][0]    # modality 't1c'
    # gt_mask  = sample['seg']         # (H, W)

    # # build a (1,1,H,W) tensor from the numpy slice
    # img_tensor = torch.tensor(
    #     image_np[None, None, :, :],
    #     dtype=torch.float32,
    #     device=device
    # )

    # model.eval()
    # transformer.eval()
    # with torch.no_grad():
    #     features = {}
    #     for mod, i in zip(['t1c','t1n','t2f','t2w'], img_tensor):
    #         mod_output = model.encoders[mod](i)
    #         features[mod] = mod_output[-3]
    #     fused  = torch.cat([features[m] for m in ['t1c','t1n','t2f','t2w']], dim=1)
    #     logits = transformer(fused)
    #     pred   = torch.sigmoid(logits)[0,0].cpu().numpy()  # (H, W)
    #     pred_mask = (pred > 0.5).astype(float)

    # # plot
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12,4))

    # plt.subplot(1,3,1)
    # plt.imshow(image_np.T, cmap='gray', origin='lower')
    # plt.title(f"T1c Input (slice {eval_idx})")
    # plt.axis('off')

    # plt.subplot(1,3,2)
    # plt.imshow(pred_mask.T, cmap='gray', origin='lower')
    # plt.title("Predicted Mask")
    # plt.axis('off')

    # plt.subplot(1,3,3)
    # plt.imshow(gt_mask.T, cmap='gray', origin='lower')
    # plt.title("Ground Truth")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()





  