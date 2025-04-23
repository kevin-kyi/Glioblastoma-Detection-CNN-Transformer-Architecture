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

# collate that ensures every slice becomes (B=1, C=1, H, W)
def collate_fn(batch):
    # batch is a list of length 1, batch[0] is a dict{'image': [...], 'seg': ...}
    sample = batch[0]
    out = {}
    # for each modality, take the single HxW then add dims → (1,1,H,W)
    for i, mod in enumerate(['t1c','t1n','t2f','t2w']):
        arr = sample['image'][i]            # (H,W)
        tensor = torch.from_numpy(arr[None,None,...]).float()  # (1,1,H,W)
        out[mod] = tensor
    # segmentation mask
    mask = sample['seg']                  # (H,W)
    out['seg'] = torch.from_numpy(mask[None,None,...]).float()  # (1,1,H,W)
    return out

# BCE + Dice
bce_loss = nn.BCEWithLogitsLoss()
def dice_loss(logits, truth, eps=1e-6):
    prob = torch.sigmoid(logits)  # Ensure proper scaling [0,1]
    inter = (prob * truth).sum(dim=[1,2,3])  # Sum over H,W,C
    union = prob.sum([1,2,3]) + truth.sum([1,2,3])
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()  # Proper mean over batch and modalities

def tversky_loss(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    probs = torch.sigmoid(logits)
    TP = (probs * targets).sum(dim=[1,2,3])
    FP = ((1 - targets) * probs).sum(dim=[1,2,3])
    FN = (targets * (1 - probs)).sum(dim=[1,2,3])
    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return 1 - tversky.mean()


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_path = "../drive/My Drive/Deep Learning Final Project/final-project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"


    ds = BraTSSliceDataset(
        train_data_path,
        modalities=['t1c','t1n','t2f','t2w'],
        positive_label=3,
        # transform=SimpleAugment()               # no random aug for overfit test
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    n_epochs = 80
    learning_rate = 4e-3
    max_learning_rate = 7e-3
    BCE_WEIGHT = 1.0
    DICE_WEIGHT = 2.0

    model = MultiModalSegModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.5e-4)
  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,  # Increased from 6e-3 (66% higher peak)
        epochs=n_epochs,
        steps_per_epoch=len(loader),
        pct_start=0.15,  # Shorter warmup (15% vs 25%)
        anneal_strategy='linear',  # More aggressive than cosine
        div_factor=15,  # Stronger initial LR boost
        final_div_factor=1e4  # Proper decay
    )


    print(f"Starting Training, Num batches: {len(loader)}")

    sample_batch = next(iter(loader))  # Get a fixed sample batch for consistent visualization
    sample_inputs = {mod: sample_batch[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
    sample_mask = sample_batch['seg'][0,0].cpu().numpy()

    RESUME_PATH = None  # Ex. "results/checkpoints/checkpoint_epoch_020.pth"
    CHECKPOINT_PATH = "../drive/My Drive/Deep Learning Final Project/final-project/kevin-checkpoints"

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print(f"Not Resuming Checkpoint, Starting at Epoch 1")
        start_epoch = 1


    for ep in range(start_epoch, n_epochs+1):
        model.train()
        epoch_bce, epoch_dice = 0, 0
        
        for batch in loader:
            inputs = {mod: batch[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
            mask = batch['seg'].to(device)
            
            # Verify mask values (critical)
            if mask.max() > 1 or mask.min() < 0:
                print(f"Warning: Mask values out of range [{mask.min()}, {mask.max()}]")
                mask = torch.clamp(mask, 0, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss calculation per modality
            loss_bce = torch.stack([bce_loss(outputs[mod], mask) for mod in outputs]).mean()
            loss_dice = torch.stack([tversky_loss(outputs[mod], mask) for mod in outputs]).mean()
            # loss = loss_bce + loss_dice
            loss = BCE_WEIGHT * loss_bce + DICE_WEIGHT * loss_dice
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_bce += loss_bce.item()
            epoch_dice += loss_dice.item()
        
        # Normalize losses
        avg_bce = epoch_bce / len(loader)
        avg_dice = epoch_dice / len(loader)
        
        print(f"Epoch {ep:3d}/{n_epochs} | BCE: {avg_bce:.4f} | Dice: {avg_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # ********* SAVED PREDICTION IN RESULTS FOLDER EVERY N EPOCHS *********
        if ep % 30 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                outputs = model(sample_inputs)
                pred = torch.sigmoid(outputs['t1c'][0,0]).cpu().numpy()
                pred_mask = (pred > 0.5).astype(float)
            
            img_np = sample_inputs['t1c'][0,0].cpu().numpy()
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(img_np.T, cmap='gray', origin='lower')
            plt.title(f"Epoch {ep} - Input"); plt.axis('off')
            
            plt.subplot(1,3,2)
            plt.imshow(pred_mask.T, cmap='gray', origin='lower')
            plt.title("Prediction"); plt.axis('off')
            
            plt.subplot(1,3,3)
            plt.imshow(sample_mask.T, cmap='gray', origin='lower')
            plt.title("Ground Truth"); plt.axis('off')

            
            plt.tight_layout()
            plt.savefig(f"results/training_progress/epoch_{ep:03d}.png")
            plt.close()  # Important for memory management in Colab

            print(f"FIGURE SAVED TO TRAINING RESULTS/PROGRESS FOLDER")

            # ********* SAVE CHECKPOINT ***********
            checkpoint = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'bce': avg_bce,
                'dice': avg_dice
            }
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/checkpoint_epoch_{ep:03d}.pth")
            print(f"Checkpoint saved to {CHECKPOINT_PATH}/checkpoint_epoch_{ep:03d}.pth")
            
            
            model.train()  # Don't forget to switch back to training mode

        # ********* SAVED PREDICTION IN RESULTS FOLDER EVERY N EPOCHS *********


    # --- visualize final fit on that slice
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs = {mod: batch[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
        mask0  = batch['seg'][0,0].cpu().numpy()

        # pick one modality to plot, e.g. 't1c'
        logits = model(inputs)['t1c']
        pred = torch.sigmoid(logits)[0,0].cpu().numpy()
        pred_mask = (pred>0.5).astype(float)

    img_np = batch['t1c'][0,0].cpu().numpy()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img_np.T, cmap='gray', origin='lower')
    plt.title("T1c Input"); plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(pred_mask.T, cmap='gray', origin='lower')
    plt.title("Predicted"); plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(mask0.T, cmap='gray', origin='lower')
    plt.title("Ground Truth"); plt.axis('off')
    plt.tight_layout()

    plt.savefig("results/testing_training_results.png")

    plt.show()
