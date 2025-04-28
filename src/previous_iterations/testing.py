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

def collate_fn(batch):
    sample = batch[0]
    out = {}
    for i, mod in enumerate(['t1c','t1n','t2f','t2w']):
        arr = sample['image'][i]            
        tensor = torch.from_numpy(arr[None,None,...]).float()  
        out[mod] = tensor
    # segmentation mask
    mask = sample['seg']                  
    out['seg'] = torch.from_numpy(mask[None,None,...]).float() 
    return out

# BCE + Dice
bce_loss = nn.BCEWithLogitsLoss()
def tversky_loss(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    probs = torch.sigmoid(logits)
    TP = (probs * targets).sum(dim=[1,2,3])
    FP = ((1 - targets) * probs).sum(dim=[1,2,3])
    FN = (targets * (1 - probs)).sum(dim=[1,2,3])
    tversky = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
    return 1 - tversky.mean()

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'data', 'BraTS2025-GLI-PRE-Challenge-TrainingData')
    ds = BraTSSliceDataset(
        train_data_path,
        modalities=['t1c','t1n','t2f','t2w'],
        positive_label=3,
        transform=SimpleAugment()              
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MultiModalSegModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=6e-3)

    n_epochs = 50
    for ep in range(1, n_epochs+1):
        model.train()
        for batch in loader:
            inputs = {mod: batch[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
            mask   = batch['seg'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            total_bce  = 0.0
            total_dice = 0.0
            # sum over modalities
            for mod, logits in outputs.items():
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                total_bce  += bce_loss(logits, mask)
                total_dice += tversky_loss(logits, mask)
            loss = total_bce + total_dice

            loss.backward()
            optimizer.step()

        if ep % 10 == 0 or ep==1:
            print(f"Epoch {ep:3d}/{n_epochs} — BCE: {total_bce.item():.4f}, Dice: {total_dice.item():.4f}, Sum: {loss.item():.4f}")

    # --- visualize final fit on that slice
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs = {mod: batch[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
        mask0  = batch['seg'][0,0].cpu().numpy()

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
    plt.show()
