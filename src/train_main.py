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

# Hybrid Dice + BCE loss
bce_loss = nn.BCEWithLogitsLoss()
def tversky_loss(logits, truth, alpha=0.3, beta=0.7, eps=1e-6):
    prob = torch.sigmoid(logits)
    TP = (prob * truth).sum(dim=[2,3])
    FP = ((1-truth) * prob).sum(dim=[2,3])
    FN = (truth * (1-prob)).sum(dim=[2,3])
    tversky = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    return 1 - tversky.mean()

bce_weight  = 1.0
dice_weight = 2.0



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset + loader
    train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'data', 'BraTS2025-GLI-PRE-Challenge-TrainingData')
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

    # Model, optimizer
    model = MultiModalSegModel().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)


    epochs = 100
    for epoch in range(1, epochs+1):
        print(f"Starting Epoch: {epoch}/{epochs} — {len(loader)} batches")
        model.train()

        epoch_bce  = 0.0
        epoch_dice = 0.0
        epoch_loss = 0.0

        for batch in loader:
            inputs = {m: batch[m].to(device) for m in ['t1c','t1n','t2f','t2w']}
            masks  = batch['seg'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            batch_bce  = 0.0
            batch_dice = 0.0

            # accumulate per‑modality losses
            for mod, logits in outputs.items():
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )

                l_bce  = bce_loss(logits, masks)
                l_dice = tversky_loss(logits, masks)

                batch_bce  += l_bce
                batch_dice += l_dice

            # apply weights
            loss = bce_weight * batch_bce + dice_weight * batch_dice
            loss.backward()
            optimizer.step()

            # track for epoch
            epoch_bce  += batch_bce.item()
            epoch_dice += batch_dice.item()
            epoch_loss += loss.item()

        # compute averages
        n_batches = len(loader)
        print(
            f"Epoch {epoch:3d}/{epochs} — "
            f"BCE: {(epoch_bce/n_batches):.4f}, "
            f"Dice: {(epoch_dice/n_batches):.4f}, "
            f"Total: {(epoch_loss/n_batches):.4f}"
        )

        # Save model
        # os.makedirs('checkpoints', exist_ok=True)
        # torch.save(model.state_dict(), f'checkpoints/multimodal_unet_epoch{epoch}.pth')




    # ***********  VISUALIZATION PORTION  ************
    eval_idx = 3
    sample   = ds[eval_idx]
    image_np = sample['image'][0]    # modality 't1c'
    gt_mask  = sample['seg']         # (H, W)

    # build a (1,1,H,W) tensor from the numpy slice
    img_tensor = torch.tensor(
        image_np[None, None, :, :],
        dtype=torch.float32,
        device=device
    )

    # forward only the t1c encoder+decoder
    model.eval()
    with torch.no_grad():
        feats  = model.encoders['t1c'](img_tensor)
        logits = model.decoders['t1c'](feats)
        pred   = torch.sigmoid(logits)[0,0].cpu().numpy()  # (H, W)
        pred_mask = (pred > 0.5).astype(float)

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image_np.T, cmap='gray', origin='lower')
    plt.title(f"T1c Input (slice {eval_idx})")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(pred_mask.T, cmap='gray', origin='lower')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(gt_mask.T, cmap='gray', origin='lower')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.tight_layout()
    plt.show()





  