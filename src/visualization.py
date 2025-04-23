import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from data_sampler import BraTSSliceDataset
from unet import MultiModalSegModel

# ---------- Utility: Collate Function ----------
def collate_fn(batch):
    sample = batch[0]
    out = {}
    for i, mod in enumerate(['t1c','t1n','t2f','t2w']):
        arr = sample['image'][i]
        out[mod] = torch.from_numpy(arr[None,None,...]).float()
    mask = sample['seg']
    out['seg'] = torch.from_numpy(mask[None,None,...]).float()
    return out

# ---------- Setup ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Paths ----------
CHECKPOINT_PATH = "../drive/My Drive/Deep Learning Final Project/final-project/kevin-checkpoints/checkpoint_epoch_060.pth"
DATA_PATH = "../drive/My Drive/Deep Learning Final Project/final-project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"

# ---------- Load Model ----------
model = MultiModalSegModel().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")

# ---------- Load Sample ----------
ds = BraTSSliceDataset(DATA_PATH, modalities=['t1c','t1n','t2f','t2w'], positive_label=3)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

sample = next(iter(loader))
inputs = {mod: sample[mod].to(device) for mod in ['t1c','t1n','t2f','t2w']}
mask = sample['seg'][0,0].cpu().numpy()

# ---------- Inference ----------
with torch.no_grad():
    outputs = model(inputs)
    logits = outputs['t1c']
    probs = torch.sigmoid(logits)[0,0].cpu().numpy()

# ---------- Visualization (Heatmap Overlay) ----------
input_image = sample['t1c'][0,0].cpu().numpy()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(input_image.T, cmap='gray', origin='lower')
plt.imshow(probs.T, cmap='hot', alpha=0.5, origin='lower')  # ← Heatmap overlay
plt.title("Heatmap Overlay (T1C + Prediction)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mask.T, cmap='gray', origin='lower')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.tight_layout()
plt.savefig("results/heatmap_overlay.png")
plt.show()
