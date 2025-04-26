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
CHECKPOINT_PATH = "/content/drive/MyDrive/Glioblastoma-Detection-CNN-Transformer-Architecture/checkpoints/transformer_testing2_epoch80.pth"
DATA_PATH = "/content/drive/MyDrive/final-project/data/BraTS2025-GLI-PRE-Challenge-TrainingData"

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
    logits_t1c = outputs['t1c']
    probs_t1c = torch.sigmoid(logits_t1c)[0,0].cpu().numpy()
    
    logits_t1n = outputs['t1n']
    probs_t1n = torch.sigmoid(logits_t1n)[0,0].cpu().numpy()
    
    logits_t2f = outputs['t2f']
    probs_t2f = torch.sigmoid(logits_t2f)[0,0].cpu().numpy()
    
    logits_t2w = outputs['t2w']
    probs_t2w = torch.sigmoid(logits_t2w)[0,0].cpu().numpy()

# ---------- Visualization (Heatmap Overlay) ----------
t1c_image = sample['t1c'][0,0].cpu().numpy()
t1n_image = sample['t1n'][0,0].cpu().numpy()
t2f_image = sample['t2f'][0,0].cpu().numpy()
t2w_image = sample['t2w'][0,0].cpu().numpy()

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(t1c_image.T, cmap='gray', origin='lower')
plt.imshow(probs_t1c.T, cmap='hot', alpha=0.5, origin='lower')  # ← Heatmap overlay
plt.title("Heatmap Overlay (T1C + Prediction)")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(t1n_image.T, cmap='gray', origin='lower')
plt.imshow(probs_t1n.T, cmap='hot', alpha=0.5, origin='lower')  # ← Heatmap overlay
plt.title("Heatmap Overlay (T1N + Prediction)")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(t2f_image.T, cmap='gray', origin='lower')
plt.imshow(probs_t2f.T, cmap='hot', alpha=0.5, origin='lower')  # ← Heatmap overlay
plt.title("Heatmap Overlay (T2F + Prediction)")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(t2w_image.T, cmap='gray', origin='lower')
plt.imshow(probs_t2w.T, cmap='hot', alpha=0.5, origin='lower')  # ← Heatmap overlay
plt.title("Heatmap Overlay (T2W + Prediction)")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(probs_t1n.T, cmap='gray', origin='lower')
plt.title("Predicted"); plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(mask.T, cmap='gray', origin='lower')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.tight_layout()
plt.savefig("results/heatmap_overlay.png")
plt.show()
