import torch
import torch.nn.functional as F
from unet import MultiModalSegModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Load Model -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalSegModel().to(device)
checkpoint_path = "checkpoints/colab/demo1_multimodal_unet (1).pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ------------------- Preprocess Input -------------------
def load_and_preprocess(png_path):
    img = Image.open(png_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None,None]  # (1,1,H,W)

t1c = load_and_preprocess('data/t1c_slice.jpeg').to(device)
t1n = load_and_preprocess('data/t1n_slice.jpeg').to(device)
t2f = load_and_preprocess('data/t2f_slice.jpeg').to(device)
t2w = load_and_preprocess('data/t2w_slice.jpeg').to(device)

inputs = {'t1c':t1c, 't1n':t1n, 't2f':t2f, 't2w':t2w}

# ------------------- Inference -------------------
with torch.no_grad():
    out = model(inputs)['t2f']                      
    out = F.interpolate(out, size=t1c.shape[-2:],   
                        mode='bilinear',
                        align_corners=False)
    mask = (torch.sigmoid(out)>0.5)[0,0].cpu().numpy()

img = t1c[0,0].cpu().numpy()

# ------------------- Overlay Visualization -------------------
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray', origin='lower')
# use a soft red overlay
plt.imshow(mask, cmap='Reds', alpha=0.3, origin='lower')
plt.title("T2f with Tumor Overlay")
plt.axis('off')
plt.tight_layout()
plt.show()
