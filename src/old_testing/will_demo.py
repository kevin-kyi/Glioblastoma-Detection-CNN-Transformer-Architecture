# import torch
# import torch.nn.functional as F
# from unet import MultiModalSegModel
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------- Load Model -------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MultiModalSegModel().to(device)
# checkpoint_path = "checkpoints/colab/demo1_multimodal_unet (1).pth"
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()
# print("✅ Model loaded and set to eval mode.")

# # ------------------- Preprocess Input -------------------
# def load_and_preprocess(png_path):
#     img = Image.open(png_path).convert('L')  # Grayscale
#     img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
#     tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#     return tensor

# t1c_tensor = load_and_preprocess('data/t1c_slice.jpeg')
# t1n_tensor = load_and_preprocess('data/t1n_slice.jpeg')
# t2f_tensor = load_and_preprocess('data/t2f_slice.jpeg')
# t2w_tensor = load_and_preprocess('data/t2w_slice.jpeg')

# inputs = {
#     't1c': t1c_tensor.to(device),
#     't1n': t1n_tensor.to(device),
#     't2f': t2f_tensor.to(device),
#     't2w': t2w_tensor.to(device)
# }

# # ------------------- Run Inference -------------------
# with torch.no_grad():
#     outputs = model(inputs)
#     logits = outputs['t2f']
#     logits = F.interpolate(logits, size=t1c_tensor.shape[-2:], mode='bilinear', align_corners=False)
#     probs = torch.sigmoid(logits)
#     pred_mask = (probs > 0.5).float()[0, 0].cpu().numpy()  # Binary mask

# # ------------------- Visualize Without Rotation -------------------
# input_img = t1c_tensor[0, 0].cpu().numpy()

# # plt.figure(figsize=(6,6))
# # plt.imshow(input_img, cmap='gray')  # ← No transpose, no flip
# # # plt.imshow(pred_mask, cmap='jet', alpha=0.4)  # Overlay mask directly
# # plt.title("Tumor Segmentation Overlay (T1C)")
# # plt.axis('off')
# # plt.savefig("inference_result_no_rotation.png")
# # plt.show()



# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(input_img, cmap='gray', origin='lower')
# plt.title(f"T2f Image")
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(pred_mask, cmap='gray', origin='lower')
# plt.title("Predicted Segmentation")
# plt.axis('off')


# plt.tight_layout()
# plt.show()
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
    out = model(inputs)['t2f']                      # pick modality
    out = F.interpolate(out, size=t1c.shape[-2:],   # match size
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
