import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_bbox_score(seg_slice, bbox):
    x_min, y_min, x_max, y_max = bbox
    tumor_inside = np.count_nonzero(seg_slice[y_min:y_max, x_min:x_max] > 0)
    area_box = (x_max - x_min) * (y_max - y_min)
    tumor_total = np.count_nonzero(seg_slice > 0)
    
    if tumor_total == 0:
        return None

    score = (3 * tumor_inside - area_box - tumor_total) / tumor_total
    return score

root_path = 'BraTS2025-GLI-PRE-Challenge-TrainingData'

subject_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
print("Found {} subjects".format(len(subject_dirs)))

subject_dir = subject_dirs[0]
print("Using subject:", subject_dir)

files = os.listdir(subject_dir)
print("Files in subject:", files)

seg_file = None
for f in files:
    if 'seg' in f.lower() and f.endswith('.nii.gz'):
        seg_file = os.path.join(subject_dir, f)
        break
if seg_file is None:
    raise ValueError("Segmentation file not found in subject directory: " + subject_dir)
print("Using segmentation file:", seg_file)

# Load the segmentation image
seg_img = nib.load(seg_file)
seg_data = seg_img.get_fdata()

modality_file = None
for f in files:
    if 't2f' in f.lower() and f.endswith('.nii.gz'):
        modality_file = os.path.join(subject_dir, f)
        break
if modality_file is None:
    raise ValueError("T2F modality file not found in subject directory: " + subject_dir)
print("Using modality file:", modality_file)

mod_img = nib.load(modality_file)
mod_data = mod_img.get_fdata()


tumor_slice_indices = [82, 63, 74]

num_plots = len(tumor_slice_indices)
fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
if num_plots == 1:
    axes = [axes]  

for i, slice_index in enumerate(tumor_slice_indices):
    img_slice = mod_data[:, :, slice_index]
    seg_slice = seg_data[:, :, slice_index]
    
    # Display the image slice
    axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[i].set_title("Slice {}".format(slice_index))
    axes[i].axis('off')
    
    tumor_pixels = np.where(seg_slice > 0)
    if tumor_pixels[0].size > 0:
        y_min, y_max = np.min(tumor_pixels[0]), np.max(tumor_pixels[0])
        x_min, x_max = np.min(tumor_pixels[1]), np.max(tumor_pixels[1])
        bbox = (x_min, y_min, x_max, y_max)
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                                 edgecolor='red', facecolor='none')
        axes[i].add_patch(rect)
        
        score = compute_bbox_score(seg_slice, bbox)
        score_text = f"Score: {score:.2f}" if score is not None else "No tumor"
        # Display the score on the image
        axes[i].text(0.05, 0.95, score_text, transform=axes[i].transAxes,
                     fontsize=12, color='yellow', verticalalignment='top',
                     bbox=dict(facecolor='black', alpha=0.5))
    else:
        axes[i].text(0.5, 0.5, 'No tumor found', color='red',
                     fontsize=12, ha='center', va='center', transform=axes[i].transAxes)

plt.tight_layout()
plt.show()

