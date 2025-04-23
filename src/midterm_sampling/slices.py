
# THIS SECTION IS FOR SHOWING THE BOUNDING BOX 

# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Define path to the training folder
# root_path = 'BraTS2025-GLI-PRE-Challenge-TrainingData'

# # List subject directories (each subject is assumed to be in its own folder)
# subject_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
# print("Found {} subjects".format(len(subject_dirs)))

# # Choose one subject from the training data, e.g., the first subject
# subject_dir = subject_dirs[0]
# print("Using subject:", subject_dir)

# # List files in the subject directory
# files = os.listdir(subject_dir)
# print("Files in subject:", files)

# # Find the segmentation file (assumes filename contains 'seg' and ends with .nii.gz)
# seg_file = None
# for f in files:
#     if 'seg' in f.lower() and f.endswith('.nii.gz'):
#         seg_file = os.path.join(subject_dir, f)
#         break
# if seg_file is None:
#     raise ValueError("Segmentation file not found in subject directory: " + subject_dir)
# print("Using segmentation file:", seg_file)

# # Load the segmentation image
# seg_img = nib.load(seg_file)
# seg_data = seg_img.get_fdata()

# # Choose a modality for visualization: For example, T2F (or change to 'flair' as needed)
# modality_file = None
# for f in files:
#     # Here we assume 't2f' appears in the file name for the T2 modality
#     if 't2f' in f.lower() and f.endswith('.nii.gz'):
#         modality_file = os.path.join(subject_dir, f)
#         break
# if modality_file is None:
#     raise ValueError("T2F modality file not found in subject directory: " + subject_dir)
# print("Using modality file:", modality_file)

# # Load the modality image (for visualization)
# mod_img = nib.load(modality_file)
# mod_data = mod_img.get_fdata()

# # Define fixed slice indices to visualize
# tumor_slice_indices = [82, 63, 74]

# # Plot the selected tumor slices with bounding boxes
# num_plots = len(tumor_slice_indices)
# fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
# if num_plots == 1:
#     axes = [axes]  # ensure axes is iterable

# for i, slice_index in enumerate(tumor_slice_indices):
#     # Extract the modality image slice and corresponding segmentation slice
#     img_slice = mod_data[:, :, slice_index]
#     seg_slice = seg_data[:, :, slice_index]
    
#     # Display the image slice
#     axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
#     axes[i].set_title("Slice {}".format(slice_index))
#     axes[i].axis('off')
    
#     # Compute bounding box if tumor pixels exist
#     tumor_pixels = np.where(seg_slice > 0)
#     if tumor_pixels[0].size > 0:
#         # Find min and max coordinates in the two axes (rows and cols)
#         y_min, y_max = np.min(tumor_pixels[0]), np.max(tumor_pixels[0])
#         x_min, x_max = np.min(tumor_pixels[1]), np.max(tumor_pixels[1])
#         width = x_max - x_min
#         height = y_max - y_min

#         # Create a Rectangle patch
#         rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
#                                  edgecolor='red', facecolor='none')
#         # Add the patch to the Axes
#         axes[i].add_patch(rect)
#     else:
#         axes[i].text(0.5, 0.5, 'No tumor found', color='red',
#                      fontsize=12, ha='center', va='center', transform=axes[i].transAxes)

# plt.tight_layout()
# plt.show()

# ******************************************************************************************

# THIS SECTION IS FOR SHOWING THE ACTUAL REGION USING THE MASK

# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt

# # Define path to the training folder
# root_path = 'BraTS2025-GLI-PRE-Challenge-TrainingData'

# # List subject directories (each subject is assumed to be in its own folder)
# subject_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
# print("Found {} subjects".format(len(subject_dirs)))

# # Choose one subject from the training data, e.g., the first subject
# subject_dir = subject_dirs[0]
# print("Using subject:", subject_dir)

# # List files in the subject directory
# files = os.listdir(subject_dir)
# print("Files in subject:", files)

# # Find the segmentation file (assumes filename contains 'seg' and ends with .nii.gz)
# seg_file = None
# for f in files:
#     if 'seg' in f.lower() and f.endswith('.nii.gz'):
#         seg_file = os.path.join(subject_dir, f)
#         break
# if seg_file is None:
#     raise ValueError("Segmentation file not found in subject directory: " + subject_dir)
# print("Using segmentation file:", seg_file)

# # Load the segmentation image
# seg_img = nib.load(seg_file)
# seg_data = seg_img.get_fdata()

# # Choose a modality for visualization: e.g., T2F (modify the keyword if needed)
# modality_file = None
# for f in files:
#     if 't2f' in f.lower() and f.endswith('.nii.gz'):
#         modality_file = os.path.join(subject_dir, f)
#         break
# if modality_file is None:
#     raise ValueError("T2F modality file not found in subject directory: " + subject_dir)
# print("Using modality file:", modality_file)

# # Load the modality image (for visualization)
# mod_img = nib.load(modality_file)
# mod_data = mod_img.get_fdata()

# # Define fixed slice indices to visualize
# slice_indices = [82, 63, 74]

# # Plot the selected slices with segmentation mask points overlaid
# num_plots = len(slice_indices)
# fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
# if num_plots == 1:
#     axes = [axes]  # Ensure axes is iterable

# for i, slice_index in enumerate(slice_indices):
#     # Extract the modality image slice and corresponding segmentation slice
#     img_slice = mod_data[:, :, slice_index]
#     seg_slice = seg_data[:, :, slice_index]
    
#     # Display the image slice
#     axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
#     axes[i].set_title("Slice {}".format(slice_index))
#     axes[i].axis('off')
    
#     # Identify the exact points from the segmentation mask (non-zero values)
#     tumor_y, tumor_x = np.where(seg_slice > 0)
    
#     # Overlay the points on the image slice using small red markers
#     axes[i].scatter(tumor_x, tumor_y, s=1, c='red', alpha=0.6)

# plt.tight_layout()
# plt.show()


# ******************************************************************************************



import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_bbox_score(seg_slice, bbox):
    """
    Compute a custom bounding box accuracy score given a segmentation mask (seg_slice) and bounding box.
    bbox: a tuple (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    # Tumor pixels inside the bounding box
    tumor_inside = np.count_nonzero(seg_slice[y_min:y_max, x_min:x_max] > 0)
    # Area of the bounding box
    area_box = (x_max - x_min) * (y_max - y_min)
    # Total tumor pixels in the entire slice
    tumor_total = np.count_nonzero(seg_slice > 0)
    
    # To avoid division by zero
    if tumor_total == 0:
        return None

    score = (3 * tumor_inside - area_box - tumor_total) / tumor_total
    return score

# Define path to the training folder
root_path = 'BraTS2025-GLI-PRE-Challenge-TrainingData'

# List subject directories (each subject is assumed to be in its own folder)
subject_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
print("Found {} subjects".format(len(subject_dirs)))

# Choose one subject from the training data, e.g., the first subject
subject_dir = subject_dirs[0]
print("Using subject:", subject_dir)

# List files in the subject directory
files = os.listdir(subject_dir)
print("Files in subject:", files)

# Find the segmentation file (assumes filename contains 'seg' and ends with .nii.gz)
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

# Choose a modality for visualization: For example, T2F (or change to 'flair' as needed)
modality_file = None
for f in files:
    # Here we assume 't2f' appears in the file name for the T2 modality
    if 't2f' in f.lower() and f.endswith('.nii.gz'):
        modality_file = os.path.join(subject_dir, f)
        break
if modality_file is None:
    raise ValueError("T2F modality file not found in subject directory: " + subject_dir)
print("Using modality file:", modality_file)

# Load the modality image (for visualization)
mod_img = nib.load(modality_file)
mod_data = mod_img.get_fdata()

# Define fixed slice indices to visualize (only tumor slices)
tumor_slice_indices = [82, 63, 74]

# Plot the selected tumor slices with bounding boxes and score
num_plots = len(tumor_slice_indices)
fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
if num_plots == 1:
    axes = [axes]  # ensure axes is iterable

for i, slice_index in enumerate(tumor_slice_indices):
    # Extract the modality image slice and corresponding segmentation slice
    img_slice = mod_data[:, :, slice_index]
    seg_slice = seg_data[:, :, slice_index]
    
    # Display the image slice
    axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[i].set_title("Slice {}".format(slice_index))
    axes[i].axis('off')
    
    # Compute bounding box if tumor pixels exist
    tumor_pixels = np.where(seg_slice > 0)
    if tumor_pixels[0].size > 0:
        y_min, y_max = np.min(tumor_pixels[0]), np.max(tumor_pixels[0])
        x_min, x_max = np.min(tumor_pixels[1]), np.max(tumor_pixels[1])
        # Increase x_max and y_max by 1 to cover the end pixel (if desired)
        bbox = (x_min, y_min, x_max, y_max)
        width = x_max - x_min
        height = y_max - y_min

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                                 edgecolor='red', facecolor='none')
        # Add the patch to the Axes
        axes[i].add_patch(rect)
        
        # Compute the bounding box accuracy score
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

