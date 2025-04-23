import os
import random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T  # Optional: for augmentations

class BraTSSliceDataset(Dataset):

    def __init__(self, data_dir, modalities=['t1c', 't1n', 't2f', 't2w'], positive_label=3, transform=None):
        self.data_dir = data_dir
        self.modalities = modalities
        self.positive_label = positive_label
        self.transform = transform

        # List subject directories from the given data_dir.
        self.subject_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, d))]


        # THIS IS FOR CREATING A SUBSET OF THE DATASET
        # self.subject_dirs = self.subject_dirs[4:10]
        self.subject_dirs = self.subject_dirs[0:10]

        print(f"# of Subject Dirs: {len(self.subject_dirs)}")
        
        # Collect samples from each subject.
        self.samples = []
        for subject_path in self.subject_dirs:
            print("***************************************")
            print(f"GETTING SUBJECT SAMPLE: {subject_path}")
            print("***************************************")
            subject_samples = self._get_subject_samples(subject_path)
            self.samples.extend(subject_samples)

        self.samples = [self.samples[4]]

        print(f"Total samples in dataset: {len(self.samples)}")


    def _get_subject_samples(self, subject_path):
        """
        Deterministically pair each new positive slice with the next healthy slice,
        up to sliceLimit of each.
        """
        # locate segmentation
        files    = os.listdir(subject_path)
        seg_file = next((f for f in files if 'seg' in f.lower() and f.endswith('.nii.gz')), None)
        if seg_file is None:
            raise ValueError(f"Segmentation file not found in {subject_path}")
        seg_data = nib.load(os.path.join(subject_path, seg_file)).get_fdata()
        num_slices = seg_data.shape[2]

        sliceLimit = 5
        pos_idxs, neg_idxs = [], []

        for i in range(num_slices):
            # stop early if we've got enough
            if len(pos_idxs) >= sliceLimit and len(neg_idxs) >= sliceLimit:
                break

            has_tumor = np.any(seg_data[:, :, i] == self.positive_label)

            # 1) if a new tumor slice
            if has_tumor and len(pos_idxs) < sliceLimit:
                pos_idxs.append(i)

                # 2) find the very next healthy slice
                for j in range(i+1, num_slices):
                    if not np.any(seg_data[:, :, j] == self.positive_label):
                        if len(neg_idxs) < sliceLimit:
                            neg_idxs.append(j)
                        break

        # Build the sample list
        samples = (
            [(subject_path, idx, True)  for idx in pos_idxs] +
            [(subject_path, idx, False) for idx in neg_idxs]
        )
        return samples


    def _load_modality_image(self, subject_path, modality_keyword):
        files = os.listdir(subject_path)
        modality_file = None
        for f in files:
            if modality_keyword in f.lower() and f.endswith('.nii.gz') and 'seg' not in f.lower():
                modality_file = os.path.join(subject_path, f)
                break
        if modality_file is None:
            raise ValueError(f"Modality file with keyword '{modality_keyword}' not found in {subject_path}")
        img = nib.load(modality_file)
        return img.get_fdata()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample with:
            'image': a list of 2D arrays (one for each modality)
            'seg': a 2D binary mask indicating the enhancing tumor regions
        """
        # print(f"Get_Item of Index: {idx}")
        subject_path, slice_index, is_positive = self.samples[idx]

        # Load each modality slice.
        images = []
        for mod in self.modalities:
            mod_data = self._load_modality_image(subject_path, mod)
            slice_img = mod_data[:, :, slice_index]
            # NEW NORMALIZATION STEP
            p1, p99 = np.percentile(slice_img, (1, 99))
            img = np.clip(slice_img, p1, p99)

            mn, mx = img.min(), img.max()
            if mx - mn > 1e-6:
                img = (img - mn) / (mx - mn)

            images.append(img)


        # Load segmentation slice and create a binary mask for the enhancing tumor.
        files = os.listdir(subject_path)
        seg_file = None
        for f in files:
            if 'seg' in f.lower() and f.endswith('.nii.gz'):
                seg_file = os.path.join(subject_path, f)
                break
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        seg_slice = seg_data[:, :, slice_index]
        binary_mask = (seg_slice == self.positive_label).astype(np.float32)

        sample = {'image': images, 'seg': binary_mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Example data augmentation transform placeholder.
class SimpleAugment(object):
    def __call__(self, sample):
        images, seg = sample['image'], sample['seg']
        
        if random.random() < 0.5:
            images = [np.fliplr(im).copy() for im in images]
            seg    = np.fliplr(seg).copy()

        if random.random() < 0.5:
            images = [np.flipud(im).copy() for im in images]
            seg    = np.flipud(seg).copy()
        
        # 4) Random intensity scaling (brightness)
        scale = random.uniform(0.9, 1.1)
        images = [im * scale for im in images]
        
        # 5) Add small Gaussian noise
        noise_sigma = 0.01
        noise = np.random.normal(0, noise_sigma, size=images[0].shape)
        images = [im + noise for im in images]

        # Make sure we stay float32
        sample['image'] = [im.astype(np.float32) for im in images]
        sample['seg']   = seg             # segmentation is binary, no change
        
        return sample





if __name__ == '__main__':
    # Construct path relative to the current file's directory.
    train_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'BraTS2025-GLI-PRE-Challenge-TrainingData'
    )
    dataset = BraTSSliceDataset(train_data_path, transform=SimpleAugment())
    print("Total slices sampled:", len(dataset))
    
    # Visualize one sample for testing.
    import matplotlib.pyplot as plt
    sample = dataset[0]
    # sample = dataset[4]

    modalities = sample['image']
    seg = sample['seg']
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(modalities[0].T, cmap='gray', origin='lower')
    plt.title("Modality: " + dataset.modalities[0])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(seg.T, cmap='gray', origin='lower')
    plt.title("Enhancing Tumor Mask")
    plt.axis('off')
    plt.show()




