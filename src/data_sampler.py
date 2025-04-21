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
        # self.subject_dirs = self.subject_dirs[:10]
        # print(f"# of Subject Dirs: {len(self.subject_dirs)}")
        
        # Collect samples from each subject.
        self.samples = []
        for subject_path in self.subject_dirs:
            print("***************************************")
            print(f"GETTING SUBJECT SAMPLE: {subject_path}")
            print("***************************************")
            subject_samples = self._get_subject_samples(subject_path)
            self.samples.extend(subject_samples)
        print(f"Total samples in dataset: {len(self.samples)}")

    def _get_subject_samples(self, subject_path):
        """
        Each subject is a 3D volumetric representation. Here are sampling all "3" labeled segments and a corresponding number of non
        tumor regions
        """
        files = os.listdir(subject_path)
        seg_file = None
        for f in files:
            if 'seg' in f.lower() and f.endswith('.nii.gz'):
                seg_file = os.path.join(subject_path, f)
                break
        if seg_file is None:
            raise ValueError(f"Segmentation file not found in subject directory: {subject_path}")

        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        num_slices = seg_data.shape[2]

        positive_slices = []
        negative_slices = []
        for i in range(num_slices):
            seg_slice = seg_data[:, :, i]
            if np.any(seg_slice == self.positive_label):
                positive_slices.append(i)
            else:
                negative_slices.append(i)
        if len(positive_slices) > 0:
            num_pos = len(positive_slices)
            negative_sampled = random.sample(negative_slices, min(num_pos, len(negative_slices)))
        else:
            negative_sampled = []

        subject_samples = []
        for i in positive_slices:
            subject_samples.append((subject_path, i, True))
        for i in negative_sampled:
            subject_samples.append((subject_path, i, False))
        return subject_samples

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
        print(f"Get_Item of Index: {idx}")
        subject_path, slice_index, is_positive = self.samples[idx]

        # Load each modality slice.
        images = []
        for mod in self.modalities:
            mod_data = self._load_modality_image(subject_path, mod)
            slice_img = mod_data[:, :, slice_index]
            images.append(slice_img.astype(np.float32))

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
        if random.random() > 0.5:
            images = [np.fliplr(im) for im in images]
            seg = np.fliplr(seg)
        angle = random.uniform(-10, 10)  # Example: placeholder for rotation.
        sample['image'] = images
        sample['seg'] = seg
        return sample




if __name__ == '__main__':
    # Construct path relative to the current file's directory.
    train_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'BraTS2025-GLI-PRE-Challenge-TrainingData'
    )
    dataset = BraTSSliceDataset(train_data_path, transform=None)
    print("Total slices sampled:", len(dataset))
    
    # Visualize one sample for testing.
    import matplotlib.pyplot as plt
    sample = dataset[5]
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



# 
def _get_subject_samples(self, subject_path):
    """
    Each subject is a 3D volume. We stop once we've collected up to 30 tumor slices
    and up to 30 healthy slices, to save compute.
    """
    # locate segmentation file
    files = os.listdir(subject_path)
    seg_file = next((os.path.join(subject_path, f)
                     for f in files if 'seg' in f.lower() and f.endswith('.nii.gz')), None)
    if seg_file is None:
        raise ValueError(f"Segmentation file not found in {subject_path}")

    seg_data = nib.load(seg_file).get_fdata()
    num_slices = seg_data.shape[2]

    positive_slices = []
    negative_slices = []

    for i in range(num_slices):
        seg_slice = seg_data[:, :, i]
        if np.any(seg_slice == self.positive_label):
            if len(positive_slices) < 5:
                positive_slices.append(i)
        else:
            if len(negative_slices) < 5:
                negative_slices.append(i)
        # break if both lists are full
        if len(positive_slices) >= 5 and len(negative_slices) >= 5:
            break

    # Build sample tuples
    subject_samples = [(subject_path, i, True)  for i in positive_slices] + \
                      [(subject_path, i, False) for i in negative_slices]
    return subject_samples
