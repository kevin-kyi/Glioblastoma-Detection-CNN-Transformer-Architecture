# Glioblastoma-Detection-UNET-Transformer-Architecture: GBM Segmentation & 3D Reconstruction

## Introduction
Glioblastoma (GBM) is a highly malignant brain tumor with poor prognosis and high recurrence.  Manual MRI interpretation is time‐consuming and variable, so we built a fully automated pipeline that:
1. Ingests four MRI modalities (T1-non-contrast, T1-contrast, T2-weighted, FLAIR).  
2. Runs each slice through a modality-specific U-Net, fuses features with a lightweight Transformer, and segments the enhancing tumor region in 2D.  
3. Stacks the per-slice masks into a 3D volume and renders an interactive surface mesh in Plotly for volumetric evaluation.  

---

## Dataset
We use the [BraTS 2025 Challenge data on Synapse](https://www.synapse.org/Synapse:syn53708249/wiki/626323):
1. Sign up for a Synapse account  
2. Register for “BraTS 2025 Challenge”  
3. Complete the Data Access form  
4. Download the four‐modality training scans and segmentation labels  

---

## Repository Structure & Storage Note
- **Due to GitHub storage limits**, we **do not** include the raw MRI data or any pretrained `.pth` files in this repo.  
- Once you’ve downloaded the BraTS data, you can launch training using visualize_stack_slices.py
![Test Image 1](https://github.com/kevin-kyi/Glioblastoma-Detection-UNET-Transformer-Architecture/blob/main/results/back_view.jpg)
