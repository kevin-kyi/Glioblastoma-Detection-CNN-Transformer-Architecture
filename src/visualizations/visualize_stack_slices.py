import os, argparse
import nibabel as nib
import numpy as np
import torch
from skimage import measure
import plotly.graph_objects as go

from unet import MultiModalSegModel

MODS = ['t1c','t1n','t2f','t2w']

def make_mesh(mask, level=0.5):
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=level)
    verts -= 1
    return verts, faces

def compute_aspect_ranges(shape, affine):
    corners = np.array([
        [0,0,0],
        [shape[0],0,0],
        [0,shape[1],0],
        [0,0,shape[2]],
        [shape[0],shape[1],shape[2]],
    ])
    world = nib.affines.apply_affine(affine, corners)
    mins, maxs = world.min(axis=0), world.max(axis=0)
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

def plot_mesh(verts, faces, aspect):
    x,y,z = verts.T
    i,j,k = faces.T
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='red', opacity=0.5, name='Tumor'
    )
    fig = go.Figure([mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[aspect[0],aspect[1]], title='X (mm)'),
            yaxis=dict(range=[aspect[2],aspect[3]], title='Y (mm)'),
            zaxis=dict(range=[aspect[4],aspect[5]], title='Z (mm)'),
            aspectmode='data'
        ),
        title="3D Predicted Enhancing Tumor (to scale)"
    )
    fig.show()

def load_model(ckpt_path, device):
    model = MultiModalSegModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def load_nifti(subject_dir, keyword):
    fn = next(f for f in os.listdir(subject_dir)
              if keyword in f.lower() and f.endswith('.nii.gz'))
    img = nib.load(os.path.join(subject_dir, fn))
    return img.get_fdata(), img.affine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize 3D tumor mesh using only positive T1-CE slices"
    )
    parser.add_argument('--subject',    required=True,
                        help='BraTS subject folder')
    parser.add_argument('--checkpoint', required=True,
                        help='trained model .pth checkpoint')
    parser.add_argument('--mri',        required=True,
                        help='full-brain MRI NIfTI for scaling')
    parser.add_argument('--label',      type=int, default=3,
                        help='Segmentation label for enhancing tumor')
    parser.add_argument('--thresh',     type=float, default=0.5,
                        help='Probability threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint, device)

    # 1) locate positive T1-CE slices
    seg_vol, _ = load_nifti(args.subject, 'seg')
    pos_slices = np.where(np.any(seg_vol == args.label, axis=(0,1)))[0]
    print("Positive slice indices:", pos_slices)

    # 2) load the MRI directly (no search in subject_dir)
    mri_img    = nib.load(args.mri)
    mri_vol    = mri_img.get_fdata()
    mri_affine = mri_img.affine
    H, W, D    = mri_vol.shape
    pred_vol   = np.zeros((H, W, D), dtype=np.uint8)

    # 3) load all four modality volumes once
    vols = {}
    for m in MODS:
        vols[m], _ = load_nifti(args.subject, m)

    # 4) infer on each positive slice and place at true z-index
    for z in pos_slices:
        inp = {}
        for m in MODS:
            sl = vols[m][:,:,z]
            p1,p99 = np.percentile(sl, (1,99))
            clipped = np.clip(sl, p1, p99)
            mn,mx = clipped.min(), clipped.max()
            norm = (clipped - mn)/(mx-mn) if mx>mn else clipped
            inp[m] = torch.from_numpy(norm[None,None,...]).float().to(device)

        with torch.no_grad():
            outs = model(inp)
            prob = torch.sigmoid(outs['t1c'])[0,0].cpu().numpy()
            mask = (prob > args.thresh).astype(np.uint8)

        pred_vol[:,:,z] = mask
        print(f"Slice {z}: predicted voxels = {mask.sum()}")

    print("Total predicted voxels:", pred_vol.sum())

    # 5) build mesh and transform vertices into MRI (mm) space
    verts, faces = make_mesh(pred_vol, level=args.thresh)
    verts_world  = nib.affines.apply_affine(mri_affine, verts)
    aspect       = compute_aspect_ranges(mri_vol.shape, mri_affine)
    plot_mesh(verts_world, faces, aspect)

