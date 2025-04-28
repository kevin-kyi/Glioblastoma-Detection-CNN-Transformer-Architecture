import argparse
import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go


def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine


def make_mesh(mask, level=0.5):
    # Pad by 1 to ensure closed surface at volume boundaries
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    verts, faces, normals, values = measure.marching_cubes(padded, level=level)
    # Remove padding offset
    verts -= 1
    return verts, faces


def compute_aspect_ranges(shape, affine):
    # corners of the volume in voxel indices
    corners = np.array([
        [0, 0, 0],
        [shape[0], 0, 0],
        [0, shape[1], 0],
        [0, 0, shape[2]],
        [shape[0], shape[1], shape[2]]
    ])
    # transform to world coords
    world_corners = nib.affines.apply_affine(affine, corners)
    mins = world_corners.min(axis=0)
    maxs = world_corners.max(axis=0)
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])


def plot_mesh(verts, faces, aspect=None):
    x, y, z = verts.T
    i, j, k = faces.T

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='red',
        opacity=0.5,
        name='Enhancing Tumor'
    )

    scene = dict(
        xaxis=dict(title='X (mm)'),
        yaxis=dict(title='Y (mm)'),
        zaxis=dict(title='Z (mm)'),
        aspectmode='data'
    )

    if aspect is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = aspect
        scene['xaxis']['range'] = [xmin, xmax]
        scene['yaxis']['range'] = [ymin, ymax]
        scene['zaxis']['range'] = [zmin, zmax]

    fig = go.Figure([mesh])
    fig.update_layout(
        scene=scene,
        title='3D Enhancing Tumor (to scale)'
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', required=True,
                        help='Path to BraTS segmentation NIfTI (contains label for tumor)')
    parser.add_argument('--label', type=int, default=3,
                        help='Segmentation label for enhancing tumor (default=3)')
    parser.add_argument('--mri', required=False,
                        help='(Optional) Path to MRI scan NIfTI (for full-volume extents)')
    args = parser.parse_args()

    # load segmentation
    seg_vol, seg_aff = load_nifti(args.seg)

    # binary mask of enhancing tumor
    mask = (seg_vol == args.label).astype(np.uint8)
    if mask.sum() == 0:
        print(f"No voxels found for label {args.label}!")
        return

    # mesh extraction
    verts, faces = make_mesh(mask, level=0.5)

    # apply affine -> real-world coords
    verts_world = nib.affines.apply_affine(seg_aff, verts)

    # compute axis ranges if MRI provided
    aspect = None
    if args.mri:
        mri_vol, mri_aff = load_nifti(args.mri)
        aspect = compute_aspect_ranges(mri_vol.shape, mri_aff)

    # plot
    plot_mesh(verts_world, faces, aspect)


if __name__ == '__main__':
    main()
    # python src/enhancing_tumor_visualizer.py --seg data/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz --mri data/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz --label 3