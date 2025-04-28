import argparse
import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go


def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine


def extract_surface(vol, level):
    padded = np.pad(vol, 1, mode='constant', constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=level)
    verts -= 1
    return verts, faces


def compute_aspect_ranges(shape, affine):
    corners = np.array([
        [0,0,0],
        [shape[0],0,0],
        [0,shape[1],0],
        [0,0,shape[2]],
        [shape[0],shape[1],shape[2]]
    ])
    world = nib.affines.apply_affine(affine, corners)
    mins, maxs = world.min(axis=0), world.max(axis=0)
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])


def main():
    p = argparse.ArgumentParser(
        description="Overlay tumor mesh on brain surface"
    )
    p.add_argument('--mri', required=True,
                   help='Path to T1-CE NIfTI file')
    p.add_argument('--seg', required=True,
                   help='Path to segmentation NIfTI file')
    p.add_argument('--thr_pct', type=float, default=70,
                   help='Percentile for brain isosurface threshold')
    p.add_argument('--label', type=int, default=3,
                   help='Segmentation label for enhancing tumor')
    args = p.parse_args()

    # 1) Brain surface
    brain_vol, brain_aff = load_nifti(args.mri)
    level = np.percentile(brain_vol, args.thr_pct)
    verts_b, faces_b = extract_surface(brain_vol, level)
    verts_b_world = nib.affines.apply_affine(brain_aff, verts_b)

    # 2) Tumor surface from segmentation
    seg_vol, seg_aff = load_nifti(args.seg)
    mask = (seg_vol == args.label).astype(np.uint8)
    verts_t, faces_t = extract_surface(mask, level=0.5)
    verts_t_world = nib.affines.apply_affine(seg_aff, verts_t)

    # 3) Compute axis limits from brain MRI
    aspect = compute_aspect_ranges(brain_vol.shape, brain_aff)

    # 4) Plot both meshes
    brain_mesh = go.Mesh3d(
        x=verts_b_world[:,0], y=verts_b_world[:,1], z=verts_b_world[:,2],
        i=faces_b[:,0], j=faces_b[:,1], k=faces_b[:,2],
        opacity=0.3, color='lightgrey', name='Brain Surface'
    )
    tumor_mesh = go.Mesh3d(
        x=verts_t_world[:,0], y=verts_t_world[:,1], z=verts_t_world[:,2],
        i=faces_t[:,0], j=faces_t[:,1], k=faces_t[:,2],
        opacity=0.6, color='red', name='Enhancing Tumor'
    )

    fig = go.Figure([brain_mesh, tumor_mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[aspect[0],aspect[1]], title='X (mm)'),
            yaxis=dict(range=[aspect[2],aspect[3]], title='Y (mm)'),
            zaxis=dict(range=[aspect[4],aspect[5]], title='Z (mm)'),
            aspectmode='data'
        ),
        title='Brain Surface with Enhancing Tumor Overlay'
    )
    fig.show()

if __name__ == '__main__':
    main()