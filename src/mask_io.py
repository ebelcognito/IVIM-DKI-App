"""Load an anatomical fitting mask in the exact grid of an IVIM scan."""

import numpy as np
import nibabel as nib


def load_anatomical_mask(path, scan):
    """Return a binary mask after checking shape and voxel-to-world geometry."""
    mask_image = nib.load(path)
    if mask_image.shape != scan.shape[:3] or not np.allclose(
        mask_image.affine, scan.affine, atol=1e-3, rtol=0
    ):
        raise ValueError('Anatomical mask and scan must have the same shape and affine; no automatic resampling.')
    values = mask_image.get_fdata()
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError('Anatomical mask must contain only 0 and 1.')
    if not np.any(values):
        raise ValueError('Anatomical mask is empty.')
    return values.astype(bool)
