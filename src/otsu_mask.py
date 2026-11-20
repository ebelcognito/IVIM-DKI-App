"""Reproducible volume-wide Otsu candidates for anatomical fitting masks."""
import numpy as np
from scipy import ndimage as ndi


def largest_component(binary):
    """Keep the largest 3D six-connected component; first label wins ties."""
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 3:
        raise ValueError('Require a 3D mask.')
    labels, count = ndi.label(binary, structure=ndi.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(binary.shape, bool), 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes)), int(count)


def otsu_body_mask(b0, bins=256):
    """Threshold all finite b0 values, including zeros, with no intensity clipping.

    Use 256 uniform bins over the finite min/max by default. Maximize between-
    class variance of bin centers. The lowest maximizing split wins ties.
    Return the split's upper bin edge so thresholding agrees with the histogram
    partition. No morphology, resampling, ROI guidance, or intensity smoothing.
    """
    b0 = np.asarray(b0, dtype=np.float64)
    if b0.ndim != 3 or not isinstance(bins, int) or bins < 2:
        raise ValueError('Require a 3D b0 image and at least two histogram bins.')
    values = b0[np.isfinite(b0)]
    if not len(values) or values.min() == values.max():
        raise ValueError('Otsu requires finite intensities with a nonzero range.')
    counts, edges = np.histogram(values, bins=bins, range=(values.min(), values.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    probabilities = counts.astype(np.float64) / counts.sum()
    weight = np.cumsum(probabilities)[:-1]
    mean = np.cumsum(probabilities * centers)[:-1]
    total_mean = np.dot(probabilities, centers)
    eligible = (weight > 0) & (weight < 1)
    score = np.full(bins-1, -np.inf)
    score[eligible] = (total_mean * weight[eligible] - mean[eligible])**2 / (
        weight[eligible] * (1-weight[eligible]))
    if not eligible.any():
        raise ValueError('Histogram has no split with two nonempty classes.')
    split = int(np.argmax(score))
    threshold = float(edges[split+1])
    foreground = np.isfinite(b0) & (b0 >= threshold)
    # np.histogram assigns exact internal edges to the upper bin: use >=.
    mask, components = largest_component(foreground)
    return mask, dict(threshold=threshold, comparison='b0 >= threshold', bins=bins,
                      finite_min=float(values.min()), finite_max=float(values.max()),
                      finite_voxels=int(len(values)), nonfinite_voxels=int(b0.size-len(values)),
                      zero_voxels=int(np.count_nonzero(values == 0)),
                      threshold_foreground_voxels=int(foreground.sum()),
                      kept_voxels=int(mask.sum()), components=components,
                      removed_component_voxels=int(foreground.sum()-mask.sum()),
                      connectivity=6, histogram_counts=counts.tolist(),
                      histogram_edges=edges.tolist(), split_index=split)
