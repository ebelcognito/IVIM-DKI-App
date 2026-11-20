"""Experimental Otsu-seeded body envelope, independent of measurement ROIs."""
import numpy as np
from scipy import ndimage as ndi
from otsu_mask import otsu_body_mask

def body_envelope_mask(b0, old_mask, spacing, high_threshold, scale, radius_mm=3.):
    b0=np.asarray(b0,float);old=np.asarray(old_mask,bool)
    if b0.ndim!=3 or old.shape!=b0.shape:raise ValueError('Require matching 3D data/mask')
    finite=np.isfinite(b0);t=np.where(finite,np.log1p(np.maximum(b0,0)/scale),np.nan)
    low_class=finite&(b0<high_threshold)
    # Otsu's implementation needs a 3D container, not a spatially continuous image.
    # Only its threshold is used; component selection on this container is ignored.
    _,meta=otsu_body_mask(t[low_class].reshape(-1,1,1))
    low=float(scale*np.expm1(meta['threshold']))
    allowed=finite&(b0>=low)&(b0>0)
    grown=ndi.binary_propagation(old&allowed,mask=allowed,structure=ndi.generate_binary_structure(3,1))|old
    sx,sy=spacing[:2];rx,ry=np.ceil(radius_mm/np.array([sx,sy])).astype(int)
    x,y=np.ogrid[-rx:rx+1,-ry:ry+1];disk=(x*sx)**2+(y*sy)**2<=radius_mm**2
    final=np.zeros_like(old)
    for z in range(old.shape[2]):
        # Padding prevents artificial erosion at a truncated field-of-view edge.
        a=np.pad(grown[:,:,z],((rx,rx),(ry,ry)),mode='edge')
        a=ndi.binary_closing(a,structure=disk)[rx:-rx,ry:-ry]
        final[:,:,z]=ndi.binary_fill_holes(a|grown[:,:,z])
    final &= finite&(b0>0)
    return dict(grown=grown,body_envelope=final),dict(high_threshold=float(high_threshold),low_threshold=low,
        radius_mm=radius_mm,spacing=list(map(float,spacing)),connectivity=6,
        scale_positive_median=scale,seed='previous log-Otsu filled mask',
        rule='One Otsu split of the lower transformed-intensity class; seeded propagation; XY 3mm closing; XY hole filling',
        parameter_fitting_performed=False,candidate_only=True)
