"""Experimental background separation using intensity-compressed Otsu."""
import numpy as np
from scipy import ndimage as ndi
from otsu_mask import otsu_body_mask, largest_component

def log_otsu_mask(b0, bins=256):
    b0=np.asarray(b0,dtype=float)
    positive=b0[np.isfinite(b0)&(b0>0)]
    if b0.ndim!=3 or not positive.size: raise ValueError('Require positive finite 3D signal')
    scale=float(np.median(positive))
    transformed=np.where(np.isfinite(b0),np.log1p(np.maximum(b0,0)/scale),np.nan)
    _,meta=otsu_body_mask(transformed,bins=bins)
    threshold=float(scale*np.expm1(meta['threshold']))
    raw=np.isfinite(b0)&(b0>=threshold)&(b0>0)
    connected,components=largest_component(raw)
    filled=np.stack([ndi.binary_fill_holes(connected[:,:,z]) for z in range(b0.shape[2])],axis=2)
    filled &= np.isfinite(b0)&(b0>0)
    stages=dict(threshold_only=raw,largest_component=connected,filled=filled)
    meta.update(thresholds=[threshold],threshold_raw=threshold,scale_positive_median=scale,
        transform='log1p(max(b0,0)/median_positive_b0)',components=components,
        stage_voxels={k:int(v.sum()) for k,v in stages.items()},candidate_only=True,
        parameter_fitting_performed=False)
    return stages,meta
