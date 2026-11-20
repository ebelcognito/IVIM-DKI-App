"""IDTV-AD fitting engine used by the IVIM-DKI app.

Runs the final IDTV-AD method: bounded damped Gauss-Newton with a backtracking line search, free voxel-wise S0
(variable projection), per-parameter TV on log-parameters (Charbonnier, eps = 0.05) in voxel units, inside a
fitting mask. Defaults are the published settings.
"""
import os
import time

import numpy as np

os.environ.setdefault('IVIM_FREE_S0', '1')       # read once by idtv_revised_pp at import
import jax
import jax.numpy as jnp

import fixed_checkpoint_backend_pp as backend
from body_envelope_mask import body_envelope_mask
from otsu_log import log_otsu_mask

NAMES = ('D', 'D*', 'f', 'k')
PAPER_WEIGHTS = (0.01, 0.0035, 0.01, 0.0017)
PAPER_ITERATIONS = 400
PAPER_EPSILON = 0.05
PAPER_UPPER = (0.005, 0.3, 1.0, 3.0)
LOWER = {'Clinical (D* floor 0.008)': (0.0001, 0.008, 0.001, 0.01),
         'Simulation (D* floor 0.0001)': (0.0001, 0.0001, 0.001, 0.01)}
INITIAL = {'Simulation': (0.0013, 0.013, 0.23, 1.1),
           'Liver (HCC)': (0.0012, 0.019, 0.25, 0.8),
           'Pancreas (PDA)': (0.0013, 0.013, 0.3, 0.3),
           'Prostate (PC)': (0.00073, 0.013, 0.1, 1.19),
           'Lymph node (HL)': (0.0012, 0.013, 0.3, 0.7)}


def valid_voxels(data):
    """Voxels with a finite signal at every b-value and a positive b = 0 signal."""
    return np.isfinite(data).all(-1) & (data[..., 0] > 0)


def body_mask(data, spacing):
    """Automatic body mask of the paper: log-Otsu seed on the b = 0 image, region growing,
    3-mm in-plane closing and hole filling, restricted to valid voxels."""
    b0 = np.asarray(data[..., 0], float)
    seed, meta = log_otsu_mask(b0, bins=256)
    body, _ = body_envelope_mask(b0, seed['filled'], spacing, meta['threshold_raw'],
                                 meta['scale_positive_median'], radius_mm=3.)
    return body['body_envelope'] & valid_voxels(data)


def tv_spacing(spacing, voxel_units=True):
    """In-plane differences in voxel units; through-plane scaled by slice thickness / pixel size."""
    sp = [float(h) for h in spacing[:3]]
    return [h / min(sp[:2]) for h in sp] if voxel_units else sp


def fit(data, b_values, mask, spacing, initial, lower, upper, weights=PAPER_WEIGHTS,
        iterations=PAPER_ITERATIONS, epsilon=PAPER_EPSILON, voxel_units=True, precision='float32',
        progress=None, cancel=None):
    """Fit the IVIM-DKI parameters inside `mask`. Returns (maps[X,Y,Z,4] with NaN outside the mask, info).

    progress(done, total, cost) is called after every block of 25 iterations; setting the threading.Event
    `cancel` stops the fit after the current block."""
    started = time.perf_counter()
    data = np.asarray(data, float)
    b = np.asarray(b_values, float)
    if data.ndim != 4:
        raise ValueError(f'Need 4D data (x, y, z, b-value); got shape {data.shape}.')
    if len(b) != data.shape[3]:
        raise ValueError(f'{len(b)} b-values given but the data has {data.shape[3]} volumes.')
    if b[0] != 0 or (b < 0).any():
        raise ValueError('The first b-value must be 0 (and none negative).')
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (4,) or upper.shape != (4,) or (lower <= 0).any() or (upper <= lower).any():
        raise ValueError('Bounds need four positive values with lower < upper.')
    initial = np.clip(np.asarray(initial, float), lower, upper)
    weights = np.asarray(weights, float)
    if weights.shape != (4,) or (weights < 0).any():
        raise ValueError('TV weights need four non-negative values.')
    mask = np.asarray(mask, bool) & valid_voxels(data)
    if not mask.any():
        raise ValueError('The fitting mask contains no valid voxels.')

    y = data[mask] / data[mask][:, :1]
    with jax.enable_x64(precision == 'float64'):
        dt = np.float64 if precision == 'float64' else np.float32
        _, _, advance = backend.kernels(mask, tuple(tv_spacing(spacing, voxel_units)), epsilon=float(epsilon),
                                        batch_size=32768)
        x = jnp.asarray(np.broadcast_to(np.log(initial), (len(y), 4)).astype(dt))
        damping = jnp.asarray(dt(.001))
        args = (jnp.asarray(y.astype(dt)), jnp.asarray(b.astype(dt)), jnp.asarray(weights.astype(dt)),
                jnp.log(jnp.asarray(lower.astype(dt))), jnp.log(jnp.asarray(upper.astype(dt))))
        compiled = {}
        done, status, cost = 0, 'completed', float('nan')
        while done < iterations:
            n = min(25, iterations - done)
            if n not in compiled:
                compiled[n] = advance.lower(x, damping, *args, steps=n).compile()
            x, damping, count, history, failed = compiled[n](x, damping, *args)
            history = np.asarray(history)
            done += int(count)
            cost = float(history[int(count), 0])
            if progress:
                progress(done, iterations, cost)
            if bool(failed):
                status = 'stopped early: the cost could not be lowered further at this precision'
                break
            if cancel is not None and cancel.is_set():
                status = 'cancelled'
                break
        params = np.clip(np.exp(np.asarray(x, np.float64)), lower, upper)
    maps = np.full(mask.shape + (4,), np.nan)
    maps[mask] = params
    return maps, dict(iterations=done, status=status, cost=cost, seconds=time.perf_counter() - started,
                      voxels=int(mask.sum()), backend=jax.default_backend(), precision=precision)
