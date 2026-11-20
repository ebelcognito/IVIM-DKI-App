"""IDTV-AD: bounded IVIM-DKI fitting with an explicit masked log-TV objective.

Per-parameter TV weights: lambda_tv may be a scalar or a 4-vector
[D, Dstar, f, K]; the TV energy is kept per parameter and weighted.

Not a replica of the legacy MATLAB SimHyHYtv trajectory. The solver uses
projected damped Gauss-Newton directions, TV diagonal preconditioning, Armijo
step acceptance, and a projected-gradient stopping criterion.
"""
from functools import lru_cache, wraps
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

# Double precision is scoped to this fitter, so importing it does not change
# the float32 behavior of the existing Adam implementation in the same process.
def _double_precision(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with jax.enable_x64(True):
            return function(*args, **kwargs)
    return wrapped

INITIAL = np.array([.0013, .013, .23, 1.1])
LOWER = np.array([.0001, .0001, .001, .01])
UPPER = np.array([.05, .5, 1., 3.])


def log_signal(x, b):
    d, dp, f, k = jnp.exp(x)
    phase = -b * d + (b * d)**2 * k / 6
    # Invalid overflowing trial points are rejected by the objective. Bounding
    # the trial exponential avoids NaNs before that rejection, including 0*Inf.
    slow = jnp.exp(jnp.minimum(phase, 700.))
    return f * jnp.exp(-b * dp) + (1 - f) * slow


# Optional free per-voxel signal scale (S0): IVIM_FREE_S0=1. The optimal scale for the
# current parameters is computed in closed form (variable projection), so the fit is no
# longer forced through the single noisy b0 sample. No extra parameter is regularized.
FREE_S0 = os.environ.get('IVIM_FREE_S0') == '1'


def model_signal(x, b, y):
    m = log_signal(x, b)
    if FREE_S0:
        m = m * (jnp.sum(m * y) / jnp.sum(m * m))
    return m


def tv_terms(q, mask, spacing, epsilon):
    """Forward differences and their exact adjoint; no mask/background edges.

    Spacing is in array-axis order (NIfTI X,Y,Z), not MATLAB meshgrid order.
    Returns energy, exact gradient, and a positive diagonal preconditioner.
    """
    differences = []
    square = jnp.zeros_like(q)
    for axis, h in enumerate(spacing):
        a = [slice(None)] * 3
        c = a.copy()
        a[axis], c[axis] = slice(None, -1), slice(1, None)
        edge = mask[tuple(a)] & mask[tuple(c)]
        delta = jnp.where(edge[..., None], jnp.diff(q, axis=axis) / h, 0.)
        pads = [(0, 0)] * 4
        pads[axis] = (0, 1)
        square += jnp.pad(delta * delta, pads)
        differences.append((axis, h, tuple(a), tuple(c), edge, delta))
    denominator = jnp.sqrt(square + epsilon**2)
    energy = jnp.sum(denominator - epsilon, axis=(0, 1, 2))  # per parameter
    gradient, diagonal = jnp.zeros_like(q), jnp.zeros_like(q)
    for axis, h, a, c, edge, delta in differences:
        weight = jnp.where(edge[..., None], 1. / denominator[a], 0.)
        flux = delta * weight / h
        left_pads, right_pads = [(0, 0)] * 4, [(0, 0)] * 4
        left_pads[axis], right_pads[axis] = (0, 1), (1, 0)
        gradient += jnp.pad(-flux, left_pads) + jnp.pad(flux, right_pads)
        diagonal += jnp.pad(weight / h**2, left_pads) + jnp.pad(weight / h**2, right_pads)
    return energy, gradient, diagonal


@lru_cache(maxsize=16)
def _kernels(shape, mask_bytes, spacing, epsilon):
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    indices = jnp.asarray(np.flatnonzero(mask.ravel()))
    domain = jnp.asarray(mask)
    lookup = np.full(mask.size, int(mask.sum()), np.int32)
    lookup[mask.ravel()] = np.arange(mask.sum(), dtype=np.int32)
    lookup = jnp.asarray(lookup)

    def scatter(x):
        # Dense gather avoids CUDA scatter kernels unsupported by some older
        # GPUs. The mathematical grid assembly is identical.
        if mask.all():
            return x.reshape(shape + (4,))
        extended = jnp.concatenate((x, jnp.zeros((1, 4), dtype=x.dtype)))
        return extended[lookup].reshape(shape + (4,))

    def residual(x, b, y):
        return model_signal(x, b, y) - y

    signals = jax.vmap(model_signal, in_axes=(0, None, 0))
    jacobian = jax.vmap(jax.jacfwd(residual, argnums=0), in_axes=(0, None, 0))

    @jax.jit
    def objective(x, y, b, strength):
        predicted = signals(x, b, y)
        d, _, _, k = jnp.exp(x).T
        phase = -d[:, None] * b + (d[:, None] * b)**2 * k[:, None] / 6
        energy = tv_terms(scatter(x), domain, spacing, epsilon)[0]
        value = .5 * jnp.sum((predicted - y)**2) + jnp.sum(strength * energy)
        return jnp.where(jnp.any(phase > 700.) | ~jnp.isfinite(value), jnp.inf, value)

    @jax.jit
    def derivatives(x, y, b, strength, damping):
        prediction = signals(x, b, y)
        r = prediction - y
        jac = jacobian(x, b, y)
        hessian = jnp.einsum("nbi,nbj->nij", jac, jac)
        gradient = jnp.einsum("nbi,nb->ni", jac, r)
        energy, tv_gradient, tv_diagonal = tv_terms(scatter(x), domain, spacing, epsilon)
        gradient += strength * tv_gradient.reshape((-1, 4))[indices]
        diagonal = jnp.diagonal(hessian, axis1=1, axis2=2)
        added = strength * tv_diagonal.reshape((-1, 4))[indices] + damping * (diagonal + .01)
        matrix = hessian + jnp.eye(4)[None] * added[:, None, :]
        direction = -jnp.linalg.solve(matrix, gradient[..., None])[..., 0]
        direction /= jnp.maximum(1., jnp.max(jnp.abs(direction), axis=1, keepdims=True))
        value = .5 * jnp.sum(r**2) + jnp.sum(strength * energy)
        return value, gradient, direction

    @jax.jit(static_argnames=("limit",))
    def compiled_run(x, y, b, strength, lo, hi, tolerance, *, limit):
        history = jnp.full((limit + 1, 3), jnp.nan)
        state = (jnp.asarray(0), x, jnp.asarray(.001), history, jnp.asarray(False), jnp.asarray(False))

        def condition(s):
            i, _, _, _, converged, failed = s
            return (i < limit) & ~converged & ~failed

        def iteration(s):
            i, x, damping, history, _, _ = s
            value, gradient, direction = derivatives(x, y, b, strength, damping)
            pg = jnp.max(jnp.abs(x - jnp.clip(x - gradient, lo, hi)))
            history = history.at[i].set(jnp.array([value, pg, damping]))
            converged = pg <= tolerance

            def update(_):
                line = (jnp.asarray(0), x, jnp.asarray(False))
                def more(z):return (z[0] < 25) & ~z[2]
                def trial(z):
                    count, previous, _ = z
                    candidate = jnp.clip(x + 2.**(-count) * direction, lo, hi)
                    slope = jnp.sum(gradient * (candidate - x))
                    trial_value = objective(candidate, y, b, strength)
                    accepted = (slope < 0) & jnp.isfinite(trial_value) & (trial_value <= value + 1e-4*slope)
                    return count+1, jnp.where(accepted, candidate, previous), accepted
                _, new_x, accepted = jax.lax.while_loop(more, trial, line)
                new_damping = jnp.where(accepted, jnp.maximum(1e-7, damping*.5), damping*10.)
                return new_x, new_damping, new_damping > 1e12
            new_x, new_damping, failed = jax.lax.cond(converged, lambda _: (x,damping,jnp.asarray(False)), update, operand=None)
            return i+1, new_x, new_damping, history, converged, failed

        count, x, damping, history, _, failed = jax.lax.while_loop(condition, iteration, state)
        value, gradient, _ = derivatives(x, y, b, strength, damping)
        pg = jnp.max(jnp.abs(x-jnp.clip(x-gradient,lo,hi)))
        history = history.at[count].set(jnp.array([value,pg,damping]))
        return x, count, history, value, pg, failed

    return objective, derivatives, compiled_run


@_double_precision
def fit_revised(data, b_values, mask, *, spacing=(1., 1., 1.),
                initial=INITIAL, lower=LOWER, upper=UPPER, lambda_tv=.001,
                epsilon=.05, max_iterations=500, gradient_tolerance=1e-7,
                execution="compiled", initial_map=None, prenormalized=False):
    """Return native-grid parameters (NaN outside mask) and fit diagnostics.

    No geometry inference or resampling is done. The caller must supply an
    anatomical mask and spacing in the array's spatial-axis order. Inputs
    inside the mask must be finite, with positive b0. Patient-specific truth
    or literature means are never used in the objective or stopping rule.
    """
    started = time.perf_counter()
    data = np.asarray(data, np.float64)
    domain = np.asarray(mask)
    b = np.asarray(b_values, np.float64)
    spacing = tuple(float(h) for h in spacing)
    initial, lower, upper = (np.asarray(v, np.float64) for v in (initial, lower, upper))
    if data.ndim != 4 or domain.shape != data.shape[:3] or not np.isin(domain, [0, 1]).all():
        raise ValueError("Require a binary mask on the same three-dimensional array grid.")
    domain = domain.astype(bool)
    if not domain.any() or b.shape != (data.shape[-1],) or b[0] != 0 or not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("Require a nonempty mask and finite nonnegative b-values with b0 first.")
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Require exactly three positive finite spacings in array-axis order.")
    if any(v.shape != (4,) or not np.isfinite(v).all() for v in (initial, lower, upper)) or (lower <= 0).any() or (upper <= lower).any() or (initial < lower).any() or (initial > upper).any():
        raise ValueError("Require four positive ordered bounds containing initialization.")
    lambda_tv = np.broadcast_to(np.asarray(lambda_tv, np.float64), (4,)).copy()
    if not np.isfinite(lambda_tv).all() or not np.isfinite([epsilon, gradient_tolerance]).all() or (lambda_tv < 0).any() or epsilon <= 0 or gradient_tolerance <= 0 or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("Invalid regularization, tolerance or iteration limit.")
    selected = data[domain]
    if prenormalized:  # Bregman targets: already b0-normalized; with free S0 no further scaling is needed
        if not np.isfinite(selected).all():
            raise ValueError("Masked signals must be finite.")
        y = selected.copy()
    else:
        if not np.isfinite(selected).all() or (selected[:, 0] <= 0).any():
            raise ValueError("Masked signals must be finite with positive b0; invalid signals are not replaced.")
        y = selected / selected[:, :1]
    if not np.isfinite(y).all():
        raise ValueError("Signal normalization overflow.")
    lo, hi = np.log(lower), np.log(upper)
    x = np.broadcast_to(np.log(initial), (len(y), 4)).copy()
    if initial_map is not None:  # per-voxel start on the grid (X,Y,Z,4), clipped to bounds
        x = np.log(np.clip(np.asarray(initial_map, np.float64)[domain], lower, upper))
    objective, derivatives, compiled_run = _kernels(domain.shape, domain.tobytes(), spacing, float(epsilon))
    yj, bj = jnp.asarray(y), jnp.asarray(b)
    if not np.isfinite(float(objective(x, yj, bj, jnp.asarray(lambda_tv)))):
        raise ValueError("Initial parameters give a nonfinite model/objective.")
    history, damping, status = [], .001, "iteration_limit"
    if execution not in ("compiled", "host"):
        raise ValueError("execution must be compiled or host.")
    if execution == "compiled":
        x, count, history, final_value, final_pg, failed = (np.asarray(v) for v in compiled_run(
            jnp.asarray(x), yj, bj, jnp.asarray(lambda_tv), jnp.asarray(lo), jnp.asarray(hi),
            gradient_tolerance, limit=int(max_iterations)))
        count = int(count)
        history = history[:count+1]
        status = "projected_gradient_converged" if float(final_pg)<=gradient_tolerance else "step_rejected" if failed else "iteration_limit"
        iteration = count-1
    for iteration in range(int(max_iterations)) if execution == "host" else ():
        value, gradient, direction = (np.asarray(v) for v in derivatives(x, yj, bj, lambda_tv, damping))
        value = float(value)
        projected_gradient = float(np.max(np.abs(x - np.clip(x - gradient, lo, hi))))
        history.append([value, projected_gradient, damping])
        if projected_gradient <= gradient_tolerance:
            status = "projected_gradient_converged"
            break
        accepted = False
        for backtrack in range(25):
            candidate = np.clip(x + 2.**(-backtrack) * direction, lo, hi)
            slope = float(np.sum(gradient * (candidate - x)))
            if slope < 0:
                trial = float(objective(candidate, yj, bj, lambda_tv))
                if np.isfinite(trial) and trial <= value + 1e-4 * slope:
                    x, damping, accepted = candidate, max(1e-7, damping * .5), True
                    break
        if not accepted:
            damping *= 10.
            if damping > 1e12:
                status = "step_rejected"
                break
    if execution == "host":
        final_value, final_gradient, _ = (np.asarray(v) for v in derivatives(x, yj, bj, lambda_tv, damping))
        final_pg = float(np.max(np.abs(x - np.clip(x - final_gradient, lo, hi))))
        if final_pg <= gradient_tolerance:
            status = "projected_gradient_converged"
        history.append([float(final_value), final_pg, damping])
    maps = np.full(domain.shape + (4,), np.nan)
    maps[domain] = np.exp(x)
    predicted = np.asarray(jax.vmap(model_signal, in_axes=(0, None, 0))(jnp.asarray(x), bj, yj))
    rss = np.full(domain.shape, np.nan)
    rss[domain] = np.sum((predicted - y)**2, axis=1)
    return maps, dict(status=status, iterations=iteration + 1, objective=float(final_value),
                      projected_gradient=float(final_pg), history=np.asarray(history), residual_sse=rss,
                      seconds=time.perf_counter() - started, backend=jax.default_backend(),
                      execution=execution, lambda_tv=lambda_tv.tolist(), algorithm="IDTV-Revised v1; distinct from legacy IDTV")
