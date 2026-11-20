"""Memory-bounded execution of IDTV-AD with fixed-count checkpoints.

Per-parameter TV weights: strength may be a scalar or a 4-vector.
Only voxel-local signal/Jacobian/linear-solve operations are batched. TV and
line-search acceptance cover the entire supplied mask at every iteration.
"""
import jax
import jax.numpy as jnp
import numpy as np

from idtv_revised_pp import log_signal, model_signal, tv_terms  # per-parameter TV energy


def kernels(mask, spacing, *, epsilon=.05, batch_size=32768):
    mask = np.asarray(mask, bool)
    shape = mask.shape
    n = int(mask.sum())
    indices = jnp.asarray(np.flatnonzero(mask.ravel()))
    domain = jnp.asarray(mask)
    lookup = np.full(mask.size, n, np.int32)
    lookup[mask.ravel()] = np.arange(n, dtype=np.int32)
    lookup = jnp.asarray(lookup)

    def grid(x):
        if mask.all():
            return x.reshape(shape + (4,))
        extended = jnp.concatenate((x, jnp.zeros((1, 4), dtype=x.dtype)))
        return extended[lookup].reshape(shape + (4,))

    def residual(x, b, y):
        return model_signal(x, b, y) - y

    jacobian = jax.jacfwd(residual, argnums=0)

    def voxel_cost(args, b):
        x, y = args
        r = residual(x, b, y)
        d, _, _, k = jnp.exp(x)
        phase = -b*d + (b*d)**2*k/6
        value = .5*jnp.sum(r*r)
        return jnp.where(jnp.any(phase > 700.) | ~jnp.isfinite(value), jnp.inf, value)

    @jax.jit
    def objective(x, y, b, strength):
        costs = jax.lax.map(lambda args: voxel_cost(args, b), (x, y), batch_size=batch_size)
        energy = tv_terms(grid(x), domain, spacing, epsilon)[0]
        return jnp.sum(costs) + jnp.sum(strength*energy)

    @jax.jit
    def derivatives(x, y, b, strength, damping):
        energy, tvg, tvd = tv_terms(grid(x), domain, spacing, epsilon)
        tvg = tvg.reshape((-1, 4))[indices]
        tvd = tvd.reshape((-1, 4))[indices]

        def voxel(args):
            z, measured, vg, vd = args
            r = residual(z, b, measured)
            jac = jacobian(z, b, measured)
            hessian = jnp.einsum('bi,bj->ij', jac, jac)
            gradient = jnp.einsum('bi,b->i', jac, r) + strength*vg
            added = strength*vd + damping*(jnp.diag(hessian) + .01)
            matrix = hessian + jnp.eye(4)*added[None, :]
            direction = -jnp.linalg.solve(matrix, gradient[:, None])[:, 0]
            direction /= jnp.maximum(1., jnp.max(jnp.abs(direction)))
            return .5*jnp.sum(r*r), gradient, direction

        costs, gradient, direction = jax.lax.map(
            voxel, (x, y, tvg, tvd), batch_size=batch_size)
        return jnp.sum(costs) + jnp.sum(strength*energy), gradient, direction

    def voxel_costs(x, y, b):
        return jax.lax.map(lambda args: voxel_cost(args, b), (x, y), batch_size=batch_size)

    def tv_density(q):
        # per-voxel, per-parameter TV energy density (sums to tv_terms' energy)
        square = jnp.zeros_like(q)
        for axis, h in enumerate(spacing):
            a = [slice(None)] * 3; c = a.copy()
            a[axis], c[axis] = slice(None, -1), slice(1, None)
            edge = domain[tuple(a)] & domain[tuple(c)]
            delta = jnp.where(edge[..., None], jnp.diff(q, axis=axis) / h, 0.)
            pads = [(0, 0)] * 4; pads[axis] = (0, 1)
            square += jnp.pad(delta * delta, pads)
        return jnp.sqrt(square + epsilon**2) - epsilon

    @jax.jit(static_argnames=('steps',))
    def advance(x, damping, y, b, strength, lo, hi, *, steps):
        # float32: accept steps on the voxel-wise summed CHANGE in cost (precision-safe), not on the
        # difference of two large totals; float64 keeps the original test (results unchanged).
        delta_test = x.dtype == jnp.float32
        history = jnp.full((steps+1, 3), jnp.nan)
        state = (jnp.asarray(0), x, damping, history, jnp.asarray(False))

        def condition(s):
            return (s[0] < steps) & ~s[4]

        def iteration(s):
            i, x, damping, history, _ = s
            value, gradient, direction = derivatives(x, y, b, strength, damping)
            pg = jnp.max(jnp.abs(x-jnp.clip(x-gradient, lo, hi)))
            history = history.at[i].set(jnp.array([value, pg, damping]))
            line = (jnp.asarray(0), x, jnp.asarray(False))
            if delta_test:
                c0 = voxel_costs(x, y, b); d0 = tv_density(grid(x))

            def more(z):
                return (z[0] < 25) & ~z[2]

            def trial(z):
                count, previous, _ = z
                candidate = jnp.clip(x + 2.**(-count)*direction, lo, hi)
                slope = jnp.sum(gradient*(candidate-x))
                if delta_test:
                    c1 = voxel_costs(candidate, y, b)
                    change = jnp.sum(c1 - c0) + jnp.sum(strength * (tv_density(grid(candidate)) - d0))
                    accepted = (slope < 0) & jnp.all(jnp.isfinite(c1)) & (change <= 1e-4*slope)
                else:
                    trial_value = objective(candidate, y, b, strength)
                    accepted = (slope < 0) & jnp.isfinite(trial_value) & (trial_value <= value+1e-4*slope)
                return count+1, jnp.where(accepted, candidate, previous), accepted

            _, new_x, accepted = jax.lax.while_loop(more, trial, line)
            new_damping = jnp.where(accepted, jnp.maximum(1e-7, damping*.5), damping*10.)
            return i+1, new_x, new_damping, history, new_damping > 1e12

        count, x, damping, history, failed = jax.lax.while_loop(condition, iteration, state)
        value, gradient, _ = derivatives(x, y, b, strength, damping)
        pg = jnp.max(jnp.abs(x-jnp.clip(x-gradient, lo, hi)))
        history = history.at[count].set(jnp.array([value, pg, damping]))
        return x, damping, count, history, failed

    return objective, derivatives, advance
