"""Solving the adjoint system as a two-point boundary value problem (BVP).

The standard shooting method is still used to compute y(T) in the forward pass,
but an implicit RK4 scheme with adaptive meshing is used to compute the adjoint
values in the backward pass."""

from functools import partial
import operator as op
from scipy.integrate import solve_bvp
import jax
from jax import vmap
import jax.numpy as jnp
from jax import core
from jax import lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from jax.experimental.ode import ravel_first_arg

def _solve_bvp(fun, bc, x, y, *args):
  num_nodes, = x.shape
  example_y_state = tree_map(op.itemgetter(0), y)
  _, unravel = ravel_pytree(example_y_state)

  def ravel(pytree):
    arr_flat, _ = ravel_pytree(pytree)
    return arr_flat

  def fun_wrap(x, y):
    return vmap(lambda xi, yi_flat: ravel(fun(xi, unravel(yi_flat), *args)), in_axes=(0, 0))(x,
                                                                                             y.T).T

  def bc_wrap(ya, yb):
    return ravel(bc(unravel(ya), unravel(yb)))

  y_flat = vmap(lambda ix: ravel(tree_map(op.itemgetter(ix), y)))(jnp.arange(num_nodes)).T
  bvp_solution = solve_bvp(fun_wrap, bc_wrap, x, y_flat)
  return bvp_solution.x, vmap(unravel)(bvp_solution.y.T), bvp_solution

def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
  """See the jax.experimental.ode.odeint docs."""
  def _check_arg(arg):
    if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
      msg = ("The contents of odeint *args must be arrays or scalars, but got \n{}.")
      raise TypeError(msg.format(arg))

  tree_map(_check_arg, args)
  return _odeint_wrapper(func, rtol, atol, mxstep, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_wrapper(func, rtol, atol, mxstep, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out = _odeint(func, rtol, atol, mxstep, y0, ts, unravel, *args)
  return jax.vmap(unravel)(out)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _odeint(func, rtol, atol, mxstep, _unravel, y0, ts, *args):
  # pylint: disable=protected-access
  return jax.experimental.ode._odeint(func, rtol, atol, mxstep, y0, ts, *args)

def _odeint_fwd(func, rtol, atol, mxstep, unwrap, y0, ts, *args):
  ys = _odeint(func, rtol, atol, mxstep, unwrap, y0, ts, *args)
  return ys, (ys, ts, args)

def _odeint_rev(func, rtol, atol, mxstep, unwrap, res, g):
  ys, ts, args = res

  def aug_dynamics(t, augmented_state, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    # Passing in `*args` is necessary
    y, y_bar, _, _ = augmented_state

    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, t, *args)
    return (y_dot, *tree_map(jnp.negative, vjpfun(y_bar)))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar

    # Run augmented system backwards to previous observation
    # _, y_bar, t0_bar, args_bar = odeint(aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
    #                                     jnp.array([-ts[i], -ts[i - 1]]),
    #                                     ys[i - 1],
    #                                     *args,
    #                                     rtol=rtol,
    #                                     atol=atol,
    #                                     mxstep=mxstep)

    def bc(aug_start, aug_end):
      y_start, _, _, _ = aug_start
      _, y_bar_end, t0_bar_end, args_bar_end = aug_end
      return (y_start - ys[i - 1], y_bar_end - y_bar, t0_bar_end - t0_bar, args_bar_end - args_bar)

    aug_start_guess = (ys[i - 1], y_bar, t0_bar, args_bar)
    aug_end_guess = (ys[i], y_bar, t0_bar, args_bar)
    _, bvp_soln_y, bvp_soln = _solve_bvp(
        aug_dynamics, bc, jnp.array([ts[i - 1], ts[i]]),
        tree_multimap(lambda x, y: jnp.stack((x, y)), aug_start_guess, aug_end_guess))

    # These are the new adjoint values after solving the BVP system.
    _, y_bar, t0_bar, args_bar = tree_map(op.itemgetter(0), bvp_soln_y)

    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(jnp.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(scan_fun, init_carry,
                                                   jnp.arange(len(ts) - 1, 0, -1))
  ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])

  return (y_bar, ts_bar, *args_bar)

_odeint.defvjp(_odeint_fwd, _odeint_rev)
