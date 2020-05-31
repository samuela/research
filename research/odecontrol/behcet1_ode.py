"""An adaptation of jax.experimental.ode that uses Behcet's idea to add a linear
"nudge" term to the reverse solve. This was actually a mis-interpretation of
what Behcet was suggesting. Instead of linear interpolating between the start
and end goals this version always pulls towards the start goal (in reverse
time).

Long story short this doesn't work in fixing the LQR failure case. It does
succeed in pulling the solution in the direction of the start, but it still
suffers from the same overshooting and instability issues."""

from functools import partial
import operator as op
import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.experimental.ode import ravel_first_arg

def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, bwd_bias=1e-6):
  """See the jax.experimental.ode.odeint docs."""
  def _check_arg(arg):
    if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
      msg = ("The contents of odeint *args must be arrays or scalars, but got \n{}.")
      raise TypeError(msg.format(arg))

  tree_map(_check_arg, args)
  return _odeint_wrapper(func, rtol, atol, mxstep, bwd_bias, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_wrapper(func, rtol, atol, mxstep, bwd_bias, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out = _odeint(func, rtol, atol, mxstep, bwd_bias, y0, ts, *args)
  return jax.vmap(unravel)(out)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _odeint(func, rtol, atol, mxstep, _bwd_bias, y0, ts, *args):
  # pylint: disable=protected-access
  return jax.experimental.ode._odeint(func, rtol, atol, mxstep, y0, ts, *args)

def _odeint_fwd(func, rtol, atol, mxstep, bwd_bias, y0, ts, *args):
  ys = _odeint(func, rtol, atol, mxstep, bwd_bias, y0, ts, *args)
  return ys, (ys, ts, args)

def _odeint_rev(func, rtol, atol, mxstep, bwd_bias, res, g):
  ys, ts, args = res

  def aug_dynamics(augmented_state, t, y_target, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state

    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot + bwd_bias * (y_target - y), *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint(aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
                                        jnp.array([-ts[i], -ts[i - 1]]),
                                        ys[i - 1],
                                        *args,
                                        rtol=rtol,
                                        atol=atol,
                                        mxstep=mxstep,
                                        bwd_bias=bwd_bias)
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(jnp.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(scan_fun, init_carry,
                                                   jnp.arange(len(ts) - 1, 0, -1))
  ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
  return (y_bar, ts_bar, *args_bar)

_odeint.defvjp(_odeint_fwd, _odeint_rev)
