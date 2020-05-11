import time
import control
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import value_and_grad
from jax import lax
from jax import vmap
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense
from jax.experimental.stax import Relu
from jax.experimental.stax import Tanh
from research.utils import make_optimizer
from research.utils import DenseNoBias
from research.utils import random_psd
from research import blt
from jax.tree_util import tree_map
import operator as op

def fixed_env(n):
  A = 0 * jp.eye(n)
  # A = jp.diag(jp.array([-1.0, 1.0]))
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))
  return A, B, Q, R, N

# This is a lightly modified version of the jax.experimental.ode._odeint_rev
# function. It is modified to additionally return the reconstructed `ys`.
def odeint_rev(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = jp.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    y_reconstructed, y_bar, t0_bar, args_bar = ode.odeint(aug_dynamics,
                                                          (ys[i], y_bar, t0_bar, args_bar),
                                                          jp.array([-ts[i], -ts[i - 1]]),
                                                          *args,
                                                          rtol=rtol,
                                                          atol=atol,
                                                          mxstep=mxstep)
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar, y_reconstructed

  init_carry = (g[-1], 0., tree_map(jp.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar, y_reconstructed = lax.scan(scan_fun, init_carry,
                                                                    jp.arange(len(ts) - 1, 0, -1))
  ts_bar = jp.concatenate([jp.array([t0_bar]), rev_ts_bar[::-1]])

  # `y_reconstructed` is the recovered set of ys found via the backwards,
  # augemented dynamics. TODO: This may need to be flipped...
  return (y_bar, ts_bar, *args_bar), y_reconstructed

def eval_and_grad_bwd(dynamics_fn, cost_fn, policy, V, gamma):
  V_vg = value_and_grad(V)

  def value_function_loss(V_params, x0, xT, total_time, discounted_reward):
    """Calculate the loss of the value function V. `discounted_reward` is the
    integral of time-discounted rewards between x0 and xT."""
    return (V(V_params, x0) - (gamma**total_time) * V(V_params, xT) - discounted_reward)**2

  Vloss_vg = value_and_grad(value_function_loss)

  def ofunc(y, t, policy_params):
    # TODO can y just be a pytree thingy?
    x = y[1:]
    u = policy(policy_params, x)
    return jp.concatenate((jp.expand_dims((gamma**t) * cost_fn(x, u), axis=0), dynamics_fn(x, u)))

  def fwd_from_x0(params, x0, total_time):
    policy_params, V_params = params

    # Zero is necessary because we need both start and ending time points.
    ts = jp.array([0.0, total_time])
    y0 = jp.concatenate((jp.zeros((1, )), x0))

    ys = ode.odeint(ofunc, y0, ts, policy_params, mxstep=1e6)
    discounted_cost = ys[1, 0]
    xT = ys[1, 1:]
    final_value = V(V_params, xT)
    return discounted_cost - (gamma**total_time) * final_value

  def bwd_from_xT(params, xT, total_time):
    policy_params, V_params = params
    V_xT, grad_V_wrt_xT = V_vg(V_params, xT)

    # Zero is necessary because we need both start and ending time points.
    ts = jp.array([0.0, total_time])
    yT = jp.concatenate((jp.zeros((1, )), xT))

    # The first row of `ys` and `g` shouldn't matter at all.
    ys = jp.stack((jp.zeros_like(yT), yT))

    # We need to multiply by -gamma^T in order to account for the coefficient
    # on V(x(T)) in the final loss.
    grad_loss_wrt_xT = tree_map(lambda x: -(gamma**total_time) * x, grad_V_wrt_xT)
    grad_ys = jp.stack((jp.zeros_like(yT), jp.concatenate((jp.ones((1, )), grad_loss_wrt_xT))))

    # pylint: disable=protected-access
    (_, _, grad_policy_params), y_bwd = odeint_rev(ofunc,
                                                   rtol=1e-6,
                                                   atol=1e-6,
                                                   mxstep=1e6,
                                                   res=(ys, ts, policy_params),
                                                   g=grad_ys)

    # TODO: Reweight according to p(x(0)).

    # These are accumulated in the negative in reverse-time, so they become
    # rewards instead of costs.
    discounted_rewards = y_bwd[0, 0]
    x0 = y_bwd[0, 1:]
    total_cost = -discounted_rewards - (gamma**total_time) * V_xT
    Vloss, grad_V_params = Vloss_vg(V_params, x0, xT, total_time, discounted_rewards)

    print(y_bwd)

    return x0, (total_cost, Vloss), (grad_policy_params, grad_V_params)

  return fwd_from_x0, bwd_from_xT

def main():
  total_time = 20.0
  gamma = 1.0
  x_dim = 2
  rng = random.PRNGKey(0)

  x0 = jp.array([2.0, 1.0])
  # rng_x0, rng = random.split(rng)
  # x0 = random.normal(rng_x0, shape=(x_dim, ))

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu
  A, B, Q, R, N = fixed_env(x_dim)
  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u
  policy_loss = policy_integrate_cost(dynamics_fn, cost_fn, gamma)

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  t0 = time.time()
  opt_y_fwd, opt_y_bwd = policy_loss(lambda _, x: -K @ x)(None, x0, total_time)
  opt_cost = opt_y_fwd[1, 0]
  print(f"opt_cost = {opt_cost} in {time.time() - t0}s")
  print(opt_y_fwd)
  print(opt_y_bwd)
  print(f"l2 error: {jp.sqrt(jp.sum((opt_y_fwd - opt_y_bwd)**2))}")

  ### Set up the learned policy model.
  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      # Dense(64),
      # Tanh,
      Dense(x_dim),
  )
  # policy_init, policy = DenseNoBias(2)

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  # cost_and_grad = jit(value_and_grad(policy_loss(policy)))
  runny_run = jit(policy_loss(policy))

  ### Main optimization loop.
  costs = []
  bwd_errors = []
  for i in range(5000):
    t0 = time.time()
    (y_fwd, y_bwd), vjp = jax.vjp(runny_run, opt.value, x0, total_time)
    cost = y_fwd[1, 0]

    y_fwd_bar = jax.ops.index_update(jp.zeros_like(y_fwd), (1, 0), 1)
    g, _, _ = vjp((y_fwd_bar, jp.zeros_like(y_bwd)))
    opt = opt.update(g)

    bwd_err = jp.sqrt(jp.sum((y_fwd - y_bwd)**2))
    bwd_errors.append(bwd_err)

    print(
        f"Episode {i}: excess cost = {cost - opt_cost}, bwd error = {bwd_err} elapsed = {time.time() - t0}"
    )
    costs.append(float(cost))

  print(f"Opt solution cost from starting point: {opt_cost}")
  # print(f"Gradient at opt solution: {opt_g}")

  # Print the identified and optimal policy. Note that layers multiply multipy
  # on the right instead of the left so we need a transpose.
  # print(f"Est solution parameters: {opt.value}")
  # print(f"Opt solution parameters: {-K.T}")

  ### Plot performance per iteration, incl. average optimal policy performance.
  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(costs, color="tab:blue")
  plt.axhline(opt_cost, linestyle="--", color="gray")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Backward solve L2 error", color="tab:red")
  ax2.set_yscale("log")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax2.plot(bwd_errors, color="tab:red")
  # plt.yscale("log")
  # plt.xlabel("Iteration")
  # plt.ylabel("Cost")
  # plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title(f"ODE control of LQR problem")

  blt.show()

if __name__ == "__main__":
  main()