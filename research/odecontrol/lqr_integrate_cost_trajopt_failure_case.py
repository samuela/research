"""This is an attempt at naively using a neural ODE for trajectory optimization
over a finite time horizon. This script shows that it fails miserably due to
issues recovering the initial conditions in reverse pass."""

from operator import itemgetter
import time
from typing import NamedTuple
import control
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import lax
from jax import jit
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense
from jax.experimental.stax import Tanh
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from research.utils import make_optimizer
from research.utils import random_psd
from research.utils import zeros_like_tree
from research import blt

def fixed_env(n):
  A = 0 * jp.eye(n)
  # A = jp.diag(jp.array([-1.0, 1.0]))
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))
  return A, B, Q, R, N

# def random_env(rng):
#   rngA, rngB, rngQ, rngR = random.split(rng, 4)
#   A = -1 * random_psd(rngA, 2)
#   B = random.normal(rngB, (2, 2))
#   Q = random_psd(rngQ, 2) + 0.1 * jp.eye(2)
#   R = random_psd(rngR, 2) + 0.1 * jp.eye(2)
#   N = jp.zeros((2, 2))
#   return A, B, Q, R, N

def policy_integrate_cost(dynamics_fn, position_cost_fn, control_cost_fn, gamma, policy):
  # Specialize to the environment.

  def ofunc(y, t, policy_params):
    _, _, x = y
    u = policy(policy_params, x)
    return ((gamma**t) * position_cost_fn(x), (gamma**t) * control_cost_fn(u), dynamics_fn(x, u))

  def eval_from_x0(policy_params, x0, total_time):
    # Zero is necessary for some reason...
    ts = jp.array([0.0, total_time])
    y0 = (jp.zeros(()), jp.zeros(()), x0)
    odeint_kwargs = {"mxstep": 1e6}
    y_fwd = ode.odeint(ofunc, y0, ts, policy_params, **odeint_kwargs)
    yT = tree_map(itemgetter(1), y_fwd)

    # This is similar but not exactly the same as the place that the rev-mode
    # solution since the step sizes can vary when using all the other
    # parameters.
    y_bwd = ode.odeint(lambda y, t, *args: tree_map(jp.negative, ofunc(y, -t, *args)), yT,
                       -ts[::-1], policy_params, **odeint_kwargs)
    y0_bwd = tree_map(itemgetter(1), y_bwd)

    return y0, yT, y0_bwd

  return eval_from_x0

def fruity_loops(outer_loop_fn, inner_loop_fn, outer_loop_count, inner_loop_count, init):
  run = jit(lambda carry: lax.scan(inner_loop_fn, carry, jp.arange(inner_loop_count)))
  last = jit(lambda seq: tree_map(itemgetter(-1), seq))

  history = []
  carry = init
  for _ in range(outer_loop_count):
    t0 = time.time()
    carry, seq = run(carry)
    seq_last = tree_map(lambda x: x.block_until_ready(), last(seq))
    history.append(seq)
    outer_loop_fn(carry, seq_last, elapsed=time.time() - t0)

  return carry, tree_multimap(lambda *args: jp.concatenate(args), history[0], *history[1:])

class Record(NamedTuple):
  x_cost_T_fwd_per_iter: jp.ndarray
  u_cost_T_fwd_per_iter: jp.ndarray
  xT_fwd_per_iter: jp.ndarray
  x_cost_0_bwd_per_iter: jp.ndarray
  u_cost_0_bwd_per_iter: jp.ndarray
  x0_bwd_per_iter: jp.ndarray

def main():
  total_time = 20.0
  gamma = 1.0
  x_dim = 2
  outer_loop_count = 5
  inner_loop_count = 3
  rng = random.PRNGKey(0)

  x0 = jp.array([2.0, 1.0])

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu
  A, B, Q, R, N = fixed_env(x_dim)
  print("System dynamics:")
  print(f"  A = {A}")
  print(f"  B = {B}")
  print(f"  Q = {Q}")
  print(f"  R = {R}")
  print(f"  N = {N}")
  print()

  dynamics_fn = lambda x, u: A @ x + B @ u
  # cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u
  position_cost_fn = lambda x: x.T @ Q @ x
  control_cost_fn = lambda u: u.T @ R @ u

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  _, (opt_x_cost_fwd, opt_u_cost_fwd,
      opt_xT_fwd), (opt_x_cost_bwd, opt_u_cost_bwd,
                    opt_x0_bwd) = policy_integrate_cost(dynamics_fn, position_cost_fn,
                                                        control_cost_fn, gamma,
                                                        lambda _, x: -K @ x)(None, x0, total_time)
  opt_cost_fwd = opt_x_cost_fwd + opt_u_cost_fwd
  print("LQR solution:")
  print(f"  K                     = {K}")
  print(f"  opt_x_cost_fwd        = {opt_x_cost_fwd}")
  print(f"  opt_u_cost_fwd        = {opt_u_cost_fwd}")
  print(f"  opt_x_cost_bwd        = {opt_x_cost_bwd}")
  print(f"  opt_u_cost_bwd        = {opt_u_cost_bwd}")
  print(f"  opt_cost_fwd          = {opt_cost_fwd}")
  print(f"  opt_xT_fwd            = {opt_xT_fwd}")
  print(f"  opt_x0_bwd            = {opt_x0_bwd}")
  print(f"  ||x0 - opt_x0_bwd||^2 = {jp.sum((x0 - opt_x0_bwd)**2)}")
  print()

  ### Set up the learned policy model.
  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  init_opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)

  def inner_loop(opt, _):
    runny_run = policy_integrate_cost(dynamics_fn, position_cost_fn, control_cost_fn, gamma, policy)

    (y0_fwd, yT_fwd, y0_bwd), vjp = jax.vjp(runny_run, opt.value, x0, total_time)
    x_cost_T_fwd, u_cost_T_fwd, xT_fwd = yT_fwd
    x_cost_0_bwd, u_cost_0_bwd, x0_bwd = y0_bwd

    yT_fwd_bar = (jp.ones(()), jp.ones(()), jp.zeros_like(x0))
    g, _, _ = vjp((zeros_like_tree(y0_fwd), yT_fwd_bar, zeros_like_tree(y0_bwd)))

    return opt.update(g), Record(x_cost_T_fwd, u_cost_T_fwd, xT_fwd, x_cost_0_bwd, u_cost_0_bwd,
                                 x0_bwd)

  def outer_loop(opt, last: Record, elapsed=None):
    x_cost_T_fwd, u_cost_T_fwd, xT_fwd, x_cost_0_bwd, u_cost_0_bwd, x0_bwd = last
    print(f"Episode {opt.iteration}:")
    print(f"  excess fwd cost = {(x_cost_T_fwd + u_cost_T_fwd) - opt_cost_fwd}")
    print(f"    excess fwd x cost = {x_cost_T_fwd - opt_x_cost_fwd}")
    print(f"    excess fwd u cost = {u_cost_T_fwd - opt_u_cost_fwd}")
    print(f"  bwd cost        = {x_cost_0_bwd + u_cost_0_bwd}")
    print(f"    bwd x cost        = {x_cost_0_bwd}")
    print(f"    bwd u cost        = {u_cost_0_bwd}")
    print(f"  bwd x0 - x0     = {x0_bwd - x0}")
    print(f"  fwd xT          = {xT_fwd}")
    print(f"  fwd xT norm sq. = {jp.sum(xT_fwd**2)}")
    print(f"  elapsed/iter    = {elapsed/inner_loop_count}s")

  ### Main optimization loop.
  t1 = time.time()
  _, history = fruity_loops(outer_loop, inner_loop, outer_loop_count, inner_loop_count, init_opt)
  print(f"total elapsed = {time.time() - t1}s")

  blt.remember({"history": history})

  cost_T_fwd_per_iter = history.x_cost_T_fwd_per_iter + history.u_cost_T_fwd_per_iter
  cost_0_bwd_per_iter = history.x_cost_0_bwd_per_iter + history.u_cost_0_bwd_per_iter

  ### Plot performance per iteration, incl. average optimal policy performance.
  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(cost_T_fwd_per_iter, color="tab:blue", label="Total rollout cost")
  ax1.plot(history.x_cost_T_fwd_per_iter,
           linestyle="dotted",
           color="tab:blue",
           label="Position cost")
  ax1.plot(history.u_cost_T_fwd_per_iter,
           linestyle="dashed",
           color="tab:blue",
           label="Control cost")
  plt.axhline(opt_cost_fwd, linestyle="--", color="gray")
  ax1.legend(loc="upper left")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Error", color="tab:red")
  ax2.set_yscale("log")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax2.plot(cost_0_bwd_per_iter**2, alpha=0.5, color="tab:red", label="Cost rewind error")
  ax2.plot(jp.sum((history.x0_bwd_per_iter - x0)**2, axis=-1),
           alpha=0.5,
           color="tab:purple",
           label="x(0) rewind error")
  ax2.plot(jp.sum(history.xT_fwd_per_iter**2, axis=-1),
           alpha=0.5,
           color="tab:brown",
           label="x(T) squared norm")
  ax2.legend(loc="upper right")

  plt.title(f"ODE control of LQR problem")

  blt.show()

if __name__ == "__main__":
  main()
