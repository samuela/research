import time

import control
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Tanh

from research import blt
from research.odecontrol.lqr_integrate_cost_trajopt_failure_case import \
    fixed_env
from research.odecontrol.radau_ode import policy_cost_and_grad
from research.utils import make_optimizer

# import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

def main():
  rng = random.PRNGKey(0)
  x_dim = 2
  T = 20.0

  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  x0 = jnp.ones(x_dim)

  A, B, Q, R, N = fixed_env(x_dim)
  print("System dynamics:")
  print(f"  A = {A}")
  print(f"  B = {B}")
  print(f"  Q = {Q}")
  print(f"  R = {R}")
  print(f"  N = {N}")
  print()

  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u

  ### Evaluate LQR solution to get a sense of optimal cost.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jnp.array(K)
  opt_policy_cost_fn = policy_cost_and_grad(dynamics_fn,
                                            cost_fn,
                                            lambda KK, x: -KK @ x,
                                            example_x=x0)
  opt_loss, _opt_K_grad = opt_policy_cost_fn(K, x0, T)

  # This is true for longer time horizons, but not true for shorter time
  # horizons due to the LQR solution being an infinite-time solution.
  # assert jnp.allclose(opt_K_grad, 0)

  ### Training loop.
  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = policy_cost_and_grad(dynamics_fn, cost_fn, policy, example_x=x0)

  loss_per_iter = []
  elapsed_per_iter = []
  for iteration in range(10000):
    t0 = time.time()
    loss, g = loss_and_grad(opt.value, x0, T)
    opt = opt.update(g)
    elapsed = time.time() - t0

    loss_per_iter.append(loss)
    elapsed_per_iter.append(elapsed)

    print(f"Iteration {iteration}")
    print(f"    excess loss = {loss - opt_loss}")
    print(f"    elapsed = {elapsed}")

  blt.remember({
      "loss_per_iter": loss_per_iter,
      "elapsed_per_iter": elapsed_per_iter,
      "opt_loss": opt_loss
  })

  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(loss_per_iter, color="tab:blue", label="Total rollout cost")
  plt.axhline(opt_loss, linestyle="--", color="gray")
  ax1.legend(loc="upper left")
  plt.title("Combined fwd-bwd BVP problem")
  blt.show()

if __name__ == "__main__":
  main()
