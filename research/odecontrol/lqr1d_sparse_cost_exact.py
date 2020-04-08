"""Try the "sparse cost" variant running on a 1d LQR problem with the exact solutions."""

import time
import control
import matplotlib.pyplot as plt
from jax import random
from jax import jit
from jax import value_and_grad
from jax import vmap
import jax.numpy as jp
from jax.experimental import optimizers
from research.utils import make_optimizer

def main():
  num_keypoints = 64
  train_batch_size = 64
  eval_batch_size = 1024
  gamma = 0.9
  rng = random.PRNGKey(0)

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu

  A = -0.1 * jp.eye(1)
  B = jp.eye(1)
  Q = jp.eye(1)
  R = jp.eye(1)
  N = jp.zeros((1, 1))

  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  def loss(batch_size):
    def lossy_loss(KK, rng):
      rng_t, rng_x0 = random.split(rng)
      x0 = random.normal(rng_x0, shape=(batch_size, ))
      t = random.exponential(rng_t, shape=(num_keypoints, )) / -jp.log(gamma)
      x_t = jp.outer(jp.exp(t * jp.squeeze(A - B @ KK)), x0)
      costs = vmap(lambda x: cost_fn(x, -KK @ x))(jp.reshape(x_t, (-1, 1)))
      return jp.mean(costs)

    return lossy_loss

  t0 = time.time()
  rng_eval_keypoints, rng = random.split(rng)
  opt_all_costs = loss(eval_batch_size)(K, rng_eval_keypoints)
  opt_cost = jp.mean(opt_all_costs)
  print(f"opt_cost = {opt_cost} in {time.time() - t0}s")

  ### Set up the learned policy model.
  rng_init_params, rng = random.split(rng)
  opt = make_optimizer(optimizers.adam(1e-3))(random.normal(rng_init_params, shape=(1, 1)))
  cost_and_grad = jit(value_and_grad(loss(train_batch_size)))

  ### Main optimization loop.
  costs = []
  for i in range(10000):
    t0 = time.time()
    rng_iter, rng = random.split(rng)
    cost, g = cost_and_grad(opt.value, rng_iter)
    opt = opt.update(g)
    print(f"Episode {i}: excess cost = {cost - opt_cost}, elapsed = {time.time() - t0}")
    costs.append(float(cost))

  print(f"Opt solution cost from starting point: {opt_cost}")
  print(f"Est solution parameters: {opt.value}")
  print(f"Opt solution parameters: {K}")

  ### Plot performance per iteration, incl. average optimal policy performance.
  plt.figure()
  plt.plot(costs)
  plt.axhline(opt_cost, linestyle="--", color="gray")
  plt.yscale("log")
  plt.xlabel("Iteration")
  plt.ylabel(f"Cost (gamma = {gamma}, num_keypoints = {num_keypoints})")
  plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title("ODE control of LQR problem (keypoint sampling)")

  plt.show()

if __name__ == "__main__":
  main()
