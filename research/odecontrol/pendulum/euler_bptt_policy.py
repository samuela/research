import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, random, value_and_grad, vmap
from jax.experimental import ode, optimizers, stax
from jax.experimental.stax import Dense, Tanh

from research import blt
from research.odecontrol.pendulum.dynamics import (cost, pendulum_dynamics, sample_x0)
from research.utils import make_optimizer

def main():
  num_iter = 50000
  # Most people run 1000 steps and the OpenAI gym pendulum is 0.05s per step.
  # The max torque that can be applied is also 2 in their setup.
  T = 1000
  time_delta = 0.05
  max_torque = 2.0
  rng = random.PRNGKey(0)

  dynamics = pendulum_dynamics(
      mass=1.0,
      length=1.0,
      gravity=9.8,
      friction=0.0,
  )

  policy_init, policy_nn = stax.serial(
      Dense(64),
      Tanh,
      Dense(64),
      Tanh,
      Dense(1),
      Tanh,
      stax.elementwise(lambda x: max_torque * x),
  )

  # Should it matter whether theta is wrapped into [0, 2pi]?
  policy = lambda params, x: policy_nn(
      params, jnp.array([x[0] % (2 * jnp.pi), x[1],
                         jnp.cos(x[0]), jnp.sin(x[0])]))

  def loss(policy_params, x0):
    x = x0
    acc_cost = 0.0
    for _ in range(T):
      u = policy(policy_params, x)
      x += time_delta * dynamics(x, u)
      acc_cost += time_delta * cost(x, u)
    return acc_cost

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (4, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = jit(value_and_grad(loss))

  loss_per_iter = []
  elapsed_per_iter = []
  x0s = vmap(sample_x0)(random.split(rng, num_iter))
  for i in range(num_iter):
    t0 = time.time()
    loss, g = loss_and_grad(opt.value, x0s[i])
    opt = opt.update(g)
    elapsed = time.time() - t0

    loss_per_iter.append(loss)
    elapsed_per_iter.append(elapsed)

    print(f"Episode {i}")
    print(f"    loss = {loss}")
    print(f"    elapsed = {elapsed}")

  blt.remember({
      "loss_per_iter": loss_per_iter,
      "elapsed_per_iter": elapsed_per_iter,
      "final_params": opt.value
  })

  plt.figure()
  plt.plot(loss_per_iter)
  plt.yscale("log")
  plt.title("ODE control of an inverted pendulum")
  plt.xlabel("Iteration")
  plt.ylabel(f"Policy cost (T = {total_secs}s)")

  # Viz
  num_viz_rollouts = 50
  framerate = 30
  timesteps = jnp.linspace(0, int(T * time_delta), num=int(T * time_delta * framerate))
  rollout = lambda x0: ode.odeint(
      lambda x, _: dynamics(x, policy(opt.value, x)), y0=x0, t=timesteps)

  plt.figure()
  states = rollout(jnp.zeros(2))
  plt.plot(states[:, 0], states[:, 1], marker=".")
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Swing up trajectory")

  plt.figure()
  states = vmap(rollout)(x0s[:num_viz_rollouts])
  for i in range(num_viz_rollouts):
    plt.plot(states[i, :, 0], states[i, :, 1], marker='.', alpha=0.5)
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Phase space trajectory")

  plot_control_contour(lambda x: policy(opt.value, x))
  plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))

  blt.show()

if __name__ == "__main__":
  main()
