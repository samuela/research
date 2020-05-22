"""A quick script to test whether or not traditional Neural ODEs are susceptible
to the same sorts of issues with their backwards passes not actually landing
back at x_0. It looks as though they do work fine as long as you don't go out to
obscene time scales."""

import matplotlib.pyplot as plt
from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.experimental.stax import Tanh
from jax import random
from jax import vmap
from jax.experimental import ode
import jax.numpy as jnp
from research.estop.frozenlake.viz import plot_errorfill
from research import blt

def main():
  num_rng_keys = 1024
  x_dim = 64
  times = jnp.arange(1, 501, dtype=jnp.float32)

  dynamics_init, dynamics = stax.serial(
      Dense(64),
      Tanh,
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  def one_rng(rng):
    _, params = dynamics_init(rng, (x_dim, ))
    fwd_x_T = ode.odeint(lambda x, _: dynamics(params, x), jnp.zeros((x_dim, )), times, mxstep=1e9)
    bwd_x_0 = vmap(lambda T, x_T: ode.odeint(
        lambda x, _: -dynamics(params, x), x_T, jnp.array([0.0, T]), mxstep=1e9)[1],
                   in_axes=(0, 0))(times, fwd_x_T)
    return jnp.sum(bwd_x_0**2, axis=-1)

  bwd_errors = vmap(one_rng)(random.split(random.PRNGKey(0), num_rng_keys))
  blt.remember({"bwd_errors": bwd_errors})

  plt.figure()
  plot_errorfill(times, bwd_errors, "tab:red")
  plt.title("Neural ODE reverse error")
  plt.xlabel("T")
  plt.ylabel("Square L2 error of the backwards recovered x_0")
  blt.show()

if __name__ == "__main__":
  main()
