"""A pendulum swing up environment. This requires a swing-up when `gravity` is
greater than `max_torque / (mass * length)` and can overpower gravity
otherwise."""

import time
import jax.numpy as jp
from jax.experimental import ode
from research.estop.pendulum.env import viz_pendulum_rollout
import matplotlib.pyplot as plt

def pendulum_dynamics(mass: float, length: float, gravity: float, friction: float):
  def f(state, action):
    """Calculate dynamics for pendulum.

    Args:
      state (ndarray): An ndarray with the current theta, and d theta/dt. Note
        that theta ranges from 0 to 2 pi, with 0 and 2 pi denoting the bottom of
        the pendulum swing and pi denoting the top.
      action (ndarray): The force to be applied. Positive force going
        counterclockwise and negative force going clockwise.

    Returns:
      Derivative of the state.
    """
    assert state.shape == (2, )
    assert action.shape == (1, )

    theta = state[0]
    theta_dot = state[1]
    u = action[0]

    theta_dotdot = (-gravity / length * jp.sin(theta) - friction * theta_dot + u /
                    (mass * length**2))

    # theta_dotdot = theta_dotdot * (theta_dot < 10) * (theta_dot > -10)

    return jp.array([theta_dot, theta_dotdot])

  return f

if __name__ == "__main__":
  total_secs = 10
  framerate = 60

  dynamics = pendulum_dynamics(
      mass=0.1,
      length=1.0,
      gravity=9.8,
      friction=0.1,
  )

  print("Solving ODE...")
  t0 = time.time()
  states = ode.odeint(lambda state, t: dynamics(state, jp.zeros((1, ))),
                      y0=jp.array([jp.pi - 1e-1, 0.0]),
                      t=jp.linspace(0, total_secs, num=total_secs * framerate))
  print(f"... and done in {time.time() - t0}s")

  plt.figure()
  plt.plot(states[:, 0], states[:, 1], marker=None)
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.show()

  viz_pendulum_rollout(states, jp.zeros((states.shape[0], )))
