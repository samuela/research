import jax.numpy as jp

from research.estop import ddpg
from research.statistax import Deterministic, Uniform

def pendulum_environment(mass: float, length: float, gravity: float,
                         friction: float, dt: float) -> ddpg.Env:
  def step(state, action):
    """Take a single step in the discretized pendulum dynamics."""
    assert state.shape == (2, )
    assert action.shape == (1, )

    theta = state[0]
    theta_dot = state[1]
    u = action[0]

    theta_dotdot = (-gravity / length * jp.sin(theta) - friction * theta_dot +
                    u / (mass * length**2))
    new_theta_dot = theta_dot + dt * theta_dotdot
    new_theta = (theta + dt * new_theta_dot) % (2 * jp.pi)
    return Deterministic(jp.array([new_theta, new_theta_dot]))

  return ddpg.Env(
      initial_distribution=Uniform(
          jp.array([0, -1]),
          jp.array([2 * jp.pi, 1]),
      ),
      step=step,
      reward=lambda s1, a, s2: -(s1[0] - jp.pi)**2,
  )
