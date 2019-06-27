from gym.envs.classic_control import PendulumEnv
from jax import random
import jax.numpy as jp

from research.estop import ddpg
from research.estop import pendulum
from research.gan_with_the_wind import dists

env = pendulum.pendulum_environment(
    mass=1.0,
    length=1.0,
    gravity=9.8,
    friction=1,
    dt=0.05,
)

states, actions = ddpg.rollout_from_state(
    random.PRNGKey(0),
    env,
    lambda _: dists.Deterministic(jp.array(0.0)),
    num_timesteps=2500,
    state=jp.array([jp.pi - 0.01, 0.0]))

gymenv = PendulumEnv()
gymenv.reset()
# array(0.0) is False-y which causes problems.
gymenv.last_u = jp.array(1e-12)

for t in range(states.shape[0]):
  gymenv.state = states[t] + jp.pi
  gymenv.render()
