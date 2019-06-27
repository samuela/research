from jax import random
import jax.numpy as jp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import FanInConcat, Dense, Relu

from research.estop import ddpg
from research.statistax import Normal
from research.estop import pendulum
from research.estop.utils import Scalarify

gamma = 0.99
tau = 0.01
episode_length = 100
buffer_size = 128
batch_size = 32
optimizer = ddpg.Optimizer(*optimizers.adam(step_size=1e-3))
noise = lambda _: Normal(jp.array(0.0), jp.array(0.1))
rng = random.PRNGKey(0)

env = pendulum.pendulum_environment(
    mass=1.0,
    length=1.0,
    gravity=9.8,
    friction=0,
    dt=0.05,
)

replay_buffer = ddpg.ReplayBuffer(
    states=jp.zeros((buffer_size, 2)),
    actions=jp.zeros((buffer_size, 1)),
    rewards=jp.zeros((buffer_size, )),
    next_states=jp.zeros((buffer_size, 2)),
    count=0,
)

actor_init, actor = stax.serial(
    Dense(32),
    Relu,
    Dense(32),
    Relu,
    Dense(1),
)
critic_init, critic = stax.serial(
    FanInConcat(),
    Dense(32),
    Relu,
    Dense(32),
    Relu,
    Dense(1),
    Scalarify,
)

actor_init_rng, critic_init_rng = random.split(rng)
_, init_actor_params = actor_init(actor_init_rng, (2, ))
_, init_critic_params = critic_init(critic_init_rng, ((2, ), (1, )))
opt_state = optimizer.init((init_actor_params, init_critic_params))
tracking_params = optimizer.get(opt_state)

for epsiode in range(100):
  opt_state, tracking_params, reward, final_state, replay_buffer = ddpg.ddpg_episode(
      rng,
      replay_buffer,
      batch_size,
      optimizer,
      opt_state,
      tracking_params,
      env,
      gamma,
      tau,
      actor,
      critic,
      episode_length,
      noise,
  )
  print(f"Episode {epsiode}, reward = {reward}")
