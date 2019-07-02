from jax import jit, random
import jax.numpy as jp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import FanInConcat, Dense, Relu
import matplotlib.pyplot as plt

from research.estop import ddpg
from research.gan_with_the_wind import dists
from research.estop import pendulum
from research.estop.utils import Scalarify
from research.utils import make_optimizer

gamma = 0.99
tau = 0.001
episode_length = 1000
num_episodes = 1000
buffer_size = 16384
batch_size = 64
opt_init = make_optimizer(optimizers.adam(step_size=1e-3))
noise = lambda _: dists.Normal(jp.array(0.0), jp.array(0.1))
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
optimizer = opt_init((init_actor_params, init_critic_params))
tracking_params = optimizer.value

run = jit(
    ddpg.ddpg_episode(
        env,
        gamma,
        tau,
        actor,
        critic,
        noise,
        episode_length,
        batch_size,
    ))

rngs = random.split(rng, num_episodes)

reward_per_episode = []
for episode in range(num_episodes):
  optimizer, tracking_params, reward, final_state, replay_buffer = run(
      rngs[episode],
      replay_buffer,
      optimizer,
      tracking_params,
  )
  print(f"Episode {episode}, reward = {reward}")
  reward_per_episode.append(reward)

  if not jp.isfinite(reward):
    break

# Plot the reward per episode.
plt.figure()
plt.plot(reward_per_episode)
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("DDPG on the pendulum environment")
plt.show()
