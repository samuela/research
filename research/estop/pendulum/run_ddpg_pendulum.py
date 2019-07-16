import pickle
import time

import matplotlib.pyplot as plt
from jax import jit, random
import jax.numpy as jp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import FanInConcat, Dense, Relu, Tanh

from research.estop import ddpg
from research.estop.pendulum import config
from research.estop.pendulum.env import viz_pendulum_rollout
from research.estop.utils import Scalarify
from research.statistax import Deterministic, Normal
from research.utils import make_optimizer

init_rng_key = 0
num_episodes = 1000
tau = 1e-3
buffer_size = 2**15
batch_size = 64
opt_init = make_optimizer(optimizers.adam(step_size=1e-3))
noise = lambda _: Normal(jp.array(0.0), jp.array(0.1))

rng = random.PRNGKey(init_rng_key)

replay_buffer = ddpg.ReplayBuffer(
    states=jp.zeros((buffer_size, ) + config.state_shape),
    actions=jp.zeros((buffer_size, ) + config.action_shape),
    rewards=jp.zeros((buffer_size, )),
    next_states=jp.zeros((buffer_size, ) + config.state_shape),
    count=0,
)

actor_init, actor = stax.serial(
    Dense(128),
    Relu,
    Dense(1),
    Tanh,
    stax.elementwise(lambda x: config.max_torque * x),
)

critic_init, critic = stax.serial(
    FanInConcat(),
    Dense(128),
    Relu,
    Dense(128),
    Relu,
    Dense(1),
    Scalarify,
)

if __name__ == "__main__":
  actor_init_rng, critic_init_rng, rng = random.split(rng, 3)
  _, init_actor_params = actor_init(actor_init_rng, config.state_shape)
  _, init_critic_params = critic_init(
      critic_init_rng, (config.state_shape, config.action_shape))
  optimizer = opt_init((init_actor_params, init_critic_params))
  tracking_params = optimizer.value

  run = jit(
      ddpg.ddpg_episode(
          config.env,
          config.gamma,
          tau,
          actor,
          critic,
          noise,
          config.episode_length,
          batch_size,
      ))

  episode_rngs = random.split(rng, num_episodes)

  reward_per_episode = []
  for episode in range(num_episodes):
    t0 = time.time()
    optimizer, tracking_params, reward, final_state, replay_buffer = run(
        episode_rngs[episode],
        replay_buffer,
        optimizer,
        tracking_params,
    )
    print(
        f"Episode {episode}, reward = {reward}, elapsed = {time.time() - t0}")
    reward_per_episode.append(reward)

    if not jp.isfinite(reward):
      raise Exception("Reached non-finite reward. Probably a NaN.")

    # Visualize a rollout under the current policy.
    if episode % 100 == 0:
      viz_num_rollouts = 100
      rollout_rngs = random.split(episode_rngs[episode], viz_num_rollouts)
      rollouts = [
          ddpg.rollout(
              rollout_rngs[i],
              config.env,
              policy=lambda s: Deterministic(actor(optimizer.value[0], s)),
              num_timesteps=250,
          ) for i in range(viz_num_rollouts)
      ]
      viz_pendulum_rollout(rollouts[0][0], rollouts[0][1])

      plt.figure()
      for states, _ in rollouts:
        plt.scatter(states[:, 0], states[:, 1], c="tab:blue", alpha=0.1)
      plt.xlabel("theta")
      plt.ylabel("theta dot")
      plt.title(f"Episode {episode}")
      plt.show()

  # Store parameters.
  # pickle.dump(optimizer.value, open(f"final_params_rng{init_rng_key}.pkl",
  #                                   "wb"))

  # Plot the reward per episode.
  plt.figure()
  plt.plot(reward_per_episode)
  plt.xlabel("Episode")
  plt.ylabel("Cumulative reward")
  plt.title("DDPG on the pendulum environment")
  plt.show()
