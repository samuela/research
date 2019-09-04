import time

import numpy as np
from jax import jit, random
import jax.numpy as jp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import FanInConcat, Dense, Relu, Tanh

from research import flax
from research.estop import ddpg, replay_buffers
from research.estop.half_cheetah import config
from research.estop.utils import Scalarify
from research.statistax import Deterministic, DiagMVN
from research.utils import make_optimizer

tau = 1e-4
buffer_size = 2**20
batch_size = 128
num_eval_rollouts = 16
policy_evaluation_frequency = 10
opt_init = make_optimizer(optimizers.adam(step_size=1e-3))
noise = lambda _1, _2: DiagMVN(
    jp.zeros(config.action_shape),
    0.5 * jp.ones(config.action_shape),
)

# Actions must be bounded [-1, 1].
actor_init, actor = stax.serial(
    Dense(64),
    Relu,
    Dense(config.action_shape[0]),
    Tanh,
)

critic_init, critic = stax.serial(
    FanInConcat(),
    Dense(64),
    Relu,
    Dense(64),
    Relu,
    Dense(1),
    Scalarify,
)

policy = lambda p: lambda s: Deterministic(actor(p, s))

def rollout(rng, policy, viz: bool = False):
  init_state = config.env.initial_distribution.sample(rng)

  states = [init_state]
  actions = []
  rewards = []
  for _ in range(config.episode_length):
    state = states[-1]

    # These are specialized to the fact the policy and the environment are both
    # deterministic because it's ~10x faster.
    action = policy(state).loc
    next_state = config.env.step(state, action).loc
    reward = config.env.reward(state, action, next_state)

    states.append(next_state)
    actions.append(action)
    rewards.append(reward)

    if viz:
      config._gym_env.render()

  # Drop the last state so that all of the lists have the same length.
  states = states[:-1]

  return states, actions, rewards

def eval_policy(rng, policy):
  total_reward = 0.0
  for r in random.split(rng, num_eval_rollouts):
    _, _, rewards = rollout(r, policy, viz=False)
    total_reward += np.sum(rewards)
  return total_reward / num_eval_rollouts

def train(rng, num_episodes, terminal_criterion, callback):
  actor_init_rng, critic_init_rng, rng = random.split(rng, 3)
  _, init_actor_params = actor_init(actor_init_rng, config.state_shape)
  _, init_critic_params = critic_init(
      critic_init_rng, (config.state_shape, config.action_shape))
  optimizer = opt_init((init_actor_params, init_critic_params))
  tracking_params = optimizer.value

  replay_buffer = replay_buffers.NumpyReplayBuffer(buffer_size,
                                                   config.state_shape,
                                                   config.action_shape)

  run = ddpg.ddpg_episode(
      config.env,
      config.gamma,
      tau,
      actor,
      critic,
      noise,
      terminal_criterion,
      batch_size,
      while_loop=flax.while_loop,
  )

  episode_rngs = random.split(rng, num_episodes)
  init_noise = Deterministic(jp.zeros(config.action_shape))
  for episode in range(num_episodes):
    t0 = time.time()
    episode_length, optimizer, tracking_params, reward, _, _, _, _ = run(
        episode_rngs[episode],
        init_noise,
        replay_buffer,
        optimizer,
        tracking_params,
    )
    if not jp.isfinite(reward):
      raise Exception("Reached non-finite reward. Probably a NaN.")

    callback({
        "episode": episode,
        "episode_length": episode_length,
        "optimizer": optimizer,
        "tracking_params": tracking_params,
        "elapsed": time.time() - t0,
        "reward": reward,
    })

def main():
  num_episodes = 1000

  rng = random.PRNGKey(0)
  np.random.seed(0)

  train_rng, rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  train_reward_per_episode = []
  policy_value_per_episode = []

  def callback(info):
    episode = info['episode']
    reward = info['reward']
    current_actor_params, _ = info["optimizer"].value

    print(f"Episode {episode}, "
          f"train reward = {reward}, "
          #   f"policy value = {policy_value}, "
          f"elapsed = {info['elapsed']}")

    train_reward_per_episode.append(reward)

    # Periodically evaluate the policy without any action noise.
    if episode % policy_evaluation_frequency == 0:
      tic = time.time()
      policy_value = eval_policy(
          callback_rngs[episode],
          jit(policy(current_actor_params)),
      )
      print(f".. policy value = {policy_value}, elapsed = {time.time() - tic}")
      policy_value_per_episode.append(policy_value)

  train(
      train_rng,
      num_episodes,
      lambda t, _: t >= config.episode_length,
      callback,
  )

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(train_reward_per_episode)
  plt.xlabel("Episode")
  plt.ylabel("Train episode reward, including action noise")

  plt.figure()
  plt.plot(
      policy_evaluation_frequency * np.arange(len(policy_value_per_episode)),
      policy_value_per_episode)
  plt.xlabel("Episode")
  plt.ylabel("Policy expected cumulative reward")
  plt.show()

if __name__ == "__main__":
  main()
