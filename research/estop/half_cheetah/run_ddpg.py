from pathlib import Path
import time

from gym.wrappers.monitoring.video_recorder import VideoRecorder
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
from research.statistax import Deterministic, Normal
from research.utils import make_optimizer

tau = 1e-4
buffer_size = 2**20
batch_size = 128
num_eval_rollouts = 16
policy_evaluation_frequency = 10
policy_video_frequency = 100
opt_init = make_optimizer(optimizers.adam(step_size=1e-3))

# For some reason using DiagMVN here is ~100x slower.
noise = lambda _1, _2: Normal(
    jp.zeros(config.action_shape),
    0.1 * jp.ones(config.action_shape),
)

# Actions must be bounded [-1, 1].
actor_init, actor = stax.serial(
    Dense(64),
    Relu,
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
    Dense(64),
    Relu,
    Dense(1),
    Scalarify,
)

deterministic_policy = lambda p: lambda s: Deterministic(actor(p, s))

def rollout(rng, policy, callback=lambda: None):
  """Rollout one episode of the half_cheetah environment. This is specialized to
  the fact that we can't use lax.scan and the policy and environment dynamics
  are deterministic."""
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

    # Here we let the caller render or capture a frame for video.
    callback()

  # Drop the last state so that all of the lists have the same length.
  states = states[:-1]

  return states, actions, rewards

def eval_policy(rng, policy):
  total_reward = 0.0
  for r in random.split(rng, num_eval_rollouts):
    _, _, rewards = rollout(r, policy)
    total_reward += np.sum(rewards)
  return total_reward / num_eval_rollouts

def film_policy(rng, policy, filepath: Path):
  video_env = VideoRecorder(config._gym_env, path=str(filepath))
  rollout(rng, policy, callback=lambda: video_env.capture_frame())
  video_env.close()

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
      # We need a flax loop here since we can't jit compile mujoco steps.
      while_loop=flax.while_loop,
  )

  episode_rngs = random.split(rng, num_episodes)
  init_noise = Deterministic(jp.zeros(config.action_shape))
  for episode in range(num_episodes):
    t0 = time.time()
    final_state = run(
        episode_rngs[episode],
        init_noise,
        replay_buffer,
        optimizer,
        tracking_params,
    )
    if not jp.isfinite(final_state.undiscounted_cumulative_reward):
      raise Exception("Reached non-finite reward. Probably a NaN.")

    callback({
        "episode": episode,
        "episode_length": final_state.episode_length,
        "optimizer": final_state.optimizer,
        "tracking_params": final_state.tracking_params,
        "discounted_cumulative_reward":
        final_state.discounted_cumulative_reward,
        "undiscounted_cumulative_reward":
        final_state.undiscounted_cumulative_reward,
        "elapsed": time.time() - t0,
    })

    # Update loop variables...
    optimizer = final_state.optimizer
    tracking_params = final_state.tracking_params
    replay_buffer = final_state.replay_buffer

def main():
  num_episodes = 10000

  rng = random.PRNGKey(0)
  np.random.seed(0)

  train_rng, rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  results_dir = Path("results/tmp_ddpg_half_cheetah/")
  results_dir.mkdir()

  def callback(info):
    episode = info["episode"]
    episode_length = info["episode_length"]
    discounted_cumulative_reward = info["discounted_cumulative_reward"]
    undiscounted_cumulative_reward = info["undiscounted_cumulative_reward"]
    current_actor_params, _ = info["optimizer"].value

    print(f"Episode {episode}\t" +
          "disc. reward = {0:.4f}\t".format(discounted_cumulative_reward) +
          "undisc. reward = {0:.4f}\t".format(undiscounted_cumulative_reward) +
          "ep. length = {}\t".format(episode_length) +
          "elapsed = {0:.4f}".format(info["elapsed"]))

    # Periodically evaluate the policy without any action noise.
    if episode % policy_evaluation_frequency == 0:
      curr_policy = jit(deterministic_policy(current_actor_params))

      tic = time.time()
      policy_value = eval_policy(callback_rngs[episode], curr_policy)
      print(f".. policy value = {policy_value}, elapsed = {time.time() - tic}")

    if (episode + 1) % policy_video_frequency == 0:
      tic = time.time()
      film_policy(callback_rngs[episode],
                  curr_policy,
                  filepath=results_dir / f"episode_{episode}.mp4")
      print(f".. saved episode video, elapsed = {time.time() - tic}")

  train(
      train_rng,
      num_episodes,
      lambda t, _: t >= config.episode_length,
      callback,
  )

if __name__ == "__main__":
  main()
