import time

from jax import jit, lax, random
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

tau = 1e-4
buffer_size = 2**15
batch_size = 64
opt_init = make_optimizer(optimizers.adam(step_size=1e-3))
init_noise = Normal(jp.array(0.0), jp.array(0.1))
noise = lambda _1, _2: Normal(jp.array(0.0), jp.array(0.1))

actor_init, actor = stax.serial(
    Dense(64),
    Relu,
    Dense(1),
    Tanh,
    stax.elementwise(lambda x: config.max_torque * x),
)

critic_init, critic = stax.serial(
    FanInConcat(),
    Dense(64),
    Relu,
    Dense(64),
    Relu,
    Dense(1),
    stax.elementwise(lambda x: x + config.reward_adjustment / (1 - config.gamma
                                                               )),
    Scalarify,
)

def train(rng, num_episodes, terminal_criterion, callback):
  actor_init_rng, critic_init_rng, rng = random.split(rng, 3)
  _, init_actor_params = actor_init(actor_init_rng, config.state_shape)
  _, init_critic_params = critic_init(
      critic_init_rng, (config.state_shape, config.action_shape))
  optimizer = opt_init((init_actor_params, init_critic_params))
  tracking_params = optimizer.value

  replay_buffer = ddpg.ReplayBuffer(
      states=jp.zeros((buffer_size, ) + config.state_shape),
      actions=jp.zeros((buffer_size, ) + config.action_shape),
      rewards=jp.zeros((buffer_size, )),
      next_states=jp.zeros((buffer_size, ) + config.state_shape),
      done=jp.zeros((buffer_size, ), dtype=bool),
      count=0,
  )

  run = jit(
      ddpg.ddpg_episode(
          config.env,
          config.gamma,
          tau,
          actor,
          critic,
          noise,
          terminal_criterion,
          batch_size,
      ))

  episode_rngs = random.split(rng, num_episodes)

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
  train_rng, rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  reward_per_episode = []

  def callback(info):
    episode = info['episode']
    reward = info['reward']
    print(f"Episode {episode}, "
          f"reward = {reward}, "
          f"elapsed = {info['elapsed']}")
    reward_per_episode.append(reward)

    # if episode == num_episodes - 1:
    if episode % 500 == 0 or episode == num_episodes - 1:
      for rollout in range(5):
        states, actions = ddpg.rollout(
            random.fold_in(callback_rngs[episode], rollout),
            config.env,
            lambda s: Deterministic(actor(info["optimizer"].value[0], s)),
            num_timesteps=250,
        )
        viz_pendulum_rollout(states, 2 * actions / config.max_torque)

  train(
      train_rng,
      num_episodes,
      lambda t, _: lax.ge(t, config.episode_length),
      callback,
  )

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(reward_per_episode)
  plt.xlabel("Episode")
  plt.ylabel("Train episode reward, including action noise")
  plt.show()

if __name__ == "__main__":
  main()
