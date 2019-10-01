import gym
from jax import random
import numpy as np

from research.estop.ddpg import Env
from research.statistax import Deterministic, SampleOnly

def openai_gym_env(construct_env, reward_adjustment: float = 0.0) -> Env:
  """A correct, safer wrapper of an OpenAI gym environment."""
  # Gym environment classes are not valid jax types so they don't play nicely
  # with things like `lax.fori_loop`.
  gym_env = construct_env()

  def init(rng):
    gym_env.seed(int(random.randint(rng, (), 0, 1e6)))
    return gym_env.reset()

  observed_rewards = {}

  def step(state, action):
    # Assert that state matches the current state of gym_env.
    assert np.allclose(state, gym_env.env._get_obs())

    obs_before = state
    obs_after, reward, _done, _info = gym_env.step(action)
    observed_rewards[(str(obs_before), str(action),
                      str(obs_after))] = reward + reward_adjustment
    return Deterministic(obs_after)

  def reward(s1, a, s2):
    # We make the assumption that we only ever calculate rewards on transitions
    # that we've already seen and added to `observed_rewards`.
    return observed_rewards[(str(s1), str(a), str(s2))]

  return Env(SampleOnly(init), step, reward)

def unsafe_openai_gym_env(construct_env,
                          reward_adjustment: float = 0.0) -> Env:
  # Gym environment classes are not valid jax types so they don't play nicely
  # with things like `lax.fori_loop`.
  gym_env = construct_env()

  def init(rng):
    gym_env.seed(int(random.randint(rng, (), 0, 1e6)))
    return gym_env.reset()

  last_reward = [0.0]

  def step(_, action):
    obs_after, reward, _done, _info = gym_env.step(action)
    last_reward[0] = reward + reward_adjustment

    return Deterministic(obs_after)

  def reward(_s1, _a, _s2):
    # We make the assumption that we only ever calculate rewards based on the
    # very last transition seen.
    return last_reward[0]

  return Env(SampleOnly(init), step, reward), gym_env

gamma = 0.99
episode_length = 1000

# env = openai_gym_env(lambda: gym.make("HalfCheetah-v3"))

# See https://github.com/facebook/pyre-check/issues/211.
# pyre-ignore
env, _gym_env = unsafe_openai_gym_env(lambda: gym.make("HalfCheetah-v3"),
                                      reward_adjustment=1.0)

# You can get these values from `state_space` and `action_shape` on the OpenAI
# gym environments.
state_shape = (17, )
action_shape = (6, )

# rng = random.PRNGKey(0)
# s = env.initial_distribution.sample(rng)
# a = random.normal(rng, action_shape)
# s2 = env.step(s, a).sample(rng)
# r = env.reward(s, a, s2)
