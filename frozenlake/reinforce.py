from typing import Optional

from jax import jit
from jax.experimental import optimizers
import numpy as np

import frozenlake

# def softmax(x, axis=None):
#   exp = np.exp(x)
#   return exp / np.sum(exp, axis=axis, keepdims=True)

def softmax(x, axis=None):
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)

class JaxAdam(object):
  def __init__(self, x0, learning_rate):
    self.opt_init, self.opt_update, self.get_params = optimizers.adam(
        step_size=learning_rate)

    self.opt_state = self.opt_init(x0)
    self.iteration = 0

  def update(self, gradient):
    self.opt_state = self.opt_update(self.iteration, gradient, self.opt_state)
    self.iteration += 1

  def get(self):
    return self.get_params(self.opt_state)

class Adam(object):
  def __init__(self, x0, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    self.x = x0
    self.learning_rate = learning_rate
    self.b1 = b1
    self.b2 = b2
    self.eps = eps

    self.m = np.zeros_like(x0)
    self.v = np.zeros_like(x0)
    self.iteration = 0

  def update(self, gradient):
    # self.m = (1 - self.b1) * gradient + self.b1 * self.m
    self.m *= self.b1
    self.m += (1 - self.b1) * gradient

    # self.v = (1 - self.b2) * (gradient**2) + self.b2 * self.v
    self.v *= self.b2
    self.v += (1 - self.b2) * (gradient**2)

    mhat = self.m / (1 - self.b1**(self.iteration + 1))
    vhat = self.v / (1 - self.b2**(self.iteration + 1))
    self.x -= self.learning_rate * mhat / (np.sqrt(vhat) + self.eps)
    self.iteration += 1

  def get(self):
    return self.x

class Momentum(object):
  def __init__(self, x0, learning_rate, mass=0.9):
    self.x = x0
    self.learning_rate = learning_rate
    self.mass = mass

    self.velocity = np.zeros_like(x0)

  def update(self, gradient):
    self.velocity = self.mass * self.velocity - (1.0 - self.mass) * gradient
    self.x += self.learning_rate * self.velocity

  def get(self):
    return self.x

def reinforce_episode(env,
                      gamma: float,
                      optimizer,
                      max_episode_length: Optional[int] = None):
  raw_policy = softmax(optimizer.get(), axis=-1)
  # epsilon_greedy_policy = 0.9 * raw_policy + 0.1 * np.ones(
  #     (env.lake.num_states, frozenlake.NUM_ACTIONS)) / frozenlake.NUM_ACTIONS
  episode, final_state = frozenlake.rollout(
      env, policy=raw_policy, max_episode_length=max_episode_length)
  weighted_rewards = [(gamma**t) * r for t, (_, _, r) in enumerate(episode)]
  # See https://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array.
  Gs = np.cumsum(weighted_rewards[::-1])[::-1]

  grad = np.zeros((env.lake.num_states, frozenlake.NUM_ACTIONS))

  for t, (state, action, _) in enumerate(episode):
    # Do this in-place for speeeeeed!
    grad[:, :] = 0.0
    grad[state, :] -= softmax(optimizer.get()[state, :])
    grad[state, action] += 1.0
    grad *= Gs[t]

    # print(t, Gs[t])
    # print(grad)

    optimizer.update(-grad)

  return episode, final_state

def run_reinforce(env,
                  gamma: float,
                  optimizer,
                  num_episodes: int,
                  policy_evaluation_frequency: int = 10,
                  verbose: bool = True,
                  deleteme_opt_policy=None):
  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(num_episodes):
    episode, _ = reinforce_episode(env,
                                   gamma,
                                   optimizer,
                                   max_episode_length=None)
    # print(f"episode length {len(episode)}")
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = softmax(optimizer.get(), axis=-1)
      V, _ = frozenlake.iterative_policy_evaluation(
          env,
          gamma,
          policy,
          tolerance=1e-6,
          init_V=V,
      )
      policy_reward = np.dot(V, env.initial_state_distribution)

      if verbose:
        print(f"Episode {episode_num}, policy reward: {policy_reward}")
        # print(optimizer.get())
        # print(softmax(optimizer.get(), axis=-1))
        # print(1.0 * env.lake.reshape(
        #     np.argmax(policy, axis=-1) == deleteme_opt_policy))

      states_seen_log.append(states_seen)
      policy_rewards_log.append(policy_reward)

    # if (episode_num + 1) % 1000 == 0:
    #   plt.figure()
    #   viz.plot_heatmap(env, V)
    #   plt.title(f"Episode {episode_num}")
    #   plt.show()

  return states_seen_log, policy_rewards_log
