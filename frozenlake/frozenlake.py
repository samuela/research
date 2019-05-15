# See https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

import math
from typing import Dict, Tuple, List, Callable

import numpy as np
import scipy.optimize

Action = int

# Order is important here because the state transitions rely on +/- 1 mod 4 to
# calculate the next state.
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

NUM_ACTIONS = 4

MAP_4x4 = np.array([["S", "F", "F", "F"], ["F", "H", "F", "H"],
                    ["F", "F", "F", "H"], ["H", "F", "F", "G"]])
MAP_8x8 = np.array([["S", "F", "F", "F", "F", "F", "F", "F"],
                    ["F", "F", "F", "F", "F", "F", "F", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "F"],
                    ["F", "F", "F", "F", "F", "H", "F", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "F"],
                    ["F", "H", "H", "F", "F", "F", "H", "F"],
                    ["F", "H", "F", "F", "H", "F", "H", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "G"]])

class FrozenLakeEnv(object):
  def __init__(self, lake_map, infinite_time: bool):
    self.lake_map = lake_map
    self.infinite_time = infinite_time

    self.lake_width, self.lake_height = self.lake_map.shape
    self.num_states = self.lake_width * self.lake_height
    self._ij_states = [(i, j) for i in range(self.lake_width)
                       for j in range(self.lake_height)]

    self.goal_states = [
        si for si, (i, j) in enumerate(self._ij_states)
        if self.lake_map[i, j] == "G"
    ]
    self.hole_states = [
        si for si, (i, j) in enumerate(self._ij_states)
        if self.lake_map[i, j] == "H"
    ]
    self.start_states = [
        si for si, (i, j) in enumerate(self._ij_states)
        if self.lake_map[i, j] == "S"
    ]
    self.frozen_states = [
        si for si, (i, j) in enumerate(self._ij_states)
        if self.lake_map[i, j] == "F"
    ]
    self.terminal_states = (self.goal_states + self.hole_states
                            if not self.infinite_time else [])
    self.nonterminal_states = [
        i for i in range(self.num_states) if i not in self.terminal_states
    ]

    self.transitions = self._build_transitions()
    self.rewards = self._build_rewards()

    # Start at any start state uniformly at random.
    ss = self.start_states
    assert len(ss) > 0
    self.initial_state_distribution = np.zeros((self.num_states, ))
    self.initial_state_distribution[ss] = 1.0 / len(ss)

  def states_reshape(self, stuff1d):
    stuff2d = np.zeros(self.lake_map.shape)
    for s, v in zip(self._ij_states, stuff1d):
      stuff2d[s] = v
    return stuff2d

  def _build_transitions(self):
    def clip(pseudo_state: Tuple[int, int]) -> int:
      i, j = pseudo_state
      return self._ij_states.index((np.clip(i, 0, self.lake_width - 1),
                                    np.clip(j, 0, self.lake_height - 1)))

    def move(state: int, action: Action) -> int:
      i, j = self._ij_states[state]
      if action == LEFT:
        return clip((i, j - 1))
      elif action == DOWN:
        return clip((i + 1, j))
      elif action == RIGHT:
        return clip((i, j + 1))
      elif action == UP:
        return clip((i - 1, j))
      else:
        raise Exception("bad action")

    transitions = np.zeros((self.num_states, NUM_ACTIONS, self.num_states))
    for si, (i, j) in enumerate(self._ij_states):
      if self.lake_map[i, j] in ["H", "G"]:
        transitions[si, :, si] = 1.0
      else:
        for a in [LEFT, DOWN, RIGHT, UP]:
          # Use += instead of = in the weird situation in which two moves
          # collide.
          transitions[si, a, move(si, (a - 1) % NUM_ACTIONS)] += 1.0 / 3.0
          transitions[si, a, move(si, a)] += 1.0 / 3.0
          transitions[si, a, move(si, (a + 1) % NUM_ACTIONS)] += 1.0 / 3.0

    return transitions

  def _build_rewards(self):
    rewards = np.zeros((self.num_states, NUM_ACTIONS, self.num_states))
    for s in self.goal_states:
      rewards[:, :, s] = 1.0

      if not self.infinite_time:
        # Staying in a goal state means no reward.
        rewards[s, :, s] = 0.0

    return rewards

def expected_rewards(env: FrozenLakeEnv):
  return np.einsum("ijk,ijk->ij", env.transitions, env.rewards)
  # expected_rewards2 = np.sum(transitions * rewards, axis=-1)
  # assert np.allclose(expected_rewards, expected_rewards2)

def value_iteration(env: FrozenLakeEnv, gamma: float, tolerance: float):
  """See Sutton & Barto page 83."""
  V = np.zeros((env.num_states, ))
  Q = np.zeros((env.num_states, NUM_ACTIONS))

  # Seed the values of the goal states with the geometric sum, since we know
  # that's the answer analytically. This only makes sense when we allow
  # ourselves to pick up rewards staying in the goal state forever.
  if env.infinite_time:
    for s in env.goal_states:
      V[s] = 1.0 / (1.0 - gamma)

  expected_r = expected_rewards(env)

  policy_rewards_per_iter = []
  while True:
    Q = expected_r + gamma * np.einsum("ijk,k->ij", env.transitions, V)
    new_state_values = np.max(Q, axis=-1)

    delta = np.abs(V - new_state_values).max()
    policy_reward = np.dot(new_state_values, env.initial_state_distribution)
    # print(
    #     f"Iteration {iteration}, delta: {delta}, policy reward {policy_reward}"
    # )
    V = new_state_values
    policy_rewards_per_iter.append(policy_reward)

    if delta <= tolerance: break

  return Q, policy_rewards_per_iter

def iterative_policy_evaluation(env: FrozenLakeEnv,
                                gamma: float,
                                policy,
                                tolerance: float,
                                init_V=None):
  """See Sutton & Barto page 75."""
  if init_V is None:
    V = np.zeros((env.num_states, ))

    # Seed the values of the goal states with the geometric sum, since we know
    # that's the answer analytically. This only makes sense when we allow
    # ourselves to pick up rewards staying in the goal state forever.
    if env.infinite_time:
      for s in env.goal_states:
        V[s] = 1.0 / (1.0 - gamma)
  else:
    V = init_V

  expected_r = expected_rewards(env)

  policy_rewards_per_iter = []
  while True:
    Q = expected_r + gamma * np.einsum("ijk,k->ij", env.transitions, V)
    new_state_values = np.einsum("ij,ij->i", Q, policy)
    delta = np.abs(V - new_state_values).max()
    policy_reward = np.dot(new_state_values, env.initial_state_distribution)
    V = new_state_values
    policy_rewards_per_iter.append(policy_reward)

    if delta <= tolerance: break

  return V, policy_rewards_per_iter

def markov_chain_stats(env: FrozenLakeEnv, policy_transitions):
  assert policy_transitions.shape == (env.num_states, env.num_states)

  # See https://en.wikipedia.org/wiki/Absorbing_Markov_chain.
  absorbing_states = env.terminal_states
  transient_states = env.nonterminal_states
  t = len(transient_states)

  # See https://stackoverflow.com/questions/19161512/numpy-extract-submatrix.
  Q = policy_transitions[np.ix_(transient_states, transient_states)]
  R = policy_transitions[np.ix_(transient_states, absorbing_states)]
  Ninv = np.eye(t) - Q
  N = np.linalg.inv(Ninv)

  # Calculate the hitting probabilities.
  transient_hp = (N - np.eye(t)) * np.power(np.diag(N), -1)[np.newaxis, :]
  absorbing_hp = np.linalg.solve(Ninv, R)

  # Assuming that the S is the first transient state!
  hitting_prob = np.zeros(env.num_states)
  hitting_prob[transient_states] = transient_hp[0, :]
  hitting_prob[absorbing_states] = absorbing_hp[0, :]

  # Calculate the expected number of steps to absorption.
  esta = np.zeros(env.num_states)
  esta[transient_states] = np.sum(N, axis=1)

  return hitting_prob, esta

def estimate_hitting_probabilities(lake_map, states, policy_transitions):
  pass

def q_learning_episode(env: FrozenLakeEnv,
                       gamma,
                       alpha,
                       Q,
                       meta_policy,
                       max_episode_length: int = 500):
  # Start off by sampling an initial state from the initial_state distribution.
  current_state = np.random.choice(
      env.num_states, p=env.initial_state_distribution)
  episode = []

  for t in range(max_episode_length):
    action = np.random.choice(
        NUM_ACTIONS, p=meta_policy(Q[current_state, :], t))
    next_state = np.random.choice(
        env.num_states, p=env.transitions[current_state, action, :])
    reward = env.rewards[current_state, action, next_state]

    Q[current_state, action] += alpha * (
        reward + gamma * Q[next_state, :].max() - Q[current_state, action])

    episode.append((current_state, action, reward))
    current_state = next_state

    if current_state in env.terminal_states: break

  # `current_state` is now the final state. Reporting it is necessary in order
  # to tell which state the episode actually ended on.
  return Q, episode, current_state

def num_mdp_states(lake_map):
  num_starts = (lake_map == "S").sum()
  num_frozen = (lake_map == "F").sum()
  num_holes = (lake_map == "H").sum()
  num_goals = (lake_map == "G").sum()
  # We can reduce all holes to a single MDP state.
  return num_starts + num_frozen + (num_holes > 0) + num_goals
