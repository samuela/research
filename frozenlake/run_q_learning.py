import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

import frozenlake
import viz

def epsilon_greedy(epsilon: float):
  def h(action_values, t: int):
    # With prob. epsilon we pick a non-greedy action uniformly at random. There
    # are NUM_ACTIONS - 1 non-greedy actions.
    p = epsilon / (frozenlake.NUM_ACTIONS - 1) * np.ones(action_values.shape)
    p[np.argmax(action_values)] = 1 - epsilon
    return p

  return h

def epsilon_greedy_annealed(epsilon: float):
  def h(action_values, t: int):
    # With prob. epsilon we pick a non-greedy action uniformly at random. There
    # are NUM_ACTIONS - 1 non-greedy actions.
    p = epsilon / (t + 1) / (frozenlake.NUM_ACTIONS - 1) * np.ones(
        action_values.shape)
    p[np.argmax(action_values)] = 1 - epsilon / (t + 1)
    return p

  return h

def optimal_policy_reward(env: frozenlake.FrozenLakeEnv):
  state_action_values, _ = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)
  state_values = np.max(state_action_values, axis=-1)
  return np.dot(state_values, env.initial_state_distribution)

if __name__ == "__main__":
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  env = frozenlake.FrozenLakeEnv(lake_map, infinite_time=False)
  Q = 1e-2 * np.random.randn(env.num_states, frozenlake.NUM_ACTIONS)
  for s in env.terminal_states:
    Q[s, :] = 0.0

  gamma = 0.99

  ### Q-learning
  # We use this to warm start iterative policy evaluation.
  V = None

  policy_rewards = []
  for episode_num in range(5000):
    Q, episode, final_state = frozenlake.q_learning_episode(
        env,
        gamma,
        alpha=0.1,
        Q=Q,
        meta_policy=epsilon_greedy(epsilon=0.1),
        # meta_policy=epsilon_greedy_annealed(epsilon=1.0),
    )

    if episode_num % 100 == 0:
      # See https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
      policy = np.zeros((env.num_states, frozenlake.NUM_ACTIONS))
      policy[np.arange(env.num_states), np.argmax(Q, axis=-1)] = 1.0

      V, _ = frozenlake.iterative_policy_evaluation(
          env, gamma, policy, tolerance=1e-6, init_V=V)
      policy_reward = np.dot(V, env.initial_state_distribution)
      print(f"Episode {episode_num}, policy reward: {policy_reward}")

      policy_rewards.append(policy_reward)

    # if (episode_num + 1) % 10000 == 0:
    #   V = np.max(Q, axis=1)
    #   plt.figure()
    #   viz.plot_heatmap(env, V)
    #   plt.title(f"Episode {episode_num}")
    #   plt.show()

  plt.figure()
  plt.plot(100 * np.arange(len(policy_rewards)), policy_rewards)
  plt.axhline(optimal_policy_reward(env), color="grey", linestyle="--")
  plt.legend(["Q-learning", "Optimal policy"])
  plt.xlabel("Episode")
  plt.ylabel("Policy reward")
  plt.title("Q-learning on the complete environment")
  plt.savefig("frozenlake_figs/q_learning_full_env.pdf")

  plt.show()
