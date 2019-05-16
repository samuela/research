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

def run_q_learning(env: frozenlake.FrozenLakeEnv,
                   gamma: float,
                   policy_evaluation_frequency: int = 10):
  # Initializing to random values is necessary to break ties, preventing the
  # agent from always picking the same action and never getting anywhere.
  Q = 1e-2 * np.random.randn(env.num_states, frozenlake.NUM_ACTIONS)

  # This is crucial! There is no positive or negative reward for taking any
  # action in a terminal state. See Sutton & Barto page 131.
  for s in env.terminal_states:
    Q[s, :] = 0.0

  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(5000):
    Q, episode, _ = frozenlake.q_learning_episode(
        env,
        gamma,
        alpha=0.1,
        Q=Q,
        meta_policy=epsilon_greedy(epsilon=0.1),
        # meta_policy=epsilon_greedy_annealed(epsilon=1.0),
    )
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = frozenlake.deterministic_policy(env, np.argmax(Q, axis=-1))
      V, _ = frozenlake.iterative_policy_evaluation(
          env, gamma, policy, tolerance=1e-6, init_V=V)
      policy_reward = np.dot(V, env.initial_state_distribution)
      print(f"Episode {episode_num}, policy reward: {policy_reward}")

      states_seen_log.append(states_seen)
      policy_rewards_log.append(policy_reward)

    # if (episode_num + 1) % 1000 == 0:
    #   V = np.max(Q, axis=-1)
    #   plt.figure()
    #   viz.plot_heatmap(env, V)
    #   plt.title(f"Episode {episode_num}")
    #   plt.show()

  return states_seen_log, policy_rewards_log

if __name__ == "__main__":
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  policy_evaluation_frequency = 10
  gamma = 0.99

  env = frozenlake.FrozenLakeEnv(lake_map, infinite_time=False)
  state_action_values, _ = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)
  state_values = np.max(state_action_values, axis=-1)
  optimal_policy_reward = np.dot(state_values, env.initial_state_distribution)

  # E-stop environment.
  optimal_policy = frozenlake.deterministic_policy(
      env, np.argmax(state_action_values, axis=-1))
  estimated_hp = frozenlake.estimate_hitting_probabilities(
      env, optimal_policy, num_rollouts=1000)
  estimated_hp2d = env.states_reshape(estimated_hp)

  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  estop_map[estimated_hp2d < threshold] = "H"

  estop_env = frozenlake.FrozenLakeEnv(estop_map, infinite_time=False)
  estop_states_seen, estop_policy_rewards = run_q_learning(
      estop_env,
      gamma,
      policy_evaluation_frequency=policy_evaluation_frequency)

  # Full environment.
  full_map_states_seen, full_map_policy_rewards = run_q_learning(
      env, gamma, policy_evaluation_frequency=policy_evaluation_frequency)

  ### Plotting
  plt.figure()
  viz.plot_heatmap(estop_env, np.zeros(estop_env.num_states))
  plt.title("E-stop map")

  plt.figure()
  plt.plot(
      policy_evaluation_frequency * np.arange(len(full_map_policy_rewards)),
      full_map_policy_rewards)
  plt.plot(policy_evaluation_frequency * np.arange(len(estop_policy_rewards)),
           estop_policy_rewards)
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  plt.xlabel("Episode")
  plt.ylabel("Policy reward")
  plt.savefig("figs/q_learning_per_episode.pdf")

  plt.figure()
  plt.plot(full_map_states_seen, full_map_policy_rewards)
  plt.plot(estop_states_seen, estop_policy_rewards)
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  plt.xlabel("Number of states seen")
  plt.ylabel("Policy reward")
  # plt.title("Q-learning on the complete environment")
  plt.savefig("figs/q_learning_per_states_seen.pdf")

  plt.show()
