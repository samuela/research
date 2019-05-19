import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import functools
from multiprocessing import Pool
from pathlib import Path
import pickle

import numpy as np
import tqdm

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

def run_q_learning(
    env: frozenlake.FrozenLakeEnv,
    gamma: float,
    num_episodes: int,
    policy_evaluation_frequency: int = 10,
    verbose: bool = True,
):
  # Initializing to random values is necessary to break ties, preventing the
  # agent from always picking the same action and never getting anywhere.
  Q = np.random.rand(env.lake.num_states, frozenlake.NUM_ACTIONS)

  # This is crucial! There is no positive or negative reward for taking any
  # action in a terminal state. See Sutton & Barto page 131.
  for s in env.terminal_states:
    Q[s, :] = 0.0

  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(num_episodes):
    Q, episode, _ = frozenlake.q_learning_episode(
        env,
        gamma,
        alpha=0.1,
        Q=Q,
        meta_policy=epsilon_greedy(epsilon=0.1),
        # meta_policy=epsilon_greedy_annealed(epsilon=1.0),
        max_episode_length=None)
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = frozenlake.deterministic_policy(env, np.argmax(Q, axis=-1))
      V, _ = frozenlake.iterative_policy_evaluation(
          env, gamma, policy, tolerance=1e-6, init_V=V)
      policy_reward = np.dot(V, env.initial_state_distribution)

      if verbose:
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

def q_learning_job(random_seed: int, env, gamma: float,
                   policy_evaluation_frequency: int, folder: Path):
  np.random.seed(random_seed)

  states_seen, policy_rewards = run_q_learning(
      env,
      gamma,
      policy_evaluation_frequency=policy_evaluation_frequency,
      num_episodes=10,
      verbose=False)

  with (folder / f"seed={random_seed}.pkl").open(mode="wb") as f:
    pickle.dump({
        "states_seen": states_seen,
        "policy_rewards": policy_rewards
    }, f)

  # print(f"Finished job for random seed {random_seed}.")

def main():
  np.random.seed(0)

  def build_env(lake: frozenlake.Lake):
    # return frozenlake.FrozenLakeEnv(lake, infinite_time=True)
    return frozenlake.FrozenLakeWithEscapingEnv(
        lake, hole_retention_probability=0.99)

  lake_map = frozenlake.MAP_8x8
  policy_evaluation_frequency = 10
  gamma = 0.99
  num_random_seeds = 100

  results_dir = Path("qlearning_pkls")
  estop_results_dir = results_dir / "estop"
  full_results_dir = results_dir / "full"
  results_dir.mkdir()
  estop_results_dir.mkdir()
  full_results_dir.mkdir()

  pool = Pool()

  # Build the full environment and run value iteration to calculate the optimal
  # policy.
  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  state_action_values, _ = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)
  state_values = np.max(state_action_values, axis=-1)
  optimal_policy_reward = np.dot(state_values, env.initial_state_distribution)

  # Estimate hitting probabilities.
  optimal_policy = frozenlake.deterministic_policy(
      env, np.argmax(state_action_values, axis=-1))
  estimated_hp = frozenlake.estimate_hitting_probabilities(
      env, optimal_policy, num_rollouts=1000)
  estimated_hp2d = lake.reshape(estimated_hp)

  # Build e-stop environment.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  estop_map[estimated_hp2d <= threshold] = "E"

  estop_lake = frozenlake.Lake(estop_map)
  estop_env = build_env(estop_lake)

  # pickle dump the environemnt setup/metadata...
  pickle.dump({
      "lake_map": lake_map,
      "policy_evaluation_frequency": policy_evaluation_frequency,
      "gamma": gamma,
      "num_random_seeds": num_random_seeds,
      "lake": lake,
      "env": env,
      "state_action_values": state_action_values,
      "state_values": state_values,
      "optimal_policy_reward": optimal_policy_reward,
      "optimal_policy": optimal_policy,
      "estimated_hp": estimated_hp,
      "estimated_hp2d": estimated_hp2d,
      "estop_map": estop_map,
      "percentile": percentile,
      "threshold": threshold,
      "estop_lake": estop_lake,
      "estop_env": estop_env,
  }, (results_dir / "metadata.pkl").open(mode="wb"))

  # plt.figure()
  # viz.plot_heatmap(estop_lake, np.zeros(estop_lake.num_states))
  # plt.title("E-stop map")
  # plt.show()

  # Run Q-learning on the full environment.
  for _ in tqdm.tqdm(
      pool.imap_unordered(
          functools.partial(
              q_learning_job,
              env=env,
              gamma=gamma,
              policy_evaluation_frequency=policy_evaluation_frequency,
              folder=estop_results_dir,
          ), range(num_random_seeds)),
      desc="full",
      total=num_random_seeds):
    pass

  # Run Q-learning on the e-stop environment.
  for _ in tqdm.tqdm(
      pool.imap_unordered(
          functools.partial(
              q_learning_job,
              env=estop_env,
              gamma=gamma,
              policy_evaluation_frequency=policy_evaluation_frequency,
              folder=estop_results_dir,
          ), range(num_random_seeds)),
      desc="estop",
      total=num_random_seeds):
    pass

  ### Plotting
  # plt.figure()
  # plt.plot(
  #     policy_evaluation_frequency * np.arange(len(full_map_policy_rewards)),
  #     full_map_policy_rewards)
  # plt.plot(policy_evaluation_frequency * np.arange(len(estop_policy_rewards)),
  #          estop_policy_rewards)
  # plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  # plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  # plt.xlabel("Episode")
  # plt.ylabel("Policy reward")
  # plt.savefig("figs/q_learning_per_episode.pdf")

  # plt.figure()
  # plt.plot(full_map_states_seen, full_map_policy_rewards)
  # plt.plot(estop_states_seen, estop_policy_rewards)
  # plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  # plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  # plt.xlabel("Number of states seen")
  # plt.ylabel("Policy reward")
  # # plt.title("Q-learning on the complete environment")
  # plt.savefig("figs/q_learning_per_states_seen.pdf")

  # plt.show()

if __name__ == "__main__":
  progress_bar = None
  main()
