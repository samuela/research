import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

import frozenlake
import viz

if __name__ == "__main__":
  lake_map = frozenlake.XMAP_9x9
  infinite_time = False

  gamma = 0.99

  env = frozenlake.FrozenLakeEnv(lake_map, infinite_time)
  state_action_values, policy_rewards_per_iter = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)
  policy_actions = np.argmax(state_action_values, axis=-1)
  state_values = np.max(state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  viz.plot_heatmap(env, state_values)
  plt.title("Value function on the full environment")
  plt.savefig("figs/value_function_full_env.pdf")

  # Show hitting probability map.
  policy_transitions = np.array([
      env.transitions[i, policy_actions[i], :] for i in range(env.num_states)
  ])
  hp, esta = frozenlake.markov_chain_stats(env, policy_transitions)
  hp2d = env.states_reshape(hp)

  plt.figure()
  viz.plot_heatmap(env, hp)
  plt.title("Hitting probabilities")
  plt.savefig("figs/hitting_probabilities.pdf")

  # Show estimated hitting probability map.
  estimated_hp = frozenlake.estimate_hitting_probabilities(
      env,
      frozenlake.deterministic_policy(env, policy_actions),
      num_rollouts=1000)
  plt.figure()
  viz.plot_heatmap(env, estimated_hp)
  plt.title("Estimated hitting probabilities")

  plt.figure()
  viz.plot_heatmap(env, esta)
  plt.title("Expected number of states to completion")

  # Show optimal policy on top of hitting probabilities.
  plt.figure()
  im = plt.imshow(hp2d)
  for s, a in zip(env._ij_states, policy_actions):
    i, j = s
    if a == 0:
      arrow = "←"
    elif a == 1:
      arrow = "↓"
    elif a == 2:
      arrow = "→"
    elif a == 3:
      arrow = "↑"
    else:
      raise Exception("bad bad bad")

    im.axes.text(j, i, arrow, {
        "horizontalalignment": "center",
        "verticalalignment": "center"
    })
  plt.title("Optimal policy overlayed on hitting probabilities")
  plt.savefig("figs/optimal_policy.pdf")

  # Show value CDF.
  plt.figure()
  plt.hist(state_values, bins=100, histtype="step", cumulative=True)
  plt.xlabel("V(s)")
  plt.ylabel(f"Number of states (out of {env.lake_width * env.lake_height})")
  plt.title("CDF of state values")
  plt.savefig("figs/value_function_cdf.pdf")

  #######

  # New map has hole everywhere with bad prob.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  # Use less than or equal because the estimated hitting probabilities can be
  # zero and the threshold can be zero, so nothing on the map changes.
  estop_map[env.states_reshape(estimated_hp) <= threshold] = "E"

  estop_env = frozenlake.FrozenLakeEnv(estop_map, infinite_time)
  estop_state_action_values, estop_policy_rewards_per_iter = frozenlake.value_iteration(
      estop_env, gamma, tolerance=1e-6)
  estop_state_values = np.max(estop_state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  viz.plot_heatmap(estop_env, estop_state_values)
  plt.title(f"E-stop map ({percentile}% of states removed)")
  plt.savefig("figs/estop_map.pdf")

  # Show policy rewards per iter
  # There are 4 S * A * S FLOPS in each iteration:
  #   * multiplying transitions with state_values
  #   * multiplying times gamma
  #   * adding expected_rewards
  #   * max'ing over state_action_values

  plt.figure()
  plt.plot(
      4 * (frozenlake.NUM_ACTIONS * (frozenlake.num_mdp_states(lake_map)**2)) *
      np.arange(len(policy_rewards_per_iter)), policy_rewards_per_iter)
  plt.plot(
      4 * (frozenlake.NUM_ACTIONS * (frozenlake.num_mdp_states(estop_map)**2))
      * np.arange(len(estop_policy_rewards_per_iter)),
      estop_policy_rewards_per_iter)
  plt.xlabel("FLOPS")
  plt.ylabel("Policy reward")
  plt.legend(["Full MDP", "E-stop MDP"])
  plt.title("Convergence comparison")
  plt.savefig("figs/convergence_comparison.pdf")

  print(
      f"Exact solution, policy value: {np.dot(env.initial_state_distribution, state_values)}"
  )
  print(
      f"E-stop solution, policy value: {np.dot(env.initial_state_distribution, estop_state_values)}"
  )

  plt.show()
