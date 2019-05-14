import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

import frozenlake

def plot_heatmap(lake_map, heat2d):
  im = plt.imshow(heat2d)

  # Add lake tile labels.
  lake_width, lake_height = lake_map.shape
  for i in range(lake_width):
    for j in range(lake_height):
      tile = lake_map[i, j]
      if tile != "F":
        im.axes.text(j, i, tile, {
            "horizontalalignment": "center",
            "verticalalignment": "center"
        })

  return im

if __name__ == "__main__":
  lake_map = frozenlake.MAP_8x8
  lake_width, lake_height = lake_map.shape

  # Assuming that the start is the first state!
  initial_state = np.zeros((lake_width * lake_height, ))
  initial_state[0] = 1.0

  gamma = 0.99

  env = frozenlake.FrozenLakeEnv(lake_map)
  state_action_values, policy_rewards_per_iter = frozenlake.value_iteration(
      env, gamma, 1e-4)
  policy = np.argmax(state_action_values, axis=-1)
  state_values = np.max(state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  states_values2d = env.states_reshape(state_values)
  plot_heatmap(lake_map, states_values2d)
  plt.title("Complete map")
  plt.savefig("frozenlake_figs/complete_map.pdf")

  # Show hitting probability map.
  policy_transitions = np.array(
      [env.transitions[i, policy[i], :] for i in range(env.num_states)])
  hp, esta = frozenlake.markov_chain_stats(env, policy_transitions)
  hp2d = env.states_reshape(hp)
  esta2d = env.states_reshape(esta)

  plt.figure()
  plot_heatmap(lake_map, hp2d)
  plt.title("Hitting probabilities")
  plt.savefig("frozenlake_figs/hitting_probabilities.pdf")

  plt.figure()
  plot_heatmap(lake_map, esta2d)
  plt.title("Expected number of states to completion")

  # Show optimal policy on top of hitting probabilities.
  plt.figure()
  im = plt.imshow(hp2d)
  for s, a in zip(env._ij_states, policy):
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
  plt.savefig("frozenlake_figs/optimal_policy.pdf")

  # Show value CDF.
  plt.figure()
  plt.hist(state_values, bins=100, histtype="step", cumulative=True)
  plt.xlabel("V(s)")
  plt.ylabel(f"Number of states (out of {lake_width * lake_height})")
  plt.title("CDF of state values")
  plt.savefig("frozenlake_figs/value_function_cdf.pdf")

  #######

  # New map has hole everywhere with bad prob.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(hp, percentile)
  estop_map[hp2d < threshold] = "H"

  estop_env = frozenlake.FrozenLakeEnv(estop_map)
  estop_state_action_values, estop_policy_rewards_per_iter = frozenlake.value_iteration(
      estop_env, gamma, 1e-4)
  estop_state_values = np.max(estop_state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  plot_heatmap(estop_map, estop_env.states_reshape(estop_state_values))
  plt.title(f"E-stop map ({percentile}% of states removed)")
  plt.savefig("frozenlake_figs/estop_map.pdf")

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
  plt.savefig("frozenlake_figs/convergence_comparison.pdf")

  print(f"Exact solution, policy value: {np.dot(initial_state, state_values)}")
  print(
      f"E-stop solution, policy value: {np.dot(initial_state, estop_state_values)}"
  )

  plt.show()
