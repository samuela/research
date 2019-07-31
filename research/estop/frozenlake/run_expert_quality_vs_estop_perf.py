import matplotlib.pyplot as plt
import tqdm
import numpy as np

from research.estop.frozenlake import frozenlake, viz

def build_env(l: frozenlake.Lake):
  return frozenlake.FrozenLakeWithEscapingEnv(l,
                                              hole_retention_probability=0.99)

if __name__ == "__main__":
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  gamma = 0.99

  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  num_states_to_remove = 0.5 * lake.num_states

  def estop_map_optimal_policy_value(hp):
    # See https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy.
    rank_hp2d = lake.reshape(np.argsort(np.argsort(hp)))

    estop_map = np.copy(lake_map)
    estop_map[rank_hp2d < num_states_to_remove] = "E"

    # Check that we haven't gotten rid of the start state yet.
    if (estop_map == "S").sum() == 0:
      # This could also just be recorded as zero, depending on how you want to
      # think about it.
      return None

    estop_env = build_env(frozenlake.Lake(estop_map))
    return frozenlake.optimal_policy_reward(estop_env, gamma)

  expert_policy_values = []
  estop_policy_values = []

  def callback(info):
    policy_value = info["policy_reward"]
    Q = np.copy(info["Q"])
    print(f"Episode {info['iteration']} policy value: {policy_value}")

    if policy_value > 0.0:
      policy_actions = np.argmax(Q, axis=-1)

      # Calculate the value of the optimal policy in the exact e-stop environment.
      policy_transitions = np.array([
          env.transitions[i, policy_actions[i], :]
          for i in range(lake.num_states)
      ])
      exact_hp, _ = frozenlake.markov_chain_stats(env, policy_transitions)
      opt_estop_policy_value = estop_map_optimal_policy_value(exact_hp)

      expert_policy_values.append(policy_value)
      estop_policy_values.append(opt_estop_policy_value)

  _, _ = frozenlake.value_iteration(env,
                                    gamma,
                                    tolerance=1e-6,
                                    callback=callback)

  expert_policy_values = np.array(expert_policy_values)
  estop_policy_values = np.array(estop_policy_values)

  plt.rcParams.update({"font.size": 16})
  plt.figure()
  plt.plot(expert_policy_values, estop_policy_values)
  plt.plot([0, np.max(expert_policy_values)],
           [0, np.max(expert_policy_values)],
           color="grey",
           linestyle="--")
  plt.xlabel("Expert policy cumulative reward")
  plt.ylabel("E-stop policy cumulative reward")
  plt.tight_layout()
  plt.savefig("figs/expert_quality_vs_estop_perf.pdf")
