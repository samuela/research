from research.estop.gym.gym_wrappers import build_env_spec

env_spec = build_env_spec("Ant-v3", reward_adjustment=2.5)
