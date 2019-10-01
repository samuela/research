from research.estop.gym.gym_wrappers import build_env_spec

spec = build_env_spec("HalfCheetah-v3", reward_adjustment=1.0)
