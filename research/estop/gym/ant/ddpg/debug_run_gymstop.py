from research.estop.gym.ddpg_training import debug_run, make_default_ddpg_train_config
from research.estop.gym.ant import env_name, reward_adjustment
from research.estop.gym.gym_wrappers import build_env_spec

env_spec = build_env_spec(env_name, reward_adjustment)
debug_run(env_spec, make_default_ddpg_train_config(env_spec), respect_gym_done=True)
