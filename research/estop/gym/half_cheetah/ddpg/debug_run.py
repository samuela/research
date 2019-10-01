from research.estop.gym.ddpg_training import debug_run, make_default_ddpg_train_config
from research.estop.gym.half_cheetah import env_spec

debug_run(env_spec, make_default_ddpg_train_config(env_spec))
