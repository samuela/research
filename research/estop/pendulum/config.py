"""All of the shared environemnt parameters and configuration."""

from research.estop.pendulum.env import pendulum_environment

gamma = 0.999
episode_length = 1000
max_torque = 0.5

env = pendulum_environment(
    mass=0.1,
    length=1.0,
    gravity=9.8,
    friction=0,
    max_speed=25,
    dt=0.05,
)

state_shape = (4, )
action_shape = (1, )
