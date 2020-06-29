import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == "__main__":
  T = 1000

  gymenv = PendulumEnv()
  gymenv.reset()
  # Force the initialization.
  gymenv.state = [np.pi, 0]
  gymenv.last_u = None
  video = VideoRecorder(gymenv, path="openai_gym_constant_torque.mp4")

  for t in range(T):
    gymenv.step([-2])
    gymenv.render()
    video.capture_frame()

  gymenv.close()
  video.close()
