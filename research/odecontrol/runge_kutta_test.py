import matplotlib.pyplot as plt
from research.odecontrol.pendulum import pendulum_dynamics
import jax.numpy as jp
from jax.experimental import ode

if __name__ == "__main__":
  total_secs = 60

  dynamics = pendulum_dynamics(
      mass=0.1,
      length=1.0,
      gravity=9.8,
      friction=0.1,
  )

  print("Solving ODEs...")
  states_sequences = [
      ode.odeint(lambda state, t: dynamics(state, jp.zeros((1, ))),
                 y0=jp.array([jp.pi - 1e-1, 0.0]),
                 t=jp.linspace(0, total_secs, num=num)) for num in [2, 10, 100]
  ]
  print(f"... and done")

  plt.figure()
  for states in states_sequences:
    print(states[-1, :])
    plt.plot(states[:, 0], states[:, 1], marker=None)
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.show()
