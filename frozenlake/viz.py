import matplotlib.pyplot as plt

import frozenlake

def plot_heatmap(env: frozenlake.FrozenLakeEnv, heat1d):
  im = plt.imshow(env.states_reshape(heat1d))

  # Add lake tile labels.
  lake_width, lake_height = env.lake_map.shape
  for i in range(lake_width):
    for j in range(lake_height):
      tile = env.lake_map[i, j]
      if tile != "F":
        im.axes.text(j, i, tile, {
            "horizontalalignment": "center",
            "verticalalignment": "center"
        })

  return im
