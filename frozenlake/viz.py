import matplotlib.pyplot as plt

import frozenlake

def plot_heatmap(lake: frozenlake.Lake, heat1d):
  im = plt.imshow(lake.reshape(heat1d))

  # Add lake tile labels.
  for i in range(lake.width):
    for j in range(lake.height):
      tile = lake.lake_map[i, j]
      if tile != "F":
        im.axes.text(j, i, tile, {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "color": "white"
        })

  return im
