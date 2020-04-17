"""A replacement for plt.show() that watermarks all your plots with QR codes
linked to reproducible code.

The code of the script or module that is currently running is uploaded to a
GitHub gist, along with all the plots, and console output. All plots are
watermarked with QR codes that link to the GitHub gist with your code. Email
collaborators plots that have the code built-in!

Almost idiot-proof."""

from datetime import datetime
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import matplotlib.pyplot as plt
import qrcode
import requests
from simplegist import Simplegist  # should be https://github.com/samuela/simplegist
import numpy as np

# Do this at import time, so foolish hoomans have less of a chance of editing
# the file while the script is running but before we save plots.
current_script_path = Path(sys.argv[0])
current_script_contents = current_script_path.read_text()

# Start copying stdout/stderr to a log file.
# See https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file.
logfile = tempfile.NamedTemporaryFile()
tee = subprocess.Popen(["tee", logfile.name], stdin=subprocess.PIPE)
# Cause tee's stdin to get a copy of our stdin/stdout (as well as that of any
# child processes we spawn).
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def show():
  sg = Simplegist(username=os.environ["GITHUB_USERNAME"], api_token=os.environ["SIMPLEGIST_TOKEN"])
  gist_name = str(datetime.now()) + " " + current_script_path.name
  gist = sg.create(name=gist_name, content=current_script_contents, public=True)
  gist_url = gist["Gist-Link"]
  gist_id = gist["id"]

  qr = qrcode.make(gist_url).resize((100, 100))

  fig_urls = []
  for fignum in plt.get_fignums():
    fig = plt.figure(num=fignum)
    fig.figimage(1 - np.array(qr), xo=0, yo=fig.bbox.ymax - qr.size[1], alpha=0.5, cmap="binary")

    # Save the fig and upload it. Note that we don't use pdf here because GitHub
    # comments don't support it. Doesn't work with img.onl either.
    plot_path = Path.home() / ".blt" / "gallery" / f"{gist_name} {fignum}.png"
    plt.savefig(plot_path, format="png")
    resp = requests.post("https://img.onl/api/upload.php", files={
        "imgFile": open(plot_path, "rb")
    }).json()
    assert resp["success"]
    fig_urls.append(resp["url"])

  # Post a comment to the gist with the stdout/err and upload figures as images.
  comment_header = "Powered by [blt.py](https://gist.github.com/samuela/fb2af385b46ab8640bbb54e25f6b6b38)"
  logfile.seek(0)
  console_output = logfile.read().decode(sys.stdout.encoding)
  console_output_section = f"<details><summary>Console output</summary>\n\n\n```\n{console_output}\n```\n</details>"
  plots_section = "\n".join([f"<img src='{url}'>" for url in fig_urls])
  sg.comments().create(id=gist_id,
                       body=f"{comment_header}\n\n{console_output_section}\n\n{plots_section}")

  print(f"[blt] results dumped: {gist_url}")
  plt.show()
