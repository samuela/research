"""A replacement for plt.show() that watermarks all your plots with QR codes
linked to reproducible code.

The code of the script or module that is currently running is uploaded to a
GitHub gist, along with all the plots, and console output. All plots are
watermarked with QR codes that link to the GitHub gist with your code. Email
collaborators plots that have the code built-in!

Almost idiot-proof.

Setup:
 * Env var GITHUB_GIST_TOKEN must be set to an API token with the `gist` scope.
 * GitHub SSH git pulls and pushes from this machine must be set up.
"""

from datetime import datetime
import os
import pickle
import subprocess
import sys
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import qrcode
import requests
import numpy as np

# A global (ugh) store of things that we'd like recorded by blt. Use
# `blt.remember()` to update entries.
_STUFF_TO_REMEMBER = {}

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

def _create_gist(description, files):
  """Create a new gist. `description` is basically the name."""

  # Note that the v4 graphql API does not yet support creating new gists.
  resp = requests.post("https://api.github.com/gists",
                       json={
                           "public": True,
                           "description": description,
                           "files": files
                       },
                       headers={"Authorization": "token " + os.environ["GITHUB_GIST_TOKEN"]})
  resp.raise_for_status()
  return resp.json()

def _add_gist_files(gist_id, files):
  # Unfortunately it's not possible to add binary files to a gist with the API,
  # and there's sign that it'll be supported anytime soon. See
  # https://stackoverflow.com/questions/9770562/posting-images-to-gists-via-the-http-api.
  with tempfile.TemporaryDirectory() as clonedir:
    # Send stderr to a pipe in order to avoid showing it to the user.
    subprocess.check_output(["git", "clone", f"git@gist.github.com:{gist_id}.git", clonedir],
                            stderr=subprocess.PIPE)

    for filename, path in files.items():
      subprocess.check_output(["cp", path, os.path.join(clonedir, filename)])

    subprocess.check_output(["git", "add"] + [fn for fn, _ in files.items()], cwd=clonedir)
    subprocess.check_output(["git", "commit", "-m", "add files"], cwd=clonedir)
    subprocess.check_output(["git", "push"], cwd=clonedir, stderr=subprocess.PIPE)

def remember(kvs):
  # pylint: disable=global-statement
  global _STUFF_TO_REMEMBER
  _STUFF_TO_REMEMBER.update(kvs)

def show():
  logfile.seek(0)
  console_output = logfile.read().decode(sys.stdout.encoding)

  # TODO: add git status here to the metadata dump.
  metadata = "Powered by [blt.py](https://gist.github.com/samuela/fb2af385b46ab8640bbb54e25f6b6b38)"

  gist_name = str(datetime.now()) + " " + current_script_path.name

  # See https://gist.github.com/fliedonion/6057f4a3a533f7992c60 for an
  # explanation of gist file ordering rules. Basically: it's alphabetical.
  create_gist_resp = _create_gist(
      gist_name, {
          "A_metadata.md": {
              "content": metadata
          },
          f"C_{current_script_path.name}": {
              "content": current_script_contents
          },
          "D_console_output.log": {
              "content": console_output
          }
      })
  gist_id = create_gist_resp["id"]
  gist_url = create_gist_resp["html_url"]

  qr = qrcode.make(gist_url).resize((100, 100))

  fig_paths = {}
  for fignum in plt.get_fignums():
    fig = plt.figure(num=fignum)
    fig.figimage(1 - np.array(qr), xo=0, yo=fig.bbox.ymax - qr.size[1], alpha=0.5, cmap="binary")

    # Save the fig and upload it. Note that we don't use pdf here because GitHub
    # comments don't support it. Doesn't work with img.onl either.
    plot_path = Path.home() / ".blt" / "gallery" / f"{gist_name} {fignum}.pdf"
    plt.savefig(plot_path, format="pdf")
    fig_paths[fignum] = plot_path

  with tempfile.NamedTemporaryFile() as stuff_file:
    # Dump stuff to remember into a stuff to remember pkl file.
    pickle.dump(_STUFF_TO_REMEMBER, stuff_file)
    stuff_file.seek(0)
    figs_files = {f"B_fig{fignum}.pdf": path for (fignum, path) in fig_paths.items()}
    _add_gist_files(gist_id, {"E_remembered_stuff.pkl": stuff_file.name, **figs_files})

  # Updating a gist does not alter its html url.
  print(f"[blt] results dumped: {gist_url}")
  plt.show()
