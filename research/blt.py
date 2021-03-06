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

import os
import pickle
import socket
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qrcode
import requests

### Setup
# The time we got imported... Hopefully about the same time the script was run.
_start_time = datetime.now()

# For long running jobs, we can't trust these things to be the same when the job
# ends vs when it began!
_git_log = subprocess.check_output(["git", "log", "-1"]).decode("utf-8")
_git_status = subprocess.check_output(["git", "status"]).decode("utf-8")
_git_diff = subprocess.check_output(["git", "--no-pager", "diff"]).decode("utf-8")

# Do this at import time, so foolish hoomans have less of a chance of editing
# the file while the script is running but before we save plots.
_current_script_path = Path(sys.argv[0])
_current_script_contents = _current_script_path.read_text()

### Mutable pieces
# A global (ugh) store of things that we'd like recorded by blt. Use
# `blt.remember()` to update entries.
_STUFF_TO_REMEMBER = {}

# Start copying stdout/stderr to a log file.
# See https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file.
_logfile = tempfile.NamedTemporaryFile()
_tee = subprocess.Popen(["tee", _logfile.name], stdin=subprocess.PIPE)
# Cause _tee's stdin to get a copy of our stdin/stdout (as well as that of any
# child processes we spawn).
os.dup2(_tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(_tee.stdin.fileno(), sys.stderr.fileno())

# Look this up here, so that we fail fast at import time if the necessary env
# vars are not present. This way a job doesn't run for days only to fail at the
# end on `blt.show()`. That would be sad.
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]

# The set of environment variables that are secret and should not be recorded.
# Users should append things to this list that they would like redacted from blt
# reports.
SECRET_ENV_VARS = ["GITHUB_GIST_TOKEN"]

# Lower-case words that are suspect if found in env vars.
_suspect_env_var_words = ["token", "pass", "password", "secret", "key"]

def _get_environ():
  env = dict(os.environ)
  secret_keys_lower = [k.lower() for k in SECRET_ENV_VARS]
  for k in env:
    k_lower = k.lower()
    if k_lower in secret_keys_lower:
      env[k] = "***REDACTED***"
    elif any(word in k_lower for word in _suspect_env_var_words):
      env[k] = "***REDACTED***"
      warnings.warn(f"Env var {k} looks suspicious, so it's been redacted.")
  return env

# It's technically possible for Python to modify its own environ, so we do this
# at module load time.
_environ = _get_environ()

def _create_gist(description, files):
  """Create a new gist. `description` is basically the name."""

  # Note that the v4 graphql API does not yet support creating new gists.
  resp = requests.post("https://api.github.com/gists",
                       json={
                           "public": True,
                           "description": description,
                           "files": files
                       },
                       headers={"Authorization": "token " + GITHUB_GIST_TOKEN})
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
  finished_time = datetime.now()
  gist_name = str(finished_time) + " " + _current_script_path.name

  _logfile.seek(0)
  console_output = _logfile.read().decode(sys.stdout.encoding)

  metadata = f"""
Powered by [blt.py](https://gist.github.com/samuela/fb2af385b46ab8640bbb54e25f6b6b38)

```
Script/module: {_current_script_path}
Hostname: {socket.gethostname()}
Started: {_start_time}
Finished: {finished_time}
Elapsed: {finished_time - _start_time}
Remembered keys: {list(_STUFF_TO_REMEMBER.keys())}
Command: {sys.argv}
Env vars: {_environ}
```
```
{_git_log}
```
```
{_git_status}
```
"""

  # See https://gist.github.com/fliedonion/6057f4a3a533f7992c60 for an
  # explanation of gist file ordering rules. Basically: it's alphabetical.
  # Note that GitHub gists do not support empty files, so we need to
  # conditionally include the diff file.
  diff_entry = {"E_git_diff.diff": {"content": _git_diff}} if _git_diff != "" else {}
  create_gist_resp = _create_gist(
      gist_name, {
          f"A_{gist_name}_metadata.md": {
              "content": metadata
          },
          f"C_{_current_script_path.name}": {
              "content": _current_script_contents
          },
          "D_console_output.log": {
              "content": console_output
          },
          **diff_entry
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
    _add_gist_files(gist_id, {"F_remembered_stuff.pkl": stuff_file.name, **figs_files})

  # Updating a gist does not alter its html url.
  print(f"[blt] results dumped: {gist_url}")
  plt.show()
