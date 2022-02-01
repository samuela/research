# Example usage:
#     python parallel_mnist_videos.py wandb/run-20220201_050226-2vzg9n1u/files/media/images/

import re
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

path = sys.argv[1]

def ffmpegify(prefix):
  with tempfile.TemporaryDirectory() as tempdir:
    for file in glob(path + f"/{prefix}*.png"):
      n = int(re.match(r"^" + prefix + r"_(\d+)_.*$", Path(file).stem).group(1))
      shutil.copy(file, Path(tempdir) / f"{prefix}_{n:05d}.png")

    subprocess.run([
        "ffmpeg", "-r", "10", "-i",
        Path(tempdir) / f"{prefix}_%05d.png", "-vcodec", "libx264", "-crf", "15", "-pix_fmt",
        "yuv420p", "-y", f"{prefix}.mp4"
    ],
                   check=True)

ffmpegify("interp_loss_plot")
ffmpegify("interp_acc_plot")
