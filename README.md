# research

ALL THE CODEZ

## Mujoco/Ubuntu setup

1. Download mujoco at https://www.roboti.us/index.html, unzip and put in `~/.mujoco/mujoco200`.
2. Add

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin
```

to `~/.profile`.

3. Put the license key at `~/.mujoco/mjkey.txt`.
4. Install dependencies

```bash
# Fixes `fatal error: GL/osmesa.h: No such file or directory`
sudo apt install libosmesa6-dev
# Fixes `/usr/bin/ld: cannot find -lGL`
sudo apt install libglew-dev
```

5. Install clang and set it as the default `cc` alternative.

```bash
sudo apt install clang
sudo update-alternatives --config cc
brew uninstall gcc
```

Logging in/out to fix `$PATH` may also be necessary.

See

- https://github.com/openai/mujoco-py/issues/455
- https://github.com/openai/mujoco-py/issues/394
- https://github.com/ethz-asl/reinmav-gym/issues/35

## CUDA/cuDNN setup

The `nvidia-driver-430` and `nvidia-cuda-toolkit` on Ubuntu 18.04 install CUDA 9.1 which is not supported by JAX at the moment.

1. Remove any current installation.

```bash
sudo apt-get purge *cuda*
sudo apt-get purge *nvidia*
sudo apt-get purge *cudnn*
```

and then follow the runfile uninstall steps (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-uninstallation).

2. Make sure that gcc is current cc alternative:

```bash
sudo update-alternatives --config cc
cc --version
```

(This was necessary for CUDA 10.1. May not be necessary for 10.0.)

3. Follow the installation instructions [here](https://developer.nvidia.com/cuda-downloads) for the "runfile (local)" version. Install version 10.0 since TF and pytorch do not yet support 10.1.

4. Add

```bash
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

to `~/.profile`.

5. Download the "cuDNN Library for Linux" (https://developer.nvidia.com/rdp/cudnn-download), not the deb version. You'll need to be logged in order for the downloads to work. Using wget/curl isn't sufficient. Easiest to download them locally and then scp them to the remote machine.

6. Install cuDNN (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar) but note that the CUDA installation directory is `/usr/local/cuda-10.0` not `/usr/local/cuda`.

7. Reboot.

8. Follow the pip instructions here (https://github.com/google/jax#pip-installation) in a `pipenv shell` to install the new GPU versions of `jax`/`jaxlib`.

See

- https://developer.nvidia.com/cuda-zone
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#faq2
- https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible
- https://discuss.pytorch.org/t/when-pytorch-supports-cuda-10-1/38852

Note that the deb installation does not seem to support multiple CUDA installations living in harmony. This may become problematic as some packages like pytorch do not yet support CUDA 10.1.

With CUDA 10.0, JAX may require the `xla_gpu_cuda_data_dir` XLA flag to be set as well:

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.0/
```

## Set hostname

On AWS Ubuntu 18.04,

```bash
user$ sudo su
root$ hostnamectl set-hostname <whatever>
```

## Expand EBS volume

No downtime is necessary.

1. Change the volume in the console.
2. Then follow https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html. Use `df -T` to get the filesystem type.
