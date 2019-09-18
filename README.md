# research

ALL THE CODEZ

## Mujoco/Ubuntu setup

1. Download mujoco at https://www.roboti.us/index.html, unzip and put in `~/.mujoco/mujoco200`.
2. Put the license key at `~/.mujoco/mjkey.txt`.
3. Install dependencies

```bash
# Fixes `fatal error: GL/osmesa.h: No such file or directory`
sudo apt install libosmesa6-dev
# Fixes `/usr/bin/ld: cannot find -lGL`
sudo apt install libglew-dev
```

4. Install clang and set it as the default `cc` alternative.

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
