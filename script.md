# Bot Colosseo dependency commands

The `botcolosseo` Conda environment already contains the M0 runtime and
training dependencies. Do not recreate the environment or reinstall PyTorch.

## Verified M0 environment status

Verified on 2026-07-20:

- `python -m pip check`: `No broken requirements found.`
- `python -m ruff --version`: `ruff 0.15.22`

No additional M0 Python packages are pending. The commands below are retained
for reproducing the verification after future dependency changes:

```bash
conda activate botcolosseo
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip check
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m ruff --version
```

## Verified: later-stage Python packages

Verified on 2026-07-20: `psutil`, `pytest-timeout`, `pytest-cov`, `scipy`, and
`seaborn` are installed. They support multiplayer process management,
experiment analysis, visualization, and test hardening. The installation
commands are retained for environment reproduction:

```bash
conda activate botcolosseo
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip install \
  'psutil>=5.9,<8.0' \
  'pytest-timeout>=2.3,<3.0' \
  'pytest-cov>=5.0,<8.0' \
  'scipy>=1.11,<2.0' \
  'seaborn>=0.13,<1.0'
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip check
```

Verify:

```bash
python -c "import psutil, scipy, seaborn; print(psutil.__version__, scipy.__version__, seaborn.__version__)"
python -m pytest --help | grep -E 'timeout|cov'
```

## Verified: ACC 1.60 ACS compiler

ACC compiles the ACS scenario scripts used by the UDMF Crystal Run map. The
binary was built and verified on 2026-07-20 at
`/home/wencong/.local/bin/acc`; it reports version 1.60. The machine already
has Git, CMake, Make, GCC, and G++. The build commands are retained for
reproduction:

```bash
mkdir -p /home/wencong/.local/src /home/wencong/.local/bin
git clone --branch 1.60 --depth 1 \
  https://github.com/ZDoom/acc.git \
  /home/wencong/.local/src/acc-1.60
cmake \
  -S /home/wencong/.local/src/acc-1.60 \
  -B /home/wencong/.local/src/acc-1.60/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/wencong/.local/src/acc-1.60/build --parallel 4
install -m 0755 \
  /home/wencong/.local/src/acc-1.60/build/acc \
  /home/wencong/.local/bin/acc
/home/wencong/.local/bin/acc
```

Expected: the final command prints the ACC usage text. Add
`/home/wencong/.local/bin` to the interactive shell PATH if desired; project
build scripts should accept an explicit `ACC_PATH` and must not assume a global
installation.

## Verified: SLADE 3.2.12 map editor

SLADE is useful for visually editing and inspecting UDMF maps. Ubuntu 20.04
uses the official DRD Team package repository for older Debian/Ubuntu releases.
Version 3.2.12 is installed. The installation commands are retained for
reproduction:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
sudo wget -4\
  https://debian.drdteam.org/drdteam.gpg \
  -O /etc/apt/keyrings/drdteam.gpg
echo 'deb [signed-by=/etc/apt/keyrings/drdteam.gpg] https://debian.drdteam.org/ stable multiverse' \
  | sudo tee /etc/apt/sources.list.d/drdteam.list
sudo apt-get update
sudo apt-get install slade
dpkg-query -W -f='${Version}\n' slade
```

Expected: `3.2.12`.

SLADE is a GUI application, and even `slade --version` initializes GTK. In a
headless SSH shell with no `DISPLAY`, that command fails with `Unable to
initialize GTK+`; this does not mean the package installation failed. Use
`dpkg-query` for headless version checks. Interactive map authoring requires
X11 forwarding, a remote desktop, or local editing. ACC is sufficient for
command-line ACS builds, and SLADE is not required for M0.

## Already installed: do not reinstall

- Python 3.10.20, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124
- ViZDoom 1.3.0 and bundled `freedoom2.wad`/`vizdoom.pk3`
- Gymnasium, NumPy, Pandas, PyYAML, Matplotlib, OpenCV, TensorBoard, tqdm
- ImageIO, imageio-ffmpeg, system FFmpeg/ffprobe
- Ruff 0.15.22 and pytest 9.1.1
- CMake, Make, GCC/G++, Git, Git LFS, tmux, screen, Xvfb, zip/unzip, jq, rsync

## Deliberately excluded until a concrete requirement appears

- Weights & Biases: TensorBoard is sufficient for the approved experiment plan.
- ONNX and ONNX Runtime: model export is not an approved milestone requirement.
- Hugging Face Hub: checkpoints can use GitHub Releases or another later-chosen
  artifact host.
- `omgifol`: the M1 plan will define the WAD packaging boundary; do not add a
  second map representation before that decision.
- GZDoom and original Doom assets: ViZDoom plus Freedoom is the approved runtime
  and licensing path.
