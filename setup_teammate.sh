#!/usr/bin/env bash
# Run on the TEAMMATE'S Linux laptop, from inside the cloned bundle repo:
#   git clone git@github.com:<you>/hrl-so101-bundle.git && cd hrl-so101-bundle
#   bash setup_teammate.sh
# Recreates the 'lerobot' conda env at the same versions, installs LeRobot at the same
# commit (+ local patch), copies calibration, and sets serial-port permissions.

set -eu
B="$(cd "$(dirname "$0")" && pwd)"
ENV=lerobot
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/Documents/so101/lerobot}"
CALIB_ROOT="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}"

command -v conda >/dev/null || { echo "!! Install Miniforge first: https://github.com/conda-forge/miniforge"; exit 1; }
eval "$(conda shell.bash hook)"
PY=$(cat "$B/env/python_version.txt")
FF=$(cat "$B/env/ffmpeg_version.txt")
TORCH_VER=$(sed -n 1p "$B/env/torch_info.txt")

echo "== 1. Conda env '$ENV' (python $PY, ffmpeg $FF)"
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "   env '$ENV' already exists - reusing it (remove with: conda env remove -n $ENV)"
else
  conda create -y -n "$ENV" "python=$PY"
fi
conda activate "$ENV"
if [ -n "$FF" ]; then conda install -y -c conda-forge "ffmpeg=$FF"; else conda install -y -c conda-forge ffmpeg; fi

echo "== 2. PyTorch $TORCH_VER"
if [[ "$TORCH_VER" == *+cu* ]]; then
  CU=${TORCH_VER##*+}                                  # e.g. cu128
  TORCH_LINES=$(grep -E '^(torch|torchvision|torchaudio)==' "$B/env/requirements-lock.txt" || true)
  [ -n "$TORCH_LINES" ] && pip install $TORCH_LINES --index-url "https://download.pytorch.org/whl/$CU"
fi

echo "== 3. Pinned packages"
grep -vE '^(torch|torchvision|torchaudio)==|^lerobot([=@ ]|$)' "$B/env/requirements-lock.txt" > /tmp/req_no_torch.txt
pip install -r /tmp/req_no_torch.txt

echo "== 4. LeRobot at the same commit"
if [ ! -d "$LEROBOT_DIR/.git" ]; then
  mkdir -p "$(dirname "$LEROBOT_DIR")"
  git clone "$(cat "$B/lerobot/REMOTE")" "$LEROBOT_DIR"
fi
cd "$LEROBOT_DIR"
git fetch -q --all
git checkout -q "$(cat "$B/lerobot/COMMIT")"
if [ -s "$B/lerobot/local_changes.patch" ]; then
  git apply --check "$B/lerobot/local_changes.patch" && git apply "$B/lerobot/local_changes.patch" && echo "   applied local_changes.patch"
fi
pip install -e ".[feetech]" --no-deps
cd "$B"

echo "== 5. Calibration -> $CALIB_ROOT"
mkdir -p "$CALIB_ROOT"
cp -r "$B/calibration/." "$CALIB_ROOT/"
find "$CALIB_ROOT" -name '*vasco*.json' | sed 's/^/   /'

echo "== 6. Serial port permissions"
sudo usermod -aG dialout "$USER" || true
if dpkg -l brltty >/dev/null 2>&1; then sudo apt-get remove -y brltty || true; fi

echo "== 7. Checks"
python - <<'PY'
import torch, lerobot
print("   torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("   lerobot imported from", lerobot.__file__)
PY
command -v lerobot-record >/dev/null && echo "   lerobot-record OK"
command -v hf >/dev/null && echo "   hf CLI OK - run: hf auth login"
echo
echo "Done. LOG OUT AND BACK IN (dialout group), then follow README.md 'First run'."
