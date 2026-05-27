#!/bin/bash
# ============================================================
# SNI-SLAM Colab setup — one-shot env + data + weights bootstrap.
#
# Usage (from a fresh Colab T4 tunnel terminal):
#   bash sni-slam/scripts/colab_setup_sni.sh
#
# What it does, in order:
#   1. GPU verify
#   2. apt: libopenexr-dev (required by openexr pip)
#   3. miniconda install (if absent)
#   4. Try to restore conda env from MyDrive/sni_cache/sni_env.tar.gz
#      → fall through to building from environment.yaml if cache is missing
#   5. Verify imports inside the env
#   6. Try to restore data+weights from MyDrive/sni_cache/sni_data/
#      → fall through to gdown --folder if cache is missing
#   7. Patch the hardcoded /data0/* config paths if a fresh repo clone
#      did not pick up our local edits
#   8. Optionally tar+save the freshly-built env back to Drive for next session
#
# Time budget:
#   First run (cold): ~25-40 min env build + ~10-20 min data download
#   Subsequent runs (cache hit): ~3-5 min env restore + ~3-5 min data restore
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE_CACHE=/content/drive/MyDrive/sni_cache
DRIVE_ENV_TAR=$DRIVE_CACHE/sni_env.tar.gz
DRIVE_DATA_DIR=$DRIVE_CACHE/sni_data
GDRIVE_FOLDER_ID="1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h"   # README link
CONDA_PREFIX_PATH=/opt/conda
ENV_NAME=sni
ENV_PATH=$CONDA_PREFIX_PATH/envs/$ENV_NAME

mkdir -p "$DRIVE_CACHE"

# ---------------------------------------------------------------------
# 1. GPU verify
# ---------------------------------------------------------------------
echo "[1/8] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: no GPU detected. Select a T4 GPU runtime in Colab and re-run."
  exit 1
}

# ---------------------------------------------------------------------
# 2. apt deps
# ---------------------------------------------------------------------
echo ""
echo "[2/8] apt: libopenexr-dev..."
sudo apt-get update -qq
sudo apt-get install -y -qq libopenexr-dev

# ---------------------------------------------------------------------
# 3. miniconda install (if needed)
# ---------------------------------------------------------------------
echo ""
echo "[3/8] conda check..."
if [ ! -x "$CONDA_PREFIX_PATH/bin/conda" ]; then
  echo "  installing miniconda to $CONDA_PREFIX_PATH..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $CONDA_PREFIX_PATH
  rm -f /tmp/miniconda.sh
fi
# shellcheck disable=SC1091
source $CONDA_PREFIX_PATH/etc/profile.d/conda.sh
conda --version

# ---------------------------------------------------------------------
# 4. Restore or build the 'sni' env
# ---------------------------------------------------------------------
echo ""
echo "[4/8] conda env 'sni'..."
if [ -d "$ENV_PATH" ]; then
  echo "  env already present at $ENV_PATH — skipping build/restore"
elif [ -f "$DRIVE_ENV_TAR" ]; then
  echo "  restoring from cache: $DRIVE_ENV_TAR"
  mkdir -p "$ENV_PATH"
  tar -xzf "$DRIVE_ENV_TAR" -C "$CONDA_PREFIX_PATH/envs"
  echo "  cache restored ($(du -sh "$ENV_PATH" | cut -f1))"
else
  echo "  no cache — building from environment.yaml (~25-35 min)..."
  conda env create -f "$REPO_ROOT/environment.yaml"
  echo "  built. Will tar+upload to Drive at step 8."
  ENV_FRESHLY_BUILT=1
fi

# ---------------------------------------------------------------------
# 5. Import sanity-check inside the env
# ---------------------------------------------------------------------
echo ""
echo "[5/8] import sanity check..."
conda activate $ENV_NAME
python - <<'PY'
import torch, cv2
print(f"  torch       {torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  cv2         {cv2.__version__}")
try:
    import pytorch3d
    print(f"  pytorch3d   {pytorch3d.__version__}")
except Exception as e:
    print(f"  pytorch3d   FAIL  {e}")
try:
    import open3d
    print(f"  open3d      {open3d.__version__}")
except Exception as e:
    print(f"  open3d      FAIL  {e}")
try:
    import OpenEXR
    print(f"  OpenEXR     OK")
except Exception as e:
    print(f"  OpenEXR     FAIL  {e}")
try:
    import timm
    print(f"  timm        {timm.__version__}")
except Exception as e:
    print(f"  timm        FAIL  {e}")
PY

# ---------------------------------------------------------------------
# 6. Data + weights — restore or download
# ---------------------------------------------------------------------
echo ""
echo "[6/8] data + weights..."
DATA_TARGET="$REPO_ROOT/data/replica"
SEG_TARGET="$REPO_ROOT/seg"
mkdir -p "$DATA_TARGET" "$SEG_TARGET"

if [ -d "$DRIVE_DATA_DIR/replica" ] && [ -f "$DRIVE_DATA_DIR/seg/dinov2_replica.pth" ]; then
  echo "  restoring from Drive cache: $DRIVE_DATA_DIR"
  cp -r "$DRIVE_DATA_DIR/replica/"* "$DATA_TARGET/" 2>/dev/null || true
  cp -r "$DRIVE_DATA_DIR/seg/"*     "$SEG_TARGET/"  2>/dev/null || true
else
  echo "  downloading from Google Drive (folder $GDRIVE_FOLDER_ID)..."
  pip install -q gdown
  STAGING=/content/sni_download
  mkdir -p "$STAGING"
  gdown --folder "$GDRIVE_FOLDER_ID" -O "$STAGING" --remaining-ok || {
    echo "  ERROR: gdown failed. The Drive folder may have a download quota or "
    echo "         require manual login. Open it in a browser and copy files into:"
    echo "         $DRIVE_DATA_DIR"
    exit 1
  }
  echo "  inventory of $STAGING:"
  ls -la "$STAGING"
  # Try to auto-organise: the Google Drive layout typically has:
  #   replica/<scene>/{rgb,depth,semantic_class,traj.txt}
  #   seg/dinov2_replica.pth
  #   seg/facebookresearch_dinov2_main.zip
  if [ -d "$STAGING/replica" ]; then
    cp -rv "$STAGING/replica/"* "$DATA_TARGET/" || true
  fi
  if [ -d "$STAGING/seg" ]; then
    cp -rv "$STAGING/seg/"* "$SEG_TARGET/" || true
  fi
  # Unzip DINOv2 backbone if zipped
  if [ -f "$SEG_TARGET/facebookresearch_dinov2_main.zip" ] && \
     [ ! -d "$SEG_TARGET/facebookresearch_dinov2_main" ]; then
    echo "  unzipping facebookresearch_dinov2_main.zip..."
    (cd "$SEG_TARGET" && unzip -q facebookresearch_dinov2_main.zip)
  fi
  # Mirror to Drive cache for next session
  echo "  caching downloaded data to $DRIVE_DATA_DIR..."
  mkdir -p "$DRIVE_DATA_DIR/replica" "$DRIVE_DATA_DIR/seg"
  cp -r "$DATA_TARGET/"* "$DRIVE_DATA_DIR/replica/" 2>/dev/null || true
  cp -r "$SEG_TARGET/"*  "$DRIVE_DATA_DIR/seg/"     2>/dev/null || true
fi

echo "  expected data layout (sample for room_0):"
ls -d "$DATA_TARGET"/*/                                              2>/dev/null | head -8 || true
[ -f "$SEG_TARGET/dinov2_replica.pth" ]                               && echo "  seg/dinov2_replica.pth                  ✓" || echo "  seg/dinov2_replica.pth                  MISSING"
[ -d "$SEG_TARGET/facebookresearch_dinov2_main" ]                     && echo "  seg/facebookresearch_dinov2_main/       ✓" || echo "  seg/facebookresearch_dinov2_main/       MISSING"

# ---------------------------------------------------------------------
# 7. Idempotent config patches (in case the repo was freshly cloned and
#    our local edits aren't present)
# ---------------------------------------------------------------------
echo ""
echo "[7/8] config patches..."
SNI_YAML="$REPO_ROOT/configs/SNI-SLAM.yaml"
if grep -q "/data0/nerf/sni-slam/seg/dinov2_replica.pth" "$SNI_YAML"; then
  echo "  patching $SNI_YAML: seg paths -> relative"
  sed -i "s|/data0/nerf/sni-slam/seg/dinov2_replica.pth|seg/dinov2_replica.pth|g" "$SNI_YAML"
  sed -i "s|/data0/nerf/sni-slam/seg/|seg/|g"                                    "$SNI_YAML"
else
  echo "  $SNI_YAML already relative"
fi
OFFICE4_YAML="$REPO_ROOT/configs/Replica/office4.yaml"
if grep -q "/data0/replica/office_4" "$OFFICE4_YAML"; then
  echo "  patching $OFFICE4_YAML: input_folder -> relative"
  sed -i "s|/data0/replica/office_4|data/replica/office_4|g" "$OFFICE4_YAML"
else
  echo "  $OFFICE4_YAML already relative"
fi

# ---------------------------------------------------------------------
# 8. Cache the env tar back to Drive (only if we built fresh)
# ---------------------------------------------------------------------
echo ""
echo "[8/8] env cache..."
if [ "${ENV_FRESHLY_BUILT:-0}" = "1" ]; then
  echo "  tarring $ENV_PATH -> $DRIVE_ENV_TAR (~5-10 min, ~3-4 GB)"
  (cd "$CONDA_PREFIX_PATH/envs" && tar -czf "$DRIVE_ENV_TAR" "$ENV_NAME")
  echo "  cached. Next session will restore instead of rebuild."
else
  echo "  env not freshly built — leaving Drive cache as-is"
fi

echo ""
echo "============================================================"
echo "Setup complete. Activate the env then test with:"
echo "  conda activate $ENV_NAME"
echo "  cd $REPO_ROOT"
echo "  python -W ignore run.py configs/Replica/room0.yaml"
echo "============================================================"
