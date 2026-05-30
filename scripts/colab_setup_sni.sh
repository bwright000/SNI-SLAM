#!/bin/bash
# ============================================================
# colab_setup_sni.sh — SINGLE self-contained SNI-SLAM Colab bootstrap.
#
# Reproduces the SNI-SLAM (CVPR 2024) Replica results on a Colab T4 by
# building the authors' EXACT environment and staging the authors' EXACT
# data + segmentation weights. Scope: room0 + room1 (the only two Replica
# scenes the authors distribute). This is a SETUP-ONLY script — it leaves
# the env ready and prints the run + eval commands to fire yourself.
#
# Run it from a fresh Colab tunnel terminal in ONE of two ways:
#
#   A) one-liner, from a bare tunnel (script self-clones the repo):
#      curl -fsSL https://raw.githubusercontent.com/bwright000/sni-slam/main/scripts/colab_setup_sni.sh | bash
#
#   B) if you already cloned the repo:
#      bash sni-slam/scripts/colab_setup_sni.sh
#
# What it does, in order:
#   0. Ensure the repo is checked out (clone if run standalone)
#   1. GPU verify
#   2. apt: libopenexr-dev (required by the openexr pip wheel)
#   3. miniconda install (if absent)
#   4. Restore conda env from Drive cache, else build from environment.yaml
#      (Py 3.7.11 / PyTorch 1.11.0+cu113 / pytorch3d 0.7.1 — the paper stack)
#   5. Verify imports inside the env
#   6. Stage data + weights (Drive cache → gdown → My-Drive quota fallback)
#   7. Patch any hardcoded /data0/* config paths to repo-relative
#   8. Cache the freshly-built env back to Drive for next session
#
# Google Drive is OPTIONAL. If /content/drive/MyDrive is mounted it is used
# for caching (faster re-runs) and as a quota fallback for the big files.
# If it is not mounted the script still works — it just can't cache or use
# the My-Drive copy workaround.
#
# Time budget (T4):
#   Cold:  ~25-40 min env build + ~10-20 min data download
#   Warm:  ~3-5 min env restore + ~3-5 min data restore (needs Drive cache)
# ============================================================
set -eo pipefail

# ---------------------------------------------------------------------
# 0. Locate or clone the repo
# ---------------------------------------------------------------------
echo "[0/8] locating repo..."
REPO_URL=${SNI_REPO_URL:-https://github.com/bwright000/sni-slam.git}
REPO_BRANCH=${SNI_REPO_BRANCH:-main}

SELF="${BASH_SOURCE[0]:-}"
CAND=""
if [ -n "$SELF" ] && [ -f "$SELF" ]; then
  CAND="$(cd "$(dirname "$SELF")/.." && pwd)"
fi
if [ -n "$CAND" ] && [ -f "$CAND/run.py" ]; then
  REPO_ROOT="$CAND"                                   # run from inside checkout
elif [ -f "./run.py" ] && [ -f "./environment.yaml" ]; then
  REPO_ROOT="$(pwd)"                                  # CWD is the checkout
else
  REPO_ROOT=${SNI_REPO_DIR:-/content/sni-slam}        # standalone → clone
  if [ ! -f "$REPO_ROOT/run.py" ]; then
    echo "  cloning $REPO_URL ($REPO_BRANCH) -> $REPO_ROOT"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_ROOT"
  fi
fi
cd "$REPO_ROOT"
echo "  repo root: $REPO_ROOT"

DRIVE_ROOT=/content/drive/MyDrive
DRIVE_CACHE=$DRIVE_ROOT/sni_cache
DRIVE_ENV_TAR=$DRIVE_CACHE/sni_env.tar.gz
DRIVE_DATA_DIR=$DRIVE_CACHE/sni_data
GDRIVE_FOLDER_ID="1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h"   # authors' README link
CONDA_PREFIX_PATH=/opt/conda
ENV_NAME=sni
ENV_PATH=$CONDA_PREFIX_PATH/envs/$ENV_NAME

if [ -d "$DRIVE_ROOT" ]; then
  DRIVE_OK=1
  mkdir -p "$DRIVE_CACHE"
  echo "  Google Drive mounted — caching + quota fallback enabled"
else
  DRIVE_OK=0
  echo "  Google Drive NOT mounted at $DRIVE_ROOT — no caching / no quota fallback."
  echo "    (To enable: run a Colab notebook cell with"
  echo "       from google.colab import drive; drive.mount('/content/drive')"
  echo "     then re-run this script.)"
fi

# ---------------------------------------------------------------------
# 1. GPU verify
# ---------------------------------------------------------------------
echo ""
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
sudo apt-get install -y -qq libopenexr-dev unzip

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

# Accept Anaconda channel Terms of Service. environment.yaml uses 'defaults'
# (repo.anaconda.com/pkgs/{main,r}); recent conda refuses non-interactive
# installs from these until ToS is accepted. Idempotent; '|| true' so older
# conda without the 'tos' subcommand doesn't abort.
echo "  accepting Anaconda channel ToS (pkgs/main, pkgs/r)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# ---------------------------------------------------------------------
# 4. Restore or build the 'sni' env
# ---------------------------------------------------------------------
echo ""
echo "[4/8] conda env 'sni'..."
if [ -d "$ENV_PATH" ]; then
  echo "  env already present at $ENV_PATH — skipping build/restore"
elif [ "$DRIVE_OK" = "1" ] && [ -f "$DRIVE_ENV_TAR" ]; then
  echo "  restoring from cache: $DRIVE_ENV_TAR"
  mkdir -p "$ENV_PATH"
  tar -xzf "$DRIVE_ENV_TAR" -C "$CONDA_PREFIX_PATH/envs"
  echo "  cache restored ($(du -sh "$ENV_PATH" | cut -f1))"
else
  echo "  no cache — building from environment.yaml (~25-35 min)..."
  conda env create -f "$REPO_ROOT/environment.yaml"
  echo "  built. Will tar+upload to Drive at step 8 (if Drive mounted)."
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
for mod in ("pytorch3d", "open3d", "OpenEXR", "timm"):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "OK")
        print(f"  {mod:<11} {v}")
    except Exception as e:
        print(f"  {mod:<11} FAIL  {e}")
PY

# ---------------------------------------------------------------------
# 6. Data + weights — restore or download (room0 + room1 only)
# ---------------------------------------------------------------------
echo ""
echo "[6/8] data + weights..."
DATA_TARGET="$REPO_ROOT/data/replica"
SEG_TARGET="$REPO_ROOT/seg"
mkdir -p "$DATA_TARGET" "$SEG_TARGET"

if [ "$DRIVE_OK" = "1" ] && [ -d "$DRIVE_DATA_DIR/replica" ] && \
   [ -f "$DRIVE_DATA_DIR/seg/dinov2_replica.pth" ]; then
  echo "  restoring from Drive cache: $DRIVE_DATA_DIR"
  cp -r "$DRIVE_DATA_DIR/replica/"* "$DATA_TARGET/" 2>/dev/null || true
  cp -r "$DRIVE_DATA_DIR/seg/"*     "$SEG_TARGET/"  2>/dev/null || true
else
  # `gdown --folder` is unreliable: it aborts the whole transfer if any single
  # file hits Google's per-file download quota ("had many accesses"), which the
  # popular dinov2_replica.pth / room_*.zip do. Fetch each file by its own ID
  # instead — direct file mode is less rate-limited than folder mode.
  # File IDs captured 2026-05-20 from folder $GDRIVE_FOLDER_ID. The authors'
  # distributed subset is room_0 + room_1 only.
  echo "  downloading individual files by ID..."
  pip install -q gdown
  STAGING=/content/sni_download
  mkdir -p "$STAGING/seg"

  # seg/ artifacts
  gdown 1ffPsH3NWK8GqoeSXDTGWo-QfC-LbhYvU -O "$STAGING/seg/dinov2_replica.pth"               || echo "  WARN dinov2_replica.pth failed (quota?) — see fallback below"
  gdown 12QnXVswA1yiet-zmhzkLNsjCpboOLQBS -O "$STAGING/seg/facebookresearch_dinov2_main.zip" || true
  gdown 1A-pm0LWQZzevNdMZ14EP0z_b83t2-JuF -O "$STAGING/seg/num_semantic_class.pkl"           || true
  gdown 1gRbgMobHhZCa63dcOSGpjKK7UUVK6vcW -O "$STAGING/seg/semantic_classes.pkl"             || true

  # Replica scene zips (the distributed subset)
  gdown 1cDpqqb49V_SpU4seSTHc2CQegUAi1Af7 -O "$STAGING/room_0.zip" || true
  gdown 16WI3WNtP0-dLw8maHOYwiZhgYQL5eNGw -O "$STAGING/room_1.zip" || true

  # Per-file daily quota blocks gdown on the large/popular files with
  # "Cannot retrieve the public link... had many accesses". gdown cannot
  # bypass it. Fallback (Drive only): look on the user's own My Drive for a
  # manual copy (Make a copy -> fresh ID -> no quota) and use that instead.
  if [ "$DRIVE_OK" = "1" ]; then
    declare -A QUOTA_FALLBACK=(
      ["$STAGING/seg/dinov2_replica.pth"]="*dinov2_replica*.pth"
      ["$STAGING/room_0.zip"]="*room_0*.zip"
      ["$STAGING/room_1.zip"]="*room_1*.zip"
    )
    for dest in "${!QUOTA_FALLBACK[@]}"; do
      if [ ! -s "$dest" ]; then
        pat="${QUOTA_FALLBACK[$dest]}"
        hit=$(find "$DRIVE_ROOT" -maxdepth 3 -iname "$pat" 2>/dev/null | head -1)
        if [ -n "$hit" ]; then
          echo "  quota fallback: $(basename "$dest") <- $hit"
          cp "$hit" "$dest"
        else
          echo "  MISSING $(basename "$dest") — gdown quota-blocked, no My-Drive copy found."
          echo "    Open it in a browser, 'Make a copy' into My Drive, then re-run."
        fi
      fi
    done
  fi

  echo "  inventory of $STAGING:"
  ls -la "$STAGING" "$STAGING/seg"

  # Organise seg artifacts
  cp -v "$STAGING/seg/"* "$SEG_TARGET/" 2>/dev/null || true
  if [ -f "$SEG_TARGET/facebookresearch_dinov2_main.zip" ] && \
     [ ! -d "$SEG_TARGET/facebookresearch_dinov2_main" ]; then
    echo "  unzipping facebookresearch_dinov2_main.zip..."
    (cd "$SEG_TARGET" && unzip -q facebookresearch_dinov2_main.zip)
  fi

  # Unzip Replica scene zips into data/replica/. Internal folder is
  # 'room_0' / 'room_1' (matching the config's data/replica/room_0 path).
  for z in "$STAGING"/room_*.zip; do
    [ -f "$z" ] || continue
    echo "  unzipping $(basename "$z") into $DATA_TARGET ..."
    unzip -q -o "$z" -d "$DATA_TARGET"
  done

  # Mirror to Drive cache for next session
  if [ "$DRIVE_OK" = "1" ]; then
    echo "  caching downloaded data to $DRIVE_DATA_DIR..."
    mkdir -p "$DRIVE_DATA_DIR/replica" "$DRIVE_DATA_DIR/seg"
    cp -r "$DATA_TARGET/"* "$DRIVE_DATA_DIR/replica/" 2>/dev/null || true
    cp -r "$SEG_TARGET/"*  "$DRIVE_DATA_DIR/seg/"     2>/dev/null || true
  fi
fi

echo "  data layout:"
ls -d "$DATA_TARGET"/*/                            2>/dev/null | head -8 || true
[ -f "$SEG_TARGET/dinov2_replica.pth" ]            && echo "  seg/dinov2_replica.pth             ✓" || echo "  seg/dinov2_replica.pth             MISSING"
[ -d "$SEG_TARGET/facebookresearch_dinov2_main" ]  && echo "  seg/facebookresearch_dinov2_main/  ✓" || echo "  seg/facebookresearch_dinov2_main/  MISSING"
[ -f "$SEG_TARGET/semantic_classes.pkl" ]          && echo "  seg/semantic_classes.pkl           ✓" || echo "  seg/semantic_classes.pkl           MISSING"
[ -f "$SEG_TARGET/num_semantic_class.pkl" ]        && echo "  seg/num_semantic_class.pkl         ✓" || echo "  seg/num_semantic_class.pkl         MISSING"

# ---------------------------------------------------------------------
# 7. Idempotent config patches (repo was freshly cloned → fix /data0/* paths)
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
if [ -f "$OFFICE4_YAML" ] && grep -q "/data0/replica/office_4" "$OFFICE4_YAML"; then
  echo "  patching $OFFICE4_YAML: input_folder -> relative"
  sed -i "s|/data0/replica/office_4|data/replica/office_4|g" "$OFFICE4_YAML"
fi

# ---------------------------------------------------------------------
# 8. Cache the env tar back to Drive (only if freshly built and Drive mounted)
# ---------------------------------------------------------------------
echo ""
echo "[8/8] env cache..."
if [ "${ENV_FRESHLY_BUILT:-0}" = "1" ] && [ "$DRIVE_OK" = "1" ]; then
  echo "  tarring $ENV_PATH -> $DRIVE_ENV_TAR (~5-10 min, ~3-4 GB)"
  (cd "$CONDA_PREFIX_PATH/envs" && tar -czf "$DRIVE_ENV_TAR" "$ENV_NAME")
  echo "  cached. Next session will restore instead of rebuild."
else
  echo "  nothing to cache (env not freshly built, or Drive not mounted)."
fi

# ---------------------------------------------------------------------
# Done — setup-only. Print the run + eval commands.
# ---------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup complete (env + data + weights staged for room0 + room1)."
echo ""
echo "Run SLAM (each ~1.5-2.5 h on a T4):"
echo "  conda activate $ENV_NAME"
echo "  cd $REPO_ROOT"
echo "  python -W ignore run.py configs/Replica/room0.yaml"
echo "  python -W ignore run.py configs/Replica/room1.yaml"
echo ""
echo "Then evaluate against the paper baselines (PAPER_BASELINES.md):"
echo "  python src/tools/eval_ate.py   configs/Replica/room0.yaml"
echo "  python src/tools/eval_recon.py configs/Replica/room0.yaml"
echo ""
echo "Or fire both scenes + eval in one go with the resumable runner:"
echo "  bash scripts/run_replica_subset.sh"
echo "============================================================"
