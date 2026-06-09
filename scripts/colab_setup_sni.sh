#!/bin/bash
# ============================================================
# colab_setup_sni.sh — SINGLE self-contained SNI-SLAM Colab bootstrap.
#
# Reproduces the SNI-SLAM (CVPR 2024) Replica results on a Colab T4 by
# building the authors' EXACT environment and staging ALL 8 Replica scenes
# (room0/1/2 + office0/1/2/3/4) in a label-compatible way, so you can claim
# "SNI-SLAM works as the authors intended". This is a SETUP-ONLY script — it
# leaves the env + data ready and prints the run + eval commands to fire.
#
# Data provenance (researched):
#   - The authors only distribute room_0 + room_1 on their Drive. Their
#     data_generation/ pipeline renders from vMAP's Replica source meshes +
#     vMAP trajectories (replica_render_config_vMAP.yaml -> /data0/vmap/...).
#   - vMAP (Kong et al., CVPR 2023) publicly distributes all 8 rendered
#     scenes. vMAP's per-scene layout is rgb/rgb_N.png, depth/depth_N.png,
#     semantic_class/semantic_class_N.png, traj_w_c.txt — a 1:1 match for
#     SNI-SLAM's loader except the trajectory file is named traj_w_c.txt
#     (both are camera-to-world), so we just rename it to traj.txt.
#   - The semantic class IDs come from the same Replica info_semantic.json
#     mapping the authors used, so the authors' pretrained segmentation head
#     (seg/dinov2_replica.pth) + semantic_classes.pkl stay valid. We VALIDATE
#     this on all 8 scenes before declaring success.
#
#   => All 8 scenes from vMAP vmap.zip + the authors' seg/ weights = a
#      faithful, label-consistent 8-scene reproduction.
#
# Run from a fresh Colab tunnel terminal, EITHER:
#   A) one-liner from a bare tunnel (script self-clones the repo):
#      curl -fsSL https://raw.githubusercontent.com/bwright000/sni-slam/main/scripts/colab_setup_sni.sh | bash
#   B) if you already cloned the repo:
#      bash sni-slam/scripts/colab_setup_sni.sh
#
# Google Drive is OPTIONAL but RECOMMENDED: if /content/drive/MyDrive is
# mounted it caches the (heavy) extracted data + the conda env so re-runs
# after a session death are fast, and acts as a quota fallback for the seg
# weights. Without it the script still works but re-downloads each session.
#
# Disk + time budget (T4):
#   vmap.zip is 44.8 GB (single file — no per-scene option). Peak disk during
#   extraction ~55-65 GB; ~20 GB after the zip is deleted. Fits a T4 runtime
#   but is tight — keep Drive mounted so you only pay this once.
#   Cold:  ~25-40 min env build + ~30-60 min vMAP download/extract
#   Warm:  ~3-5 min env restore + ~3-5 min data restore (needs Drive cache)
# ============================================================
set -eo pipefail

# ---------------------------------------------------------------------
# 0. Locate or clone the repo
# ---------------------------------------------------------------------
echo "[0] locating repo..."
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
  REPO_ROOT=${SNI_REPO_DIR:-/content/sni-slam}        # standalone -> clone
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
CONDA_PREFIX_PATH=/opt/conda
ENV_NAME=sni
ENV_PATH=$CONDA_PREFIX_PATH/envs/$ENV_NAME

# All 8 Replica scenes (folder names as the SNI-SLAM configs expect them).
SCENES="room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4"
VMAP_ZIP_URL="https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip"

if [ -d "$DRIVE_ROOT" ]; then
  DRIVE_OK=1
  mkdir -p "$DRIVE_CACHE"
  echo "  Google Drive mounted — caching + quota fallback enabled"
else
  DRIVE_OK=0
  echo "  Google Drive NOT mounted at $DRIVE_ROOT — no caching / no quota fallback."
  echo "    (To enable: in a Colab notebook cell run"
  echo "       from google.colab import drive; drive.mount('/content/drive')"
  echo "     then re-run this script.)"
fi

# ---------------------------------------------------------------------
# 1. GPU verify
# ---------------------------------------------------------------------
echo ""
echo "[1] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: no GPU detected. Select a T4 GPU runtime in Colab and re-run."
  exit 1
}

# ---------------------------------------------------------------------
# 2. apt deps
# ---------------------------------------------------------------------
echo ""
echo "[2] apt: libopenexr-dev, unzip, wget..."
sudo apt-get update -qq
sudo apt-get install -y -qq libopenexr-dev unzip wget

# ---------------------------------------------------------------------
# 3. miniconda install (if needed) + channel ToS
# ---------------------------------------------------------------------
echo ""
echo "[3] conda check..."
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
# environment.yaml uses the 'defaults' channel (repo.anaconda.com/pkgs/{main,r});
# recent conda refuses non-interactive installs until ToS is accepted. Idempotent.
echo "  accepting Anaconda channel ToS (pkgs/main, pkgs/r)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# ---------------------------------------------------------------------
# 4. Restore or build the 'sni' env (Py3.7 / PT1.11+cu113 / pytorch3d 0.7.1)
# ---------------------------------------------------------------------
echo ""
echo "[4] conda env 'sni'..."
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
  echo "  built. Will tar+upload to Drive at the env-cache step (if Drive mounted)."
  ENV_FRESHLY_BUILT=1
fi

# ---------------------------------------------------------------------
# 5. Import sanity-check inside the env
# ---------------------------------------------------------------------
echo ""
echo "[5] import sanity check..."
conda activate $ENV_NAME
python - <<'PY'
import torch, cv2
print(f"  torch       {torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  cv2         {cv2.__version__}")
for mod in ("pytorch3d", "open3d", "OpenEXR", "timm"):
    try:
        m = __import__(mod)
        print(f"  {mod:<11} {getattr(m,'__version__','OK')}")
    except Exception as e:
        print(f"  {mod:<11} FAIL  {e}")
PY

# ---------------------------------------------------------------------
# 6. Segmentation weights (authors' Drive) — restore / gdown / quota fallback
# ---------------------------------------------------------------------
echo ""
echo "[6] segmentation weights (seg/)..."
SEG_TARGET="$REPO_ROOT/seg"
mkdir -p "$SEG_TARGET"

if [ "$DRIVE_OK" = "1" ] && [ -f "$DRIVE_DATA_DIR/seg/dinov2_replica.pth" ]; then
  echo "  restoring seg/ from Drive cache"
  cp -r "$DRIVE_DATA_DIR/seg/"* "$SEG_TARGET/" 2>/dev/null || true
elif [ -f "$SEG_TARGET/dinov2_replica.pth" ] && [ -d "$SEG_TARGET/facebookresearch_dinov2_main" ]; then
  echo "  seg/ already present — skipping"
else
  echo "  downloading seg artifacts by file ID (authors' Drive folder 1BCu8...)..."
  pip install -q gdown
  STAGING=/content/sni_seg_dl
  mkdir -p "$STAGING"
  gdown 1ffPsH3NWK8GqoeSXDTGWo-QfC-LbhYvU -O "$STAGING/dinov2_replica.pth"               || echo "  WARN dinov2_replica.pth failed (quota?) — fallback below"
  gdown 12QnXVswA1yiet-zmhzkLNsjCpboOLQBS -O "$STAGING/facebookresearch_dinov2_main.zip" || true
  gdown 1A-pm0LWQZzevNdMZ14EP0z_b83t2-JuF -O "$STAGING/num_semantic_class.pkl"           || true
  gdown 1gRbgMobHhZCa63dcOSGpjKK7UUVK6vcW -O "$STAGING/semantic_classes.pkl"             || true

  # dinov2_replica.pth is popular -> Google per-file quota can block gdown with
  # "had many accesses". Fallback (Drive only): use a manual My-Drive copy.
  if [ "$DRIVE_OK" = "1" ] && [ ! -s "$STAGING/dinov2_replica.pth" ]; then
    hit=$(find "$DRIVE_ROOT" -maxdepth 3 -iname "*dinov2_replica*.pth" 2>/dev/null | head -1)
    if [ -n "$hit" ]; then
      echo "  quota fallback: dinov2_replica.pth <- $hit"
      cp "$hit" "$STAGING/dinov2_replica.pth"
    fi
  fi
  if [ ! -s "$STAGING/dinov2_replica.pth" ]; then
    echo "  MISSING dinov2_replica.pth — gdown quota-blocked and no My-Drive copy."
    echo "    Open https://drive.google.com/uc?id=1ffPsH3NWK8GqoeSXDTGWo-QfC-LbhYvU"
    echo "    -> 'Make a copy' into My Drive (your copy has no quota) -> re-run."
    exit 1
  fi

  cp -v "$STAGING/"* "$SEG_TARGET/" 2>/dev/null || true
  if [ -f "$SEG_TARGET/facebookresearch_dinov2_main.zip" ] && \
     [ ! -d "$SEG_TARGET/facebookresearch_dinov2_main" ]; then
    echo "  unzipping facebookresearch_dinov2_main.zip..."
    (cd "$SEG_TARGET" && unzip -q facebookresearch_dinov2_main.zip)
  fi
  rm -rf "$STAGING"
  if [ "$DRIVE_OK" = "1" ]; then
    mkdir -p "$DRIVE_DATA_DIR/seg"
    cp -r "$SEG_TARGET/"* "$DRIVE_DATA_DIR/seg/" 2>/dev/null || true
  fi
fi
[ -f "$SEG_TARGET/dinov2_replica.pth" ]           && echo "  seg/dinov2_replica.pth             ✓" || { echo "  seg/dinov2_replica.pth MISSING"; exit 1; }
[ -d "$SEG_TARGET/facebookresearch_dinov2_main" ] && echo "  seg/facebookresearch_dinov2_main/  ✓" || echo "  seg/facebookresearch_dinov2_main/  MISSING"
[ -f "$SEG_TARGET/semantic_classes.pkl" ]         && echo "  seg/semantic_classes.pkl           ✓" || echo "  seg/semantic_classes.pkl           MISSING"
[ -f "$SEG_TARGET/num_semantic_class.pkl" ]       && echo "  seg/num_semantic_class.pkl         ✓" || echo "  seg/num_semantic_class.pkl         MISSING"

# ---------------------------------------------------------------------
# 7. Data — all 8 scenes from vMAP vmap.zip (restore / download / extract)
# ---------------------------------------------------------------------
echo ""
echo "[7] data (8 Replica scenes from vMAP)..."
DATA_TARGET="$REPO_ROOT/data/replica"
mkdir -p "$DATA_TARGET"

scene_ready() { [ -f "$DATA_TARGET/$1/traj.txt" ] && [ -d "$DATA_TARGET/$1/semantic_class" ] && [ -d "$DATA_TARGET/$1/rgb" ]; }

have_all=1
for S in $SCENES; do scene_ready "$S" || have_all=0; done

if [ "$have_all" = "1" ]; then
  echo "  all 8 scenes already present at $DATA_TARGET — skipping fetch"
elif [ "$DRIVE_OK" = "1" ] && [ -d "$DRIVE_DATA_DIR/replica" ] && \
     [ -f "$DRIVE_DATA_DIR/replica/office_4/traj.txt" ] && \
     [ ! -f "$DRIVE_ROOT/Datasets/Replica/Replica.zip" ] && \
     ! ls -d "$DRIVE_ROOT/Datasets/Replica"/*/ >/dev/null 2>&1; then
  # NOTE: this vMAP cache is SKIPPED when authors' Datasets/Replica is present
  # (the next elif), so the authors' data takes precedence over the stale cache.
  echo "  restoring 8 scenes from Drive cache: $DRIVE_DATA_DIR/replica"
  cp -r "$DRIVE_DATA_DIR/replica/"* "$DATA_TARGET/" 2>/dev/null || true
elif [ "$DRIVE_OK" = "1" ] && { [ -f "$DRIVE_ROOT/Datasets/Replica/Replica.zip" ] || ls -d "$DRIVE_ROOT/Datasets/Replica"/*/ >/dev/null 2>&1; }; then
  # AUTHORS' Replica data (preferred over vMAP — fixes the 2.08x reproduction gap).
  # Robust to layout: SNI (rgb/,depth/,semantic_class/,traj.txt) OR NICE-SLAM
  # (results/frame*.jpg + depth*.png + traj.txt); room_0 or room0 naming.
  AR="$DRIVE_ROOT/Datasets/Replica"
  echo "  [authors' Replica] staging from $AR (preferred over vMAP)"
  python3 - "$AR" "$DATA_TARGET" "$SCENES" <<'PY'
import sys, os, glob, zipfile, shutil
ar, dst_root, scenes = sys.argv[1], sys.argv[2], sys.argv[3].split()
src = ar
zips = glob.glob(os.path.join(ar, '*.zip'))
have_extracted = any(os.path.isdir(os.path.join(ar, s)) or os.path.isdir(os.path.join(ar, s.replace('_',''))) for s in scenes)
if zips and not have_extracted:
    ex = '/content/authors_replica'; os.makedirs(ex, exist_ok=True)
    print(f'  extracting {os.path.basename(zips[0])} -> {ex}')
    with zipfile.ZipFile(zips[0]) as z: z.extractall(ex)
    src = ex
def scene_root(scene):
    for cand in (scene, scene.replace('_','')):
        for h in glob.glob(os.path.join(src, '**', cand), recursive=True):
            if os.path.isdir(h) and (os.path.isdir(os.path.join(h,'rgb')) or os.path.isdir(os.path.join(h,'results')) or os.path.exists(os.path.join(h,'traj.txt'))):
                return h
    return None
for s in scenes:
    root = scene_root(s)
    if not root: print(f'  WARN {s}: NOT FOUND under {src}'); continue
    d = os.path.join(dst_root, s); os.makedirs(d, exist_ok=True)
    if os.path.isdir(os.path.join(root,'rgb')):
        for sub in ('rgb','depth','semantic_class'):
            if os.path.isdir(os.path.join(root,sub)): shutil.copytree(os.path.join(root,sub), os.path.join(d,sub), dirs_exist_ok=True)
    elif os.path.isdir(os.path.join(root,'results')):
        os.makedirs(os.path.join(d,'rgb'), exist_ok=True); os.makedirs(os.path.join(d,'depth'), exist_ok=True)
        for f in glob.glob(os.path.join(root,'results','frame*.jpg')): shutil.copy(f, os.path.join(d,'rgb', os.path.basename(f)))
        for f in glob.glob(os.path.join(root,'results','depth*.png')): shutil.copy(f, os.path.join(d,'depth', os.path.basename(f)))
        if os.path.isdir(os.path.join(root,'semantic_class')): shutil.copytree(os.path.join(root,'semantic_class'), os.path.join(d,'semantic_class'), dirs_exist_ok=True)
    for t in ('traj_w_c.txt','traj.txt'):
        if os.path.exists(os.path.join(root,t)): shutil.copy(os.path.join(root,t), os.path.join(d,'traj.txt')); break
    nrgb=len(glob.glob(os.path.join(d,'rgb','*'))); sem=os.path.isdir(os.path.join(d,'semantic_class')); tr=os.path.exists(os.path.join(d,'traj.txt'))
    print(f'  {s}: rgb={nrgb} semantic_class={"yes" if sem else "NO(!)"}  traj={"yes" if tr else "NO(!)"}  root={root}')
PY
  if [ "$DRIVE_OK" = "1" ]; then mkdir -p "$DRIVE_DATA_DIR/replica"; cp -r "$DATA_TARGET/"* "$DRIVE_DATA_DIR/replica/" 2>/dev/null || true; fi
else
  ZIP=/content/vmap.zip
  EXTRACT=/content/vmap_extract
  echo "  downloading vmap.zip (44.8 GB, resumable)..."
  wget -c -O "$ZIP" "$VMAP_ZIP_URL"

  echo "  indexing zip members..."
  unzip -Z1 "$ZIP" > /tmp/vmap_members.txt
  echo "  total members: $(wc -l < /tmp/vmap_members.txt)"

  # Discover each scene's traj-00 root WITHOUT hardcoding the internal nesting
  # (could be <prefix>/room_0/imap/00 or .../vmap/00 etc.). Prefer the '00'
  # trajectory (== iMAP trajectory, what SNI-SLAM/NICE-SLAM use); fall back to
  # any semantic_class dir for the scene. NOTE: `grep ... | head -1` is unsafe
  # under `set -eo pipefail` because head closing the pipe early triggers
  # SIGPIPE on grep, which pipefail propagates and set -e turns into a silent
  # script death. Use `grep -m 1 ... || true` instead — built-in match limit
  # avoids the SIGPIPE, and `|| true` covers the legitimate no-match case.
  declare -A SCENE_ROOT
  PATTERNS=()
  for S in $SCENES; do
    line=$(grep -m 1 -E "(^|/)${S}/.*00/semantic_class/" /tmp/vmap_members.txt || true)
    [ -z "$line" ] && line=$(grep -m 1 -E "(^|/)${S}/.*semantic_class/" /tmp/vmap_members.txt || true)
    if [ -z "$line" ]; then
      echo "  WARN: no semantic_class entry found for $S in vmap.zip"
      continue
    fi
    root=$(echo "$line" | sed -E 's#/semantic_class/.*##')
    SCENE_ROOT[$S]="$root"
    echo "    $S -> $root"
    PATTERNS+=("$root/rgb/*" "$root/depth/*" "$root/semantic_class/*" "$root/traj_w_c.txt" "$root/traj.txt")
  done

  if [ "${#PATTERNS[@]}" -eq 0 ]; then
    echo "  FATAL: could not locate any scene inside vmap.zip. First 40 members:"
    head -40 /tmp/vmap_members.txt
    echo "  (Internal nesting differs from expectations — paste the above for a patch.)"
    exit 1
  fi

  echo "  extracting traj-00 rgb+depth+semantic_class+traj for matched scenes..."
  mkdir -p "$EXTRACT"
  unzip -q -o "$ZIP" "${PATTERNS[@]}" -d "$EXTRACT" || true

  for S in $SCENES; do
    root="${SCENE_ROOT[$S]:-}"
    [ -z "$root" ] && continue
    dst="$DATA_TARGET/$S"
    mkdir -p "$dst"
    cp -r "$EXTRACT/$root/rgb"            "$dst/" 2>/dev/null || true
    cp -r "$EXTRACT/$root/depth"          "$dst/" 2>/dev/null || true
    cp -r "$EXTRACT/$root/semantic_class" "$dst/" 2>/dev/null || true
    if   [ -f "$EXTRACT/$root/traj_w_c.txt" ]; then cp "$EXTRACT/$root/traj_w_c.txt" "$dst/traj.txt"
    elif [ -f "$EXTRACT/$root/traj.txt" ];     then cp "$EXTRACT/$root/traj.txt"     "$dst/traj.txt"
    else echo "  WARN: no trajectory file for $S"; fi
  done

  echo "  cleaning up zip + extract scratch..."
  rm -f "$ZIP"; rm -rf "$EXTRACT"

  if [ "$DRIVE_OK" = "1" ]; then
    echo "  caching extracted 8 scenes to Drive ($DRIVE_DATA_DIR/replica)..."
    mkdir -p "$DRIVE_DATA_DIR/replica"
    cp -r "$DATA_TARGET/"* "$DRIVE_DATA_DIR/replica/" 2>/dev/null || true
  fi
fi

echo "  scene inventory:"
ready_count=0
for S in $SCENES; do
  if scene_ready "$S"; then
    n=$(ls "$DATA_TARGET/$S/rgb" 2>/dev/null | wc -l)
    echo "    $S  ✓  ($n rgb frames)"
    ready_count=$((ready_count + 1))
  else
    echo "    $S  MISSING"
  fi
done

# Hard assertion: at least one scene must be staged, else downstream SLAM
# crashes with a confusing IndexError 7s into run.py. Fail loud here instead.
expected=$(echo $SCENES | wc -w)
if [ "$ready_count" -eq 0 ]; then
  echo ""
  echo "FATAL [7]: no scenes staged after data step. data/replica/ is empty."
  echo "  Common causes:"
  echo "    - vmap.zip corrupt or truncated -> rm /content/vmap.zip and re-run"
  echo "    - internal zip layout differs from expectations -> paste /tmp/vmap_members.txt head"
  echo "    - set -eo pipefail killed an earlier grep|head pipeline -> already patched"
  exit 1
elif [ "$ready_count" -lt "$expected" ]; then
  echo ""
  echo "WARN [7]: only $ready_count/$expected scenes staged. Sweep will skip missing ones."
fi

# ---------------------------------------------------------------------
# 8. Semantic-label compatibility gate (vMAP pixels vs authors' pkl)
#    The authors ship ONE semantic_classes.pkl; datasets.py remaps each raw
#    Replica ID to 0..N-1. Any ID present in the images but absent from the
#    pkl would stay unmapped and corrupt the (pretrained) semantic head. Check
#    every staged scene before declaring success.
# ---------------------------------------------------------------------
echo ""
echo "[8] semantic-label validation (all scenes)..."
SCENES_ENV="$SCENES" DATA_ENV="$DATA_TARGET" SEG_ENV="$SEG_TARGET" python - <<'PY'
import os, glob, pickle, sys
import numpy as np, cv2
scenes = os.environ["SCENES_ENV"].split()
data   = os.environ["DATA_ENV"]
seg    = os.environ["SEG_ENV"]
with open(os.path.join(seg, "semantic_classes.pkl"), "rb") as f:
    classes = set(int(x) for x in pickle.load(f))
print(f"  authors' semantic_classes.pkl: {len(classes)} classes")
bad = {}
for s in scenes:
    files = sorted(glob.glob(os.path.join(data, s, "semantic_class", "semantic_class_*.png")))
    if not files:
        print(f"  {s:<10} no semantic_class images — skipped")
        continue
    sample = files[:: max(1, len(files)//25)][:25]   # ~25 frames spread across the scene
    ids = set()
    for p in sample:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        ids |= set(int(v) for v in np.unique(img))
    missing = sorted(ids - classes)
    status = "OK" if not missing else f"MISMATCH missing={missing}"
    print(f"  {s:<10} {len(ids):>3} unique IDs  -> {status}")
    if missing:
        bad[s] = missing
if bad:
    print("\n  FAIL: vMAP semantic IDs not covered by the authors' pkl for:",
          ", ".join(bad.keys()))
    print("  The pretrained segmentation head would be inconsistent on these scenes.")
    sys.exit(2)
print("  PASS: every scene's semantic IDs are covered by the authors' pkl.")
PY

# ---------------------------------------------------------------------
# 9. Idempotent config-path safeguard (no-op if already relative)
# ---------------------------------------------------------------------
echo ""
echo "[9] config path safeguard..."
SNI_YAML="$REPO_ROOT/configs/SNI-SLAM.yaml"
if grep -q "/data0/" "$SNI_YAML"; then
  sed -i "s|/data0/nerf/sni-slam/seg/dinov2_replica.pth|seg/dinov2_replica.pth|g" "$SNI_YAML"
  sed -i "s|/data0/nerf/sni-slam/seg/|seg/|g" "$SNI_YAML"
  echo "  patched $SNI_YAML"
else
  echo "  configs already use repo-relative paths"
fi

# ---------------------------------------------------------------------
# 10. Cache the env tar back to Drive (only if freshly built and Drive mounted)
# ---------------------------------------------------------------------
echo ""
echo "[10] env cache..."
if [ "${ENV_FRESHLY_BUILT:-0}" = "1" ] && [ "$DRIVE_OK" = "1" ]; then
  echo "  tarring $ENV_PATH -> $DRIVE_ENV_TAR (~5-10 min, ~3-4 GB)"
  (cd "$CONDA_PREFIX_PATH/envs" && tar -czf "$DRIVE_ENV_TAR" "$ENV_NAME")
  echo "  cached."
else
  echo "  nothing to cache (env not freshly built, or Drive not mounted)."
fi

# ---------------------------------------------------------------------
# Done — setup-only. Print the run + eval plan for all 8 scenes.
# ---------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup complete: env + seg weights + all 8 Replica scenes staged,"
echo "and semantic labels validated against the authors' pretrained head."
echo ""
echo "Run all 8 scenes (1 run each, ~16 h total; resumable across sessions):"
echo "  conda activate $ENV_NAME && cd $REPO_ROOT"
echo "  SCENES='room0 room1 room2 office0 office1 office2 office3 office4' \\"
echo "    bash scripts/run_replica_subset.sh"
echo ""
echo "That runner saves each scene to Drive, skips finished scenes, and writes"
echo "ATE + reconstruction metrics to MyDrive/Outputs/sni_slam_replica/_eval/."
echo "Compare the 8-scene average against scripts/PAPER_BASELINES.md"
echo "(paper: ATE RMSE 0.456 cm, Depth L1 0.766 cm, Comp.Ratio 96.6%)."
echo ""
echo "Single-scene smoke test first (recommended, ~2 h):"
echo "  python -W ignore run.py configs/Replica/room0.yaml"
echo "============================================================"
