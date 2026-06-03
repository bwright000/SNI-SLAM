#!/bin/bash
# ============================================================
# run_crcd_overnight.sh — SNI-SLAM on a CRCD-Published snippet.
#
# Per-snippet pipeline (resumable, sentinel-gated):
#   0. Pre-flight: GPU type, calib_pkl, dinov2_replica.pth, MoGe importable
#   1. Stage raw snippet from Drive to /content/crcd_raw/<EP>/snippet_NNN
#   2. preprocess_crcd_for_sni.py: rectify rgb + sem, write traj.txt, emit yaml
#   3. MoGe-2 depth (reuse Drive cache if frame counts match, else generate)
#   4. SNI-SLAM run (run.py)
#   5. Eval: raw ATE via eval_ate.py + Sim3 ATE inline from final ckpt
#   6. Ship to Drive — only mark .DONE on confirmed copy
#
# Pre-flight + each phase exits non-zero on the first failure: a partial run
# never leaves a sentinel that would silently skip retry.
#
# Usage:
#   bash scripts/run_crcd_overnight.sh f1_002
#   bash scripts/run_crcd_overnight.sh c1_001
#   bash scripts/run_crcd_overnight.sh c2_001
#   bash scripts/run_crcd_overnight.sh f3_007
#
# Or all four in order (smallest first):
#   for s in c1_001 f3_007 c2_001 f1_002; do
#     bash scripts/run_crcd_overnight.sh $s
#   done
# ============================================================
set -u
set -o pipefail

SNIPPET_KEY="${1:-}"
if [ -z "$SNIPPET_KEY" ]; then
  echo "Usage: $0 <snippet_key>  where snippet_key in {f1_002, c1_001, c2_001, f3_007}"
  exit 2
fi

case "$SNIPPET_KEY" in
  f1_002) SNIPPET_ID=F1_002; DRIVE_EP=F_1; DRIVE_SNIP=snippet_002 ;;
  c1_001) SNIPPET_ID=C1_001; DRIVE_EP=C_1; DRIVE_SNIP=snippet_001 ;;
  c2_001) SNIPPET_ID=C2_001; DRIVE_EP=C_2; DRIVE_SNIP=snippet_001 ;;
  f3_007) SNIPPET_ID=F3_007; DRIVE_EP=F_3; DRIVE_SNIP=snippet_007 ;;
  *) echo "Unknown snippet_key: $SNIPPET_KEY"; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CRCD_DRIVE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$CRCD_DRIVE_ROOT/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
RAW_STAGED=/content/crcd_raw/$SNIPPET_ID
STAGED=$REPO_ROOT/data/CRCD/$SNIPPET_ID
CFG=$REPO_ROOT/configs/CRCD/${SNIPPET_KEY}.yaml
LOCAL_OUT=$REPO_ROOT/output/CRCD/$SNIPPET_ID/test
DRIVE_OUT=/content/drive/MyDrive/Outputs/sni_slam_crcd/$SNIPPET_ID
MASTER_LOG=$REPO_ROOT/output/run_crcd_overnight_${SNIPPET_ID}.log
DRIVE_SNIPPET_DIR=$CRCD_DRIVE_ROOT/$DRIVE_EP/$DRIVE_SNIP
DINOV2_PTH=$REPO_ROOT/seg/dinov2_replica.pth

mkdir -p "$DRIVE_OUT" "$RAW_STAGED" "$(dirname "$MASTER_LOG")"
touch "$MASTER_LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SNIPPET_ID] $*" | tee -a "$MASTER_LOG"; }

scene_complete() {
  local m="$1/mesh/final_mesh_color_culled.ply"
  [ -f "$m" ] && [ "$(stat -c%s "$m" 2>/dev/null || echo 0)" -gt 10000 ]
}

log "=== $SNIPPET_ID start ==="
log "Drive raw : $DRIVE_SNIPPET_DIR"
log "Staged    : $STAGED"
log "Config    : $CFG"
log "Drive out : $DRIVE_OUT"

# ---------------------------------------------------------------------
# Short-circuit: drive output already complete
# ---------------------------------------------------------------------
if [ -f "$DRIVE_OUT/.DONE" ] && scene_complete "$DRIVE_OUT"; then
  log "SKIP — $DRIVE_OUT/.DONE present and final mesh on Drive. Nothing to do."
  exit 0
fi

# ---------------------------------------------------------------------
# PHASE 0 — pre-flight checks (no work, just fail fast on missing deps)
# ---------------------------------------------------------------------
log "PHASE 0: pre-flight"

# GPU type — paper config OOMs on T4 (~28 MB/frame mapper growth, see
# project_sni_slam_replica_stall_20260530). A100 is required for full sequences.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  log "  GPU: $GPU_NAME"
  if [[ ! "$GPU_NAME" =~ A100 ]]; then
    log "FATAL: need A100 (got '$GPU_NAME'). T4 will OOM-deadlock at ~frame 2000."
    log "       Stop the runbook and request a fresh A100 runtime before retrying."
    exit 10
  fi
else
  log "FATAL: nvidia-smi not on PATH"
  exit 10
fi

# Required artefacts on disk
if [ ! -f "$CALIB_PKL" ]; then
  log "FATAL: calib_pkl not found at $CALIB_PKL — stage CRCD-Published/cam_calib/ to Drive."
  exit 11
fi
if [ ! -f "$DINOV2_PTH" ]; then
  log "FATAL: $DINOV2_PTH not found — colab_setup_sni.sh must download seg/dinov2_replica.pth."
  exit 12
fi
if [ ! -f "$REPO_ROOT/Addons/preprocess/preprocess_crcd_for_sni.py" ]; then
  log "FATAL: preprocess script missing (git pull not run?)"
  exit 13
fi
if [ ! -f "$REPO_ROOT/Addons/depth/generate_depth_moge.py" ]; then
  log "FATAL: generate_depth_moge.py missing"
  exit 13
fi
log "  pre-flight OK"

# ---------------------------------------------------------------------
# PHASE 1 — stage raw snippet locally
# ---------------------------------------------------------------------
log "PHASE 1: staging raw from Drive"
if [ ! -f "$RAW_STAGED/.STAGED" ]; then
  if [ ! -d "$DRIVE_SNIPPET_DIR" ]; then
    log "FATAL: drive snippet not found at $DRIVE_SNIPPET_DIR"
    exit 3
  fi
  mkdir -p "$RAW_STAGED"
  (cd "$DRIVE_SNIPPET_DIR" && tar cf - rgb semantic_instance groundtruth.txt 2>/dev/null) | \
    (cd "$RAW_STAGED" && tar xf -) || {
      log "FATAL: tar copy of raw snippet failed"
      exit 3
    }
  if [ -d "$DRIVE_SNIPPET_DIR/depth" ] && [ -n "$(ls "$DRIVE_SNIPPET_DIR/depth" 2>/dev/null)" ]; then
    log "  Drive depth/ non-empty — staging too"
    (cd "$DRIVE_SNIPPET_DIR" && tar cf - depth) | (cd "$RAW_STAGED" && tar xf -) || {
      log "FATAL: tar copy of depth/ failed"
      exit 3
    }
  fi
  touch "$RAW_STAGED/.STAGED"
fi
N_RAW_RGB=$(ls "$RAW_STAGED/rgb"/*.png 2>/dev/null | wc -l)
N_RAW_SEM=$(ls "$RAW_STAGED/semantic_instance"/*.png 2>/dev/null | wc -l)
N_RAW_DEPTH=0
[ -d "$RAW_STAGED/depth" ] && N_RAW_DEPTH=$(ls "$RAW_STAGED/depth"/*.png 2>/dev/null | wc -l)
log "  staged: rgb=$N_RAW_RGB sem=$N_RAW_SEM depth=$N_RAW_DEPTH gt=$(wc -l < "$RAW_STAGED/groundtruth.txt" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------------
# PHASE 2 — preprocess to SNI-SLAM layout + emit config
# Sentinel touched ONLY on preprocess success — partial runs retry cleanly.
# ---------------------------------------------------------------------
log "PHASE 2: preprocess"
if [ ! -f "$STAGED/.PREPROCESSED" ]; then
  python3 Addons/preprocess/preprocess_crcd_for_sni.py \
    --snippet_dir "$RAW_STAGED" \
    --calib_pkl   "$CALIB_PKL" \
    --output_dir  "$STAGED" \
    --emit_config "$CFG" \
    --snippet_id  "$SNIPPET_ID" \
    --pad_m       0.25 \
    2>&1 | tee -a "$MASTER_LOG"
  pp_rc=${PIPESTATUS[0]}
  if [ "$pp_rc" -ne 0 ]; then
    log "FATAL: preprocess exited rc=$pp_rc — NOT touching .PREPROCESSED so retry re-runs."
    exit 4
  fi
  touch "$STAGED/.PREPROCESSED"
fi

N_RGB=$(ls "$STAGED/rgb"/*.png 2>/dev/null | wc -l)
N_SEM=$(ls "$STAGED/semantic_class"/*.png 2>/dev/null | wc -l)
N_TRAJ=0
[ -f "$STAGED/traj.txt" ] && N_TRAJ=$(grep -c '[^[:space:]]' "$STAGED/traj.txt" 2>/dev/null || echo 0)
log "  preprocessed: rgb=$N_RGB sem=$N_SEM traj=$N_TRAJ"
if [ "$N_RGB" -lt 100 ]; then
  log "FATAL: preprocess looks incomplete (rgb=$N_RGB)"
  exit 4
fi
if [ "$N_RGB" -ne "$N_SEM" ] || [ "$N_RGB" -ne "$N_TRAJ" ]; then
  log "FATAL: frame count mismatch after preprocess (rgb=$N_RGB sem=$N_SEM traj=$N_TRAJ)"
  exit 4
fi
if [ ! -f "$CFG" ] || [ ! -s "$CFG" ]; then
  log "FATAL: config not emitted (or empty) at $CFG"
  exit 4
fi

# ---------------------------------------------------------------------
# PHASE 3 — MoGe-2 depth
# ---------------------------------------------------------------------
log "PHASE 3: MoGe-2 depth"
N_DEPTH_OUT=$(ls "$STAGED/depth"/depth_*.png 2>/dev/null | wc -l)
if [ "$N_DEPTH_OUT" -eq "$N_RGB" ]; then
  log "  depth/ already complete ($N_DEPTH_OUT/$N_RGB) — skip"
else
  mkdir -p "$STAGED/depth"
  N_DRIVE_DEPTH=0
  [ -d "$RAW_STAGED/depth" ] && N_DRIVE_DEPTH=$(ls "$RAW_STAGED/depth"/*.png 2>/dev/null | wc -l)

  if [ "$N_DRIVE_DEPTH" -eq "$N_RGB" ]; then
    # Path A: Drive snippet shipped pre-baked MoGe depth from an earlier session
    # (e.g. F1_002). Remap by index, NOT by fuzzy filename matching — both lists
    # are sorted, so rgb_orig[i] pairs with sorted(depth_in)[i].
    log "  Drive depth/ count matches ($N_DRIVE_DEPTH==$N_RGB) — remap names"
    python3 - 2>&1 | tee -a "$MASTER_LOG" <<PYEOF
import os, shutil
RAW = "$RAW_STAGED"
STAGED = "$STAGED"
rgb_orig = sorted(f for f in os.listdir(os.path.join(RAW, 'rgb')) if f.endswith('.png'))
depth_files = sorted(f for f in os.listdir(os.path.join(RAW, 'depth')) if f.endswith('.png'))
assert len(rgb_orig) == len(depth_files), f"rgb={len(rgb_orig)} depth={len(depth_files)}"
n = 0
for i, (rgb_name, depth_name) in enumerate(zip(rgb_orig, depth_files)):
    src = os.path.join(RAW, 'depth', depth_name)
    dst = os.path.join(STAGED, 'depth', f'depth_{i:06d}.png')
    if not os.path.exists(dst):
        shutil.copy2(src, dst); n += 1
print(f"  remapped {n} depth pngs (rgb_orig[0]={rgb_orig[0]!r} <- depth_files[0]={depth_files[0]!r})")
PYEOF
  else
    log "  Drive depth count mismatch ($N_DRIVE_DEPTH vs $N_RGB) — generate fresh MoGe-2"
    rm -rf "$STAGED/_moge_tmp" 2>/dev/null || true
    # MoGe-2 needs torch >= 2.0; the sni conda env ships torch 2.x. Check import,
    # install only if missing, then VERIFY the install actually loads the class.
    if ! python3 -c 'from moge.model.v2 import MoGeModel' 2>/dev/null; then
      log "  installing MoGe-2..."
      python3 -m pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tee -a "$MASTER_LOG"
      python3 -c 'from moge.model.v2 import MoGeModel' 2>&1 | tee -a "$MASTER_LOG" || {
        log "FATAL: MoGe-2 import still fails after pip install"
        exit 5
      }
    fi
    MOGE_TMP="$STAGED/_moge_tmp"
    mkdir -p "$MOGE_TMP/in" "$MOGE_TMP/npy"
    for f in "$STAGED"/rgb/rgb_*.png; do
      fid=$(basename "$f" .png | sed 's/^rgb_//')
      ln -sf "$f" "$MOGE_TMP/in/${fid}-left.png"
    done
    python3 "$REPO_ROOT/Addons/depth/generate_depth_moge.py" \
      --rgb "$MOGE_TMP/in" --out "$MOGE_TMP/npy" \
      --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 \
      2>&1 | tee -a "$MASTER_LOG"
    moge_rc=${PIPESTATUS[0]}
    if [ "$moge_rc" -ne 0 ]; then
      log "FATAL: generate_depth_moge.py exited rc=$moge_rc"
      exit 5
    fi
    # NPY -> uint16 PNG at scale 10000. Iterate by FRAME ID (not glob) so missing
    # NPYs are individually visible in the log.
    python3 - 2>&1 | tee -a "$MASTER_LOG" <<PYEOF
import os, cv2, numpy as np
NPY = "$MOGE_TMP/npy"
OUT = "$STAGED/depth"
N   = $N_RGB
os.makedirs(OUT, exist_ok=True)
missing, written = [], 0
for i in range(N):
    fid = f"{i:06d}"
    src = os.path.join(NPY, f"{fid}-left_depth.npy")
    dst = os.path.join(OUT, f"depth_{fid}.png")
    if os.path.exists(dst):
        continue
    if not os.path.exists(src):
        missing.append(fid); continue
    d = np.load(src).astype(np.float32)
    tmp = os.path.join(OUT, f".depth_{fid}.tmp.png")
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, dst)
    written += 1
print(f"  npy->png wrote {written} (missing NPYs: {len(missing)})")
if missing:
    print(f"  missing frame IDs: {missing[:20]}{'...' if len(missing) > 20 else ''}")
PYEOF
    # Persist back to Drive snippet for future reuse
    log "  syncing depth back to Drive snippet"
    python3 - 2>&1 | tee -a "$MASTER_LOG" <<PYEOF
import os, shutil
RAW = "$RAW_STAGED"
DRIVE_SNIP = "$DRIVE_SNIPPET_DIR"
STAGED = "$STAGED"
os.makedirs(os.path.join(DRIVE_SNIP, 'depth'), exist_ok=True)
rgb_orig = sorted(f for f in os.listdir(os.path.join(RAW, 'rgb')) if f.endswith('.png'))
copied = 0
for i, orig in enumerate(rgb_orig):
    src = os.path.join(STAGED, 'depth', f'depth_{i:06d}.png')
    if not os.path.exists(src):
        continue
    dst = os.path.join(DRIVE_SNIP, 'depth', orig)
    if not os.path.exists(dst):
        shutil.copy2(src, dst); copied += 1
print(f"  Drive sync copied={copied}")
PYEOF
    rm -rf "$MOGE_TMP"
  fi
fi
N_DEPTH_OUT=$(ls "$STAGED/depth"/depth_*.png 2>/dev/null | wc -l)
if [ "$N_DEPTH_OUT" -ne "$N_RGB" ]; then
  log "FATAL: depth count mismatch after PHASE 3 (depth=$N_DEPTH_OUT rgb=$N_RGB)"
  exit 5
fi

# ---------------------------------------------------------------------
# PHASE 4 — SNI-SLAM
# ---------------------------------------------------------------------
log "PHASE 4: SNI-SLAM run"
if [ ! -f "$CFG" ]; then
  log "FATAL: config disappeared at $CFG"
  exit 6
fi
if scene_complete "$LOCAL_OUT"; then
  log "  local output already complete — skip SLAM"
else
  t0=$(date +%s)
  python -W ignore run.py "$CFG" 2>&1 | tee -a "$MASTER_LOG"
  rc=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - t0 ))
  log "  SLAM rc=$rc elapsed=${elapsed}s"
  if ! scene_complete "$LOCAL_OUT"; then
    log "FATAL: SLAM did not produce final_mesh_color_culled.ply"
    exit 6
  fi
fi

# ---------------------------------------------------------------------
# PHASE 5 — eval (raw ATE via eval_ate.py + Sim3 ATE inline from final ckpt)
# SNI-SLAM stores estimate_c2w_list inside the ckpt tar (Logger.py:65) and does
# NOT write a plain-text est_c2w_data.txt. We load the tar directly here.
# ---------------------------------------------------------------------
log "PHASE 5: ATE eval (raw + Sim3 from final ckpt)"
EVAL_OUT=$DRIVE_OUT/metrics.txt
: > "$EVAL_OUT"
{
  echo "# $SNIPPET_ID metrics — $(date)"
  echo ""
  echo "## ATE (eval_ate.py, raw)"
} >> "$EVAL_OUT"
python src/tools/eval_ate.py "$CFG" >> "$EVAL_OUT" 2>&1 || \
  echo "(eval_ate.py exited with error)" >> "$EVAL_OUT"
{
  echo ""
  echo "## ATE (Sim3-aligned — see project_crcd_f1_002_sim3_finding_20260603)"
} >> "$EVAL_OUT"
python3 - >> "$EVAL_OUT" 2>&1 <<PYEOF || true
import os, glob, numpy as np, torch
CKPT_DIR = "$LOCAL_OUT/ckpts"
GT       = "$RAW_STAGED/groundtruth.txt"

def parse_tum(p):
    poses = []
    with open(p) as f:
        for line in f:
            v = line.strip().split()
            if not v or v[0].startswith('#'):
                continue
            if len(v) >= 8:   # matches preprocess: timestamp + tx ty tz qx qy qz qw
                poses.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(poses)

def horn(m, d, with_scale=False):
    mc = m.mean(0); dc = d.mean(0); mm = m - mc; dd = d - dc
    H = mm.T @ dd; U, S, Vt = np.linalg.svd(H)
    ds = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, ds]); R = Vt.T @ D @ U.T
    s = (S * np.array([1, 1, ds])).sum() / (mm * mm).sum() if with_scale else 1.0
    t = dc - s * R @ mc
    return (s * (R @ m.T)).T + t, s

ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.tar")))
if not ckpts or not os.path.isfile(GT):
    print(f"(missing ckpt or gt — skipping Sim3 block; ckpts={len(ckpts)} gt_exists={os.path.isfile(GT)})")
    raise SystemExit
ckpt = torch.load(ckpts[-1], map_location='cpu')
est_c2w_list = ckpt['estimate_c2w_list']     # tensor [N, 4, 4]
idx = int(ckpt['idx'])
# est_c2w_list is in the SAME frame as the GT (the loader has flipped GT into
# OpenGL, so estimates are OpenGL too). Translation is invariant to the flip.
est_t = est_c2w_list[:idx + 1, :3, 3].cpu().numpy().astype(np.float64)
g     = parse_tum(GT)
n     = min(len(est_t), len(g))
if n < 10:
    print(f"(too few matched poses for Sim3: n={n})")
    raise SystemExit
e, g = est_t[:n], g[:n]

ase3, _  = horn(e, g, with_scale=False); ese3  = np.linalg.norm(ase3  - g, axis=1) * 1000
asim3, s = horn(e, g, with_scale=True);  esim3 = np.linalg.norm(asim3 - g, axis=1) * 1000
gt_path  = np.linalg.norm(np.diff(g, axis=0), axis=1).sum() * 1000
est_path = np.linalg.norm(np.diff(e, axis=0), axis=1).sum() * 1000
print(f"frames matched : {n}")
print(f"SE3  ATE Mean   : {ese3.mean():.3f} mm")
print(f"SE3  ATE RMSE   : {np.sqrt((ese3**2).mean()):.3f} mm")
print(f"Sim3 ATE Mean   : {esim3.mean():.3f} mm")
print(f"Sim3 ATE RMSE   : {np.sqrt((esim3**2).mean()):.3f} mm")
print(f"Sim3 scale s    : {s:.6f}")
print(f"GT  path length : {gt_path:.2f} mm")
print(f"Est path length : {est_path:.2f} mm")
PYEOF
echo "## Eval completed at $(date)" >> "$EVAL_OUT"
log "  metrics -> $EVAL_OUT"

# ---------------------------------------------------------------------
# PHASE 6 — ship to Drive  (write .DONE only on confirmed copy)
# ---------------------------------------------------------------------
log "PHASE 6: ship to Drive"
mkdir -p "$DRIVE_OUT"
if cp -r "$LOCAL_OUT/"* "$DRIVE_OUT/" 2>>"$MASTER_LOG"; then
  if scene_complete "$DRIVE_OUT"; then
    touch "$DRIVE_OUT/.DONE"
    log "  shipped to $DRIVE_OUT"
  else
    log "FATAL: Drive copy completed but final mesh not present on Drive — leaving .DONE off."
    exit 7
  fi
else
  log "FATAL: Drive copy failed (cp -r returned non-zero). Leaving .DONE off."
  rm -f "$DRIVE_OUT/.DONE" 2>/dev/null || true
  exit 7
fi

cp "$MASTER_LOG" "$DRIVE_OUT/run.log" 2>/dev/null || true
log "=== $SNIPPET_ID done ==="
