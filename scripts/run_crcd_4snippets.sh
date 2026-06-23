#!/bin/bash
# ============================================================================
# SNI-SLAM CRCD 4-snippet batch — paired with DDS-SLAM crcd_paperfaith_lrfix.
#
# Per-snippet pipeline (sentinel-gated, resumable):
#   Phase 0    pre-flight (A100, calib_pkl, dinov2, MoGe-2 importable)
#   Phase 1    stage raw — TARBALL-FIRST (reuses DDS-SLAM tarballs on Drive)
#   Phase 2    preprocess_crcd_for_sni.py: rectify rgb + sem + traj.txt
#   Phase 2.5  ALSO emit rectified_calib.txt for Phase 1.6 stereo anchor
#   Phase 3    MoGe-2 depth generation (skip if Drive cache complete)
#   Phase 3.6  stereo anchor SGBM calibration; if sc_factor != 1.0 (>10% off),
#              REGENERATE depth PNGs at corrected scale (SNI-SLAM has no
#              sc_factor knob; cfg['scale'] is a global divisor only)
#   Phase 4    motion profile sanity (sentinel-flagged, non-blocking)
#   Phase 5    SNI-SLAM run.py
#   Phase 6    eval: raw + SE3 + Sim3 + est_path/GT_path + per-axis Pearson
#              via eval_traj_extended.py + Depth L1 via eval_recon.py
#   Phase 7    ship to Drive
#
# After all 4: Phase 8 combined 4-snippet summary for DDS-SLAM comparison.
#
# Order: F_3/007 (300) -> C_1/001 (360) -> C_2/001 (730) -> F_1/002 (1287)
# A100 wall: ~4-5 hr total.
#
# Configs (already pushed via 1403de8):
#   configs/CRCD/crcd_sni_base.yaml  + per-snippet inheritors
# ============================================================================
set +u   # NB: some phase blocks rely on optional env vars
set -o pipefail

DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/sni_crcd_4snippets_${DATE}
mkdir -p "$DRIVE_ROOT"
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== sni-slam crcd 4-snippets start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }

CRCD_DRIVE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$CRCD_DRIVE_ROOT/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
DINOV2_PTH=/content/sni-slam/seg/dinov2_replica.pth
REPO_ROOT=/content/sni-slam
# Drive cache for MoGe input depth + sc_factor sentinel.  Workflow w7imqnt4t
# chose this for atomicity (PNG + sc_factor travel together) and idempotency
# (skip MoGe regen if DDS-SLAM already produced it).
DRIVE_CACHE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2

# Snippet table: key episode snippet_id slam_config frames
SNIPPETS=(
  "f3_007  F_3  snippet_007  configs/CRCD/f3_007.yaml  300"
  "c1_001  C_1  snippet_001  configs/CRCD/c1_001.yaml  360"
  "c2_001  C_2  snippet_001  configs/CRCD/c2_001.yaml  730"
  "f1_002  F_1  snippet_002  configs/CRCD/f1_002.yaml  1287"
)

# ----------------------------------------------------------------------------
# Pre-flight checks (run once)
# ----------------------------------------------------------------------------
phase 0 "pre-flight"
[ -d /content/drive/MyDrive ] || { echo "FATAL: Drive not mounted"; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  echo "  GPU: $GPU"
  if [[ ! "$GPU" =~ A100 ]]; then
    echo "  WARN: non-A100 GPU.  SNI-SLAM mapper grows ~28 MB/frame; T4 likely OOM-deadlock."
    echo "       Continuing — user may need to abort + switch."
  fi
else
  echo "FATAL: nvidia-smi not on PATH"; exit 1
fi
[ -f "$CALIB_PKL" ] || { echo "FATAL: calib_pkl missing at $CALIB_PKL"; exit 1; }
[ -f "$DINOV2_PTH" ] || { echo "FATAL: dinov2 pretrained missing at $DINOV2_PTH (run colab_setup_sni.sh)"; exit 1; }
[ -f "$REPO_ROOT/Addons/preprocess/preprocess_crcd_for_sni.py" ] || { echo "FATAL: preprocess missing"; exit 1; }
[ -f "$REPO_ROOT/Addons/depth/generate_depth_moge.py" ] || { echo "FATAL: MoGe gen missing"; exit 1; }
[ -f "$REPO_ROOT/src/tools/eval_traj_extended.py" ] || { echo "FATAL: eval_traj_extended.py missing (git pull?)"; exit 1; }

cd "$REPO_ROOT"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull"
else
  git fetch && git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD) || echo "  WARN: non-FF on remote, continuing"
fi

ensure_moge() {
  if ! python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null; then
    echo "  installing MoGe-2..."
    pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -3
    python -c 'from moge.model.v2 import MoGeModel' \
      || { echo "FATAL: MoGe-2 not importable after install"; return 1; }
  fi
}

# ============================================================================
# PER-SNIPPET LOOP
# ============================================================================
for ROW in "${SNIPPETS[@]}"; do
  read -r KEY EP SNIP CONFIG FRAMES <<< "$ROW"
  # ONLY="c1_001" (or "c1_001 f3_007") restricts the batch to those snippets —
  # used for the c1-focused baseline/DINOv3 A/B comparison.
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $KEY "* ]]; then continue; fi
  NAME=$(echo "$KEY" | tr '[:lower:]' '[:upper:]')   # e.g. c1_001 -> C1_001

  echo ""
  echo "############################################################"
  echo "## SNIPPET $KEY  ($EP/$SNIP, $FRAMES frames)"
  echo "############################################################"

  RAW=/content/crcd_raw/${NAME}
  STAGED=/content/sni-slam/data/CRCD/${NAME}
  OUTPUT=/content/sni-slam/output/CRCD/${KEY}/test
  DRIVE_DST=$DRIVE_ROOT/$KEY
  mkdir -p "$DRIVE_DST"

  if done_marker "$DRIVE_DST"; then
    echo "  $KEY already shipped to Drive -- skip"; continue
  fi

  # --------------------------------------------------------------------------
  # PHASE 1 -- stage raw from Drive (tarball-first; per-item cp fallback)
  # --------------------------------------------------------------------------
  phase "1.$KEY" "stage raw $EP/$SNIP from Drive"
  DRIVE_TAR=$CRCD_DRIVE_ROOT/${EP}_${SNIP}_staging.tar
  if [ -f "$RAW/.STAGED" ]; then
    echo "  raw already staged"
  elif [ -f "$DRIVE_TAR" ]; then
    echo "  using tarball: $DRIVE_TAR ($(du -h "$DRIVE_TAR" | cut -f1))"
    mkdir -p "$RAW"
    T0=$(date +%s)
    tar xf "$DRIVE_TAR" -C "$RAW" || { echo "FATAL: tarball extract failed"; exit 3; }
    echo "  extracted in $(( ($(date +%s) - T0) / 60 )) min"
    touch "$RAW/.STAGED"
  else
    DRIVE_SRC=$CRCD_DRIVE_ROOT/$EP/$SNIP
    [ -d "$DRIVE_SRC" ] || { echo "FATAL: missing $DRIVE_SRC"; exit 3; }
    echo "  WARN: no tarball at $DRIVE_TAR; falling back to per-item cp"
    mkdir -p "$RAW"
    for ITEM in rgb rgbright semantic_instance groundtruth.txt intrinsics.yaml; do
      [ -e "$DRIVE_SRC/$ITEM" ] || { echo "  $ITEM missing on Drive, skipping"; continue; }
      [ -e "$RAW/$ITEM" ] || cp -r "$DRIVE_SRC/$ITEM" "$RAW/$ITEM" \
        || { echo "FATAL: cp failed for $ITEM"; exit 3; }
    done
    touch "$RAW/.STAGED"
  fi
  N_RAW=$(ls "$RAW/rgb"/*.png 2>/dev/null | wc -l)
  echo "  raw rgb count: $N_RAW (expected $FRAMES)"
  [ "$N_RAW" -ge "$FRAMES" ] || { echo "FATAL: raw count $N_RAW < $FRAMES"; exit 3; }

  # --------------------------------------------------------------------------
  # PHASE 2 -- preprocess: rectify rgb + sem, write traj.txt (use existing
  # static config from configs/CRCD/<key>.yaml; do NOT --emit_config since
  # we want the manually-tuned bound + DDS-SLAM-parity knobs)
  # --------------------------------------------------------------------------
  phase "2.$KEY" "preprocess (rectify via ECM_STEREO_L2R pickle)"
  if [ -f "$STAGED/.PREPROCESSED" ]; then
    echo "  preprocess already complete"
  else
    rm -rf "$STAGED"
    mkdir -p "$STAGED"
    cd "$REPO_ROOT"
    # --emit_config is a required arg in preprocess_crcd_for_sni.py, but we
    # use the pre-built static configs (configs/CRCD/<key>.yaml from commit
    # 1403de8) instead of the emit_config-derived one — those have the
    # paper-faithful + F1 lr fix knobs.  Pass a throwaway path that gets
    # ignored downstream (run.py reads $CONFIG, not this file).
    python Addons/preprocess/preprocess_crcd_for_sni.py \
      --snippet_dir "$RAW" \
      --calib_pkl   "$CALIB_PKL" \
      --output_dir  "$STAGED" \
      --snippet_id  "$NAME" \
      --emit_config "$STAGED/_emitted_unused.yaml" \
      || { echo "FATAL: preprocess failed for $KEY"; exit 4; }
    touch "$STAGED/.PREPROCESSED"
  fi
  N_RGB=$(ls "$STAGED/rgb"/*.png 2>/dev/null | wc -l)
  N_SEM=$(ls "$STAGED/semantic_class"/*.png 2>/dev/null | wc -l)
  N_TRAJ=$(wc -l < "$STAGED/traj.txt" 2>/dev/null || echo 0)
  echo "  rgb=$N_RGB sem=$N_SEM traj=$N_TRAJ"

  # Emit rectified_calib.txt for Phase 3.6 stereo anchor (mirrors DDS-SLAM)
  if [ ! -f "$STAGED/rectified_calib.txt" ]; then
    python - <<PYEOF
import yaml
with open('$RAW/intrinsics.yaml') as f: intr = yaml.safe_load(f)
cam = intr['camera']; stereo = intr['stereo']
with open('$STAGED/rectified_calib.txt', 'w') as f:
    f.write(f"fx {cam['fx']}\n")
    f.write(f"fy {cam['fy']}\n")
    f.write(f"cx {cam['cx']}\n")
    f.write(f"cy {cam['cy']}\n")
    f.write(f"baseline_m {stereo['baseline_m']}\n")
PYEOF
  fi

  # --------------------------------------------------------------------------
  # PHASE 3 -- MoGe-2 depth generation
  # --------------------------------------------------------------------------
  phase "3.$KEY" "MoGe-2 depth gen ($FRAMES frames at 1280x720)"
  EXPECTED=$FRAMES
  ACTUAL=0
  [ -d "$STAGED/depth" ] && ACTUAL=$(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l)
  # Drive cache pre-check (workflow w7imqnt4t):  rehydrate MoGe + sc_factor
  # from the published cache before regenerating.  PNGs need to be renamed
  # to SNI-SLAM's depth_NNNNNN.png convention (DDS-SLAM uses NNNNNN.png).
  DRIVE_DEPTH_CACHE=$DRIVE_CACHE_ROOT/$EP/snippet_$SID/depth
  if [ ! -f "$STAGED/depth/.DONE" ] && [ -d "$DRIVE_DEPTH_CACHE" ]; then
    CACHE_PNG=$(ls "$DRIVE_DEPTH_CACHE"/*.png 2>/dev/null | wc -l)
    if [ "$CACHE_PNG" -ge "$EXPECTED" ] && [ -f "$DRIVE_DEPTH_CACHE/.sc_factor" ]; then
      echo "  rehydrating MoGe from cache: $DRIVE_DEPTH_CACHE ($CACHE_PNG PNGs)"
      mkdir -p "$STAGED/depth"
      for SRC in "$DRIVE_DEPTH_CACHE"/*.png; do
        FID=$(basename "$SRC" .png)
        DST="$STAGED/depth/depth_${FID}.png"
        [ -f "$DST" ] || cp "$SRC" "$DST"
      done
      cp "$DRIVE_DEPTH_CACHE/.sc_factor" "$STAGED/.sc_factor"
      sync; touch "$STAGED/depth/.DONE"; sync
      ACTUAL=$(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l)
      echo "  rehydrated $ACTUAL renamed PNGs + sc_factor=$(cat "$STAGED/.sc_factor")"
    fi
  fi
  if [ -f "$STAGED/depth/.DONE" ] && [ "$ACTUAL" -ge "$EXPECTED" ]; then
    echo "  depth/ complete ($ACTUAL/$EXPECTED) -- skip MoGe gen"
  else
    ensure_moge || exit 5
    cd "$STAGED"
    mkdir -p _moge_in _moge_npy depth.tmp
    for f in rgb/rgb_*.png; do
      fid=$(basename "$f" | sed 's/rgb_//; s/.png//')
      [ -L "_moge_in/${fid}-left.png" ] || ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"
    done
    echo "  symlinks: $(ls _moge_in/ | wc -l)"
    python /content/sni-slam/Addons/depth/generate_depth_moge.py \
      --rgb _moge_in --out _moge_npy \
      --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 \
      || { echo "FATAL: MoGe gen failed"; exit 5; }
    python - <<'PYEOF'
import numpy as np, cv2, glob, os
n_in = sorted(glob.glob('_moge_npy/*-left_depth.npy'))
written = 0
for p in n_in:
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/depth_{fid}.png'
    if os.path.exists(out): continue
    d = np.load(p).astype(np.float32)
    cv2.imwrite(out, np.clip(d, 0, 65535).astype(np.uint16))
    written += 1
print(f'npy->png wrote {written}')
PYEOF
    PNG=$(ls depth.tmp/*.png 2>/dev/null | wc -l)
    [ "$PNG" -ge "$EXPECTED" ] || { echo "FATAL: depth count $PNG < $EXPECTED"; cd "$REPO_ROOT"; exit 5; }
    rm -rf depth && mv depth.tmp depth
    sync; touch depth/.DONE; sync
    rm -rf _moge_in _moge_npy
    cd "$REPO_ROOT"
  fi
  N_D=$(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l)
  echo "  depth ready: $N_D files (depth_NNNNNN.png convention)"

  # --------------------------------------------------------------------------
  # PHASE 3.6 -- stereo anchor + in-place depth rescale (SNI-SLAM has no
  # sc_factor knob; we rescale on disk to maintain png_depth_scale=10000)
  # --------------------------------------------------------------------------
  phase "3.6.$KEY" "stereo anchor calibration + depth rescale if needed"
  # User 2026-06-05 root cause: prior runs touched .sc_factor_applied as an
  # empty marker, so when sc_factor changed (1.0 fallback -> 0.1337 hardcoded)
  # Phase 3.6 silently skipped the actual rescale -> SLAM ran on uncalibrated
  # MoGe -> sim3_s=0.0009 (1100x too big) instead of expected ~0.08.
  # Fix: .sc_factor_applied now stores the VALUE; we re-rescale on drift.
  SCF_APPLIED_VAL=""
  [ -f "$STAGED/.sc_factor_applied" ] && SCF_APPLIED_VAL=$(cat "$STAGED/.sc_factor_applied" 2>/dev/null)
  SCF_EXPECTED_VAL=""
  [ -f "$STAGED/.sc_factor" ] && SCF_EXPECTED_VAL=$(cat "$STAGED/.sc_factor")
  # Always export SC_FACTOR for downstream Phase 3.7 / 5 / 6.  Use the
  # expected value if present; the if/else below handles application logic.
  SC_FACTOR="${SCF_EXPECTED_VAL:-1.0}"
  if [ -n "$SCF_APPLIED_VAL" ] && [ -n "$SCF_EXPECTED_VAL" ] && \
     [ "$(python -c "print(abs(float('$SCF_APPLIED_VAL') - float('$SCF_EXPECTED_VAL')) < 0.001)")" = "True" ]; then
    echo "  sc_factor already applied (cached: $SCF_APPLIED_VAL)"
  else
    # SNI-SLAM preprocess (preprocess_crcd_for_sni.py) only rectifies LEFT —
    # the rectified RIGHT we'd need for SGBM is not staged, and re-rectifying
    # right here in bash is fragile.  Instead reuse DDS-SLAM's per-snippet
    # sc_factor which was computed on the same data via the proper Phase 1.6
    # SGBM in the DDS-SLAM runbook.  Lookup order:
    #   (1) $STAGED/.sc_factor       — Phase 3 rehydration wrote it
    #   (2) DDS-SLAM local Colab     — same-session DDS-SLAM produced it
    #   (3) Drive published cache    — cross-session published source
    #   (4) Hardcoded known values   — last-resort: values verified from DDS-SLAM
    #       2026-06-04 batch (config.json in each snippet's payload.tgz):
    #         F3_007: 0.133682, C1_001: 0.169897, C2_001: 0.129950, F1_002: 0.143676
    DDS_SCF=/content/DDS-SLAM/data/CRCD/${NAME}/.sc_factor
    DRIVE_SCF=$DRIVE_CACHE_ROOT/$EP/snippet_$SID/depth/.sc_factor
    if [ -f "$STAGED/.sc_factor" ]; then
      SC_FACTOR=$(cat "$STAGED/.sc_factor")
      echo "  reusing sc_factor from Phase 3 rehydration: $SC_FACTOR"
    elif [ -f "$DDS_SCF" ]; then
      SC_FACTOR=$(cat "$DDS_SCF")
      echo "  reusing DDS-SLAM local Colab sc_factor: $SC_FACTOR"
    elif [ -f "$DRIVE_SCF" ]; then
      SC_FACTOR=$(cat "$DRIVE_SCF")
      echo "  reusing Drive-cached sc_factor: $SC_FACTOR"
    else
      case "$KEY" in
        f3_007) SC_FACTOR=0.133682 ;;
        c1_001) SC_FACTOR=0.169897 ;;
        c2_001) SC_FACTOR=0.129950 ;;
        f1_002) SC_FACTOR=0.143676 ;;
        *)      SC_FACTOR=1.0 ;;
      esac
      echo "  no .sc_factor in any cache; using hardcoded DDS-SLAM-computed value: $SC_FACTOR"
      # Persist for resume / downstream Phase 6 etc.
      echo "$SC_FACTOR" > "$STAGED/.sc_factor"
    fi

    APPLY=$(python -c "
import math
s = $SC_FACTOR
print('1' if abs(math.log(s)) > 0.1 else '0')")
    if [ "$APPLY" = "1" ]; then
      echo "  sc_factor $SC_FACTOR is >10% off 1.0 -- rescaling depth PNGs in place"
      python - <<PYEOF
import cv2, numpy as np, glob, os
SC = $SC_FACTOR
files = sorted(glob.glob('$STAGED/depth/depth_*.png'))
n = 0
for p in files:
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * SC
    cv2.imwrite(p, np.clip(d, 0, 65535).astype(np.uint16))
    n += 1
print(f'  rescaled {n} depth PNGs by factor {SC:.4f}')
PYEOF
    else
      echo "  sc_factor $SC_FACTOR within 10% of 1.0 -- no rescale needed"
    fi
    echo "$SC_FACTOR" > "$STAGED/.sc_factor_applied"
  fi

  # ----------------------------------------------------------------------------
  # PHASE 3.7 -- truncation tightening + freeze joint pose opt (render fix)
  # User 2026-06-05 root cause analysis: with depth-only rescale (Phase 3.6),
  # the geometry IS internally consistent:
  #   - rescaled depth 0.10m (true metric)
  #   - GT poses ~0.78m (true metric)
  #   - scene world coord = pose + depth ≈ 0.88m
  #   - bound z [0.68, 0.90] CONTAINS 0.88m ✓
  # So the black renders are NOT from bound mismatch.  Diagnosis: joint pose
  # optimization (joint_opt_cam_lr=0.0001) drifts est poses DURING mapping
  # because trunc=0.06m is 60% of scene depth (0.10m) -> SDF gradient too
  # shallow -> optimizer has slack to move poses without loss penalty.  Result:
  # sim3_s=2.92 (est trajectory 3x compressed) -> est poses end up OUTSIDE
  # bound -> tri-plane queries at render time return zero -> black renders.
  # Surgical fix: shrink truncation by sc_factor + zero out joint_opt_cam_lr.
  # Leave bound, traj.txt, and depth scale untouched.
  # ----------------------------------------------------------------------------
  phase "3.7.$KEY" "truncation tighten + freeze joint pose opt"
  python - <<PYEOF
import os, yaml
SC = $SC_FACTOR
def load_chain(p):
    cfg = yaml.safe_load(open(p))
    inh = cfg.get('inherit_from')
    base = load_chain(inh) if inh else {}
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
            else: a[k] = v
        return a
    return merge(dict(base), cfg)

orig = load_chain('$CONFIG')
trunc_old = orig['model']['truncation']
jopt_old  = orig['mapping'].get('joint_opt_cam_lr', 0.0)

# Truncation: scale by sc_factor.  Empirically sharper SDF gradient prevents
# the optimizer's pose drift.  For SC=0.17, trunc 0.06 -> 0.010 (10% of scene).
trunc_new = trunc_old * SC

# Freeze joint pose opt -- decouples mapper from tracker pose, prevents the
# compression-during-mapping that produces sim3_s=2.92 and pushes est poses
# outside the bound at render time.
jopt_new = 0.0

override = {
    'inherit_from': os.path.abspath('$CONFIG'),
    'mapping': {
        'joint_opt_cam_lr': jopt_new,
    },
    'model': {
        'truncation': trunc_new,
    },
}
with open('$STAGED/scaled_config.yaml', 'w') as f:
    yaml.safe_dump(override, f, default_flow_style=None, sort_keys=False)
print(f'  truncation: {trunc_old:.4f} -> {trunc_new:.5f} (× sc_factor {SC:.4f})')
print(f'  joint_opt_cam_lr: {jopt_old} -> {jopt_new} (frozen)')
print(f'  bound + traj.txt UNCHANGED (geometry already consistent post Phase 3.6)')
PYEOF

  # Phase 5+6 use the scaled config so all downstream eval reads the same trunc
  if [ -f "$STAGED/scaled_config.yaml" ]; then
    CONFIG="$STAGED/scaled_config.yaml"
    echo "  Phase 5+6 will use: $CONFIG"
  fi

  # --------------------------------------------------------------------------
  # PHASE 4 -- motion profile sanity (non-blocking)
  # --------------------------------------------------------------------------
  phase "4.$KEY" "GT motion profile sanity"
  python - <<PYEOF
import numpy as np
rows = []
with open('$STAGED/traj.txt') as f:
    for line in f:
        v = line.strip().split()
        if not v: continue
        try:
            arr = np.array(list(map(float, v)))
            if arr.size >= 12:
                # First 12 numbers are 3x4 (row-major flatten) c2w
                t = arr[:12].reshape(3, 4)[:3, 3]
                rows.append(t)
        except Exception:
            pass
xyz = np.array(rows)
if len(xyz) < 2:
    print('  WARN: too few traj rows')
else:
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
    print(f'  GT poses    : {len(xyz)}')
    print(f'  extent (mm) : x={(xyz[:,0].max()-xyz[:,0].min())*1000:.2f}  '
          f'y={(xyz[:,1].max()-xyz[:,1].min())*1000:.2f}  '
          f'z={(xyz[:,2].max()-xyz[:,2].min())*1000:.2f}')
    print(f'  total path  : {d.sum():.2f} mm')
    print(f'  per-frame   : median={np.median(d):.4f}  max={d.max():.4f} mm/f')
    active = d > 0.001
    print(f'  active frac : {100*active.mean():.1f}%')
    if active.sum() > 0:
        am = np.median(d[active])
        print(f'  active median: {am:.4f} mm/f  ({am/1.0:.2f}x noise floor)')
        if am < 0.5:
            print(f'  SENTINEL: active median < 0.5 mm/f -- sub-SNR')
PYEOF

  # --------------------------------------------------------------------------
  # PHASE 5 -- SNI-SLAM run
  # --------------------------------------------------------------------------
  phase "5.$KEY" "SNI-SLAM run.py --config $CONFIG"
  if [ -f "$OUTPUT/ckpts" ] && ls "$OUTPUT/ckpts/"*.tar >/dev/null 2>&1; then
    echo "  ckpt already present -- skipping SLAM"
  else
    T0=$(date +%s)
    cd "$REPO_ROOT"
    if ! python run.py "$CONFIG"; then
      echo "  $KEY SLAM crashed; continuing to next snippet"
      echo "FAILED_SLAM_$KEY" >> "$DRIVE_ROOT/_failures.log"
      continue
    fi
    echo "  $KEY SLAM elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  fi

  # --------------------------------------------------------------------------
  # PHASE 6 -- eval: extended trajectory metrics + render PSNR/SSIM/LPIPS + Depth L1
  # --------------------------------------------------------------------------
  phase "6.$KEY" "eval (extended ATE + render + Depth L1)"

  # Extended trajectory metrics (Sim3 + est_path/GT_path + per-axis Pearson)
  python src/tools/eval_traj_extended.py "$CONFIG" \
    --csv "$DRIVE_ROOT/_traj_summary.csv" \
    --name "$KEY" 2>&1 | tee "$DRIVE_DST/extended_metrics.txt" || true

  # Post-hoc per-frame render (DDS-SLAM parity).
  # SNI-SLAM's Frame_Visualizer (src/utils/Frame_Visualizer.py:55) saves only
  # sparse composite matplotlib figures at vis_freq intervals — NOT clean
  # per-frame RGB+depth like DDS-SLAM does at ddsslam.py:806-828.
  # Without this post-hoc render, the runbook below would silently find 0
  # rendered images and skip render eval entirely (validated by user on
  # DDS-SLAM CRCD batch where the same issue led to render eval no-op).
  # Renders go to $OUTPUT/rendered/*.jpg + $OUTPUT/rendered/depth/*.png
  # — DDS-SLAM-compatible layout, so generate_video.py just works.
  if [ ! -d "$OUTPUT/rendered" ] || [ -z "$(ls $OUTPUT/rendered/*.jpg 2>/dev/null | head -1)" ]; then
    echo "  Phase 6a: rendering all frames from final ckpt..."
    python Addons/viz/render_all_frames_sni.py "$CONFIG" --skip 1 2>&1 \
      | tee "$DRIVE_DST/render_all.log" \
      || echo "  WARN: render_all_frames_sni.py failed (eval will skip render metrics)"
  fi

  # Render eval (PSNR/SSIM/LPIPS) — now points at the post-hoc rendered dir
  if [ -d "$OUTPUT/rendered" ] && ls "$OUTPUT/rendered"/*.jpg >/dev/null 2>&1; then
    python Addons/eval/eval_rendering.py \
      --gt_dir "$STAGED/rgb" \
      --render_dir "$OUTPUT/rendered" \
      --name "$KEY" \
      --output_csv "$DRIVE_DST/render_eval.csv" \
      --summary_csv "$DRIVE_ROOT/_render_summary.csv" \
      --sequence "CRCD (${NAME})" 2>&1 | tee "$DRIVE_DST/render_eval.txt" || \
      echo "  render eval failed"
  else
    echo "  no rendered/*.jpg present — skip render eval"
  fi

  # Depth L1 (recon metric) — requires GT mesh (not available for CRCD;
  # SNI-SLAM Depth L1 protocol is mesh-vs-mesh).  Skip with note.
  echo "  Depth L1: skipped (no GT mesh for CRCD; would require generating one)" \
    | tee -a "$DRIVE_DST/extended_metrics.txt"

  # --------------------------------------------------------------------------
  # PHASE 7 -- ship to Drive
  # --------------------------------------------------------------------------
  phase "7.$KEY" "ship to Drive"
  if [ ! -f "$DRIVE_DST/payload.tgz" ] && [ -d "$OUTPUT" ]; then
    tar czf "$DRIVE_DST/payload.tgz.partial" \
      -C "$OUTPUT" . 2>/dev/null
    mv "$DRIVE_DST/payload.tgz.partial" "$DRIVE_DST/payload.tgz"
  fi

  # Also extract est_c2w_data.txt from the ckpt (DDS-SLAM-style flat 12-float
  # per row) so generate_video.py can build the 6-panel video without needing
  # to re-load the ckpt.  Use --skip_existing so this is idempotent.
  if [ -f "$OUTPUT/ckpts" ] || ls "$OUTPUT/ckpts/"*.tar >/dev/null 2>&1; then
    CKPT=$(ls -t "$OUTPUT/ckpts/"*.tar 2>/dev/null | head -1)
    if [ -n "$CKPT" ] && [ ! -f "$OUTPUT/est_c2w_data.txt" ]; then
      python - <<PYEOF
import torch
ckpt = torch.load('$CKPT', map_location='cpu')
est = ckpt['estimate_c2w_list'].cpu().numpy()
with open('$OUTPUT/est_c2w_data.txt', 'w') as f:
    for i in range(est.shape[0]):
        flat = est[i, :3, :4].flatten()
        f.write(' '.join(f'{v:.10f}' for v in flat) + '\n')
print(f'  wrote {est.shape[0]} poses to est_c2w_data.txt')
PYEOF
    fi
  fi

  # Ship the inputs needed for 6-panel video generation (rectified RGB input,
  # MoGe-rescaled depth input, traj.txt).  Tiny relative to payload.tgz and
  # makes the video pipeline reproducible from Drive alone.
  if [ -d "$STAGED" ] && [ ! -f "$DRIVE_DST/inputs.tgz" ]; then
    tar czf "$DRIVE_DST/inputs.tgz.partial" \
      -C "$STAGED" \
      rgb depth semantic_class traj.txt scaled_config.yaml .sc_factor 2>/dev/null
    mv "$DRIVE_DST/inputs.tgz.partial" "$DRIVE_DST/inputs.tgz" 2>/dev/null \
      || echo "  WARN: inputs.tgz ship failed (some files missing)"
  fi

  mark_done "$DRIVE_DST"
  echo "## $KEY complete"
done

# ============================================================================
# PHASE 8 -- combined 4-snippet summary
# ============================================================================
phase 8 "combined summary"
COMBINED=$DRIVE_ROOT/COMBINED_SUMMARY.txt
{
  echo "=== SNI-SLAM CRCD 4-snippet preliminary ($(date -Iseconds)) ==="
  echo ""
  if [ -f "$DRIVE_ROOT/_traj_summary.csv" ]; then
    echo "--- trajectory metrics (csv) ---"
    cat "$DRIVE_ROOT/_traj_summary.csv"
  fi
  echo ""
  if [ -f "$DRIVE_ROOT/_render_summary.csv" ]; then
    echo "--- render quality (csv) ---"
    cat "$DRIVE_ROOT/_render_summary.csv"
  fi
  echo ""
  echo "Per-snippet payloads:"
  for ROW in "${SNIPPETS[@]}"; do
    read -r KEY _ _ _ _ <<< "$ROW"
    echo "  $DRIVE_ROOT/$KEY/payload.tgz"
  done
} | tee "$COMBINED"
echo ""
echo "=== done $(date -Iseconds) ==="
echo "Log:     $LOG"
echo "Summary: $COMBINED"
