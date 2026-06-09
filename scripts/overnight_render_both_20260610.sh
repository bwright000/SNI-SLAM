#!/usr/bin/env bash
# =====================================================================
# overnight_render_both_20260610.sh — SNI-SLAM overnight that produces
# RENDERABLE results for BOTH Replica and CRCD, in one run.
# =====================================================================
# Replica  -> run SLAM, ship mesh/ckpts/poses (for the Replica Visualiser,
#             run LOCALLY) + ATE/recon eval + 2D frame renders.
# CRCD     -> run SLAM (baseline AND the pixel-density A/B 'hipix' config on the
#             same staged data) + render frames + PSNR/SSIM/LPIPS + Sim3.
#
# The A/B tests the stage-1 audit finding (CRCD samples ~3x fewer pixels per unit
# area than Replica): baseline 2048/1024 vs hipix 6000/3000.
#
# DESIGN: the heavy, proven steps are delegated to the existing per-scene/snippet
# runbooks (run_replica_subset.sh / run_crcd_overnight.sh) which are idempotent +
# sentinel-gated. The render + eval layer here is BEST-EFFORT (|| warn) so a
# render/pairing hiccup never aborts the night — the SLAM outputs always ship.
#
# REQUIRES A100 (both sub-runbooks hard-gate it; Replica OOM-deadlocks on T4 at
# ~frame 2000 per project_sni_slam_replica_stall_20260530).
#
# PREREQS (fresh Colab A100, each line a separate cell/command):
#   from google.colab import drive; drive.mount('/content/drive')        # (cell)
#   # one-time env + data stage (vMAP Replica is 44.8 GB; caches to Drive):
#   cd /content && (git clone https://github.com/bwright000/SNI-SLAM.git sni-slam || true) \
#     && cd sni-slam && git checkout sni-pixel-density && git pull
#   bash scripts/colab_setup_sni.sh
#   conda activate sni
#   nohup bash scripts/overnight_render_both_20260610.sh >/content/sni_both.out 2>&1 &
#   tail -f /content/sni_both.out
#
# SCOPE OVERRIDES (env at launch):
#   REPLICA_SCENES="room0"            CRCD_SNIPPETS="c1_001"
#   REPLICA_SCENES=""  (skip Replica) CRCD_SNIPPETS="" (skip CRCD)
#   RENDER_SKIP=10  (render every Nth Replica frame; CRCD renders all)
# =====================================================================
set -u
set -o pipefail
DATE=20260610
REPO=${REPO:-/content/sni-slam}
REPLICA_SCENES=${REPLICA_SCENES:-"room0 office0"}
CRCD_SNIPPETS=${CRCD_SNIPPETS:-"c1_001 c2_001"}
RENDER_SKIP=${RENDER_SKIP:-10}
DRIVE=/content/drive/MyDrive/Outputs/sni_render_both_${DATE}
LOG=$DRIVE/runbook.log
mkdir -p "$DRIVE"
cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# env (assumes colab_setup_sni.sh already built the 'sni' conda env + staged data)
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate sni 2>/dev/null || say "WARN: 'conda activate sni' failed — env may be missing"
python -c "import torch,trimesh;print('env OK',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { say "FATAL: sni env import failed (run colab_setup_sni.sh)"; exit 1; }

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
say "=== SNI render-both overnight $(date -Iseconds) === GPU $GPU | HEAD $(git rev-parse --short HEAD 2>/dev/null)"
[[ "$GPU" =~ A100 ]] || say "WARN: GPU is '$GPU', not A100. Replica will OOM near frame ~2000; CRCD snippets (short) may still run."

last_ckpt(){ ls -t "$1"/ckpts/*.tar 2>/dev/null | head -1; }   # SNI saves NNNNN.tar

# ---------------------------------------------------------------------
# PHASE A — Replica: run + ship (for the Visualiser) + 2D renders
# ---------------------------------------------------------------------
if [ -n "$REPLICA_SCENES" ]; then
  say "PHASE A: Replica scenes = $REPLICA_SCENES"
  SCENES="$REPLICA_SCENES" bash scripts/run_replica_subset.sh 2>&1 | tee -a "$LOG" || say "  WARN: replica subset returned nonzero"
  for s in $REPLICA_SCENES; do
    OUT=$REPO/output/Replica/$s/test
    CK=$(last_ckpt "$OUT")
    RDST=$DRIVE/replica/$s
    mkdir -p "$RDST"
    if [ -n "$CK" ]; then
      say "  Replica render $s (skip $RENDER_SKIP) from $(basename "$CK")"
      python -W ignore Addons/viz/render_all_frames_sni.py configs/Replica/$s.yaml \
        --checkpoint "$CK" --output_dir "$OUT/rendered" --skip "$RENDER_SKIP" --png 2>&1 | tee -a "$LOG" \
        || say "  WARN: replica render $s failed"
      cp -rn "$OUT/rendered" "$RDST/rendered" 2>/dev/null || true
    else
      say "  WARN: no Replica ckpt for $s (run incomplete — T4 OOM?); skip render"
    fi
    say "  Replica $s -> renders in $RDST/rendered ; mesh/ckpts/poses shipped by run_replica_subset to MyDrive/Outputs/sni_slam_replica/$s (view 3D LOCALLY with visualizer.py)"
  done
else say "PHASE A: Replica skipped (REPLICA_SCENES empty)"; fi

# ---------------------------------------------------------------------
# PHASE B — CRCD: baseline + hipix A/B, run + render + eval
# ---------------------------------------------------------------------
crcd_id(){ case "$1" in c1_001)echo C1_001;; c2_001)echo C2_001;; f1_002)echo F1_002;; f3_007)echo F3_007;; *)echo "";; esac; }

render_eval_crcd(){   # $1 config-key  $2 output-dir  $3 staged-rgb  $4 seq-id  $5 dst
  local cfg="configs/CRCD/$1.yaml" out="$2" gt="$3" seq="$4" dst="$5"
  local ck; ck=$(last_ckpt "$out")
  [ -n "$ck" ] || { say "    WARN: no ckpt in $out — skip render/eval"; return 0; }
  mkdir -p "$dst"
  python -W ignore Addons/viz/render_all_frames_sni.py "$cfg" --checkpoint "$ck" \
    --output_dir "$out/rendered" --png 2>&1 | tee -a "$LOG" || say "    WARN: render $1 failed"
  python Addons/eval/eval_rendering.py --gt_dir "$gt" --render_dir "$out/rendered" \
    --name "$1" --output_csv "$dst/render_eval.csv" --summary_csv "$DRIVE/crcd/_render_summary.csv" \
    --sequence "CRCD ($seq)" 2>&1 | tee "$dst/render_eval.txt" || say "    WARN: eval_rendering $1 failed"
  cp -rn "$out/rendered" "$dst/rendered" 2>/dev/null || true
  cp -n "$out"/mesh/*culled*.ply "$dst/" 2>/dev/null || true
}

if [ -n "$CRCD_SNIPPETS" ]; then
  say "PHASE B: CRCD snippets = $CRCD_SNIPPETS  (baseline + hipix A/B)"
  mkdir -p "$DRIVE/crcd"
  for snip in $CRCD_SNIPPETS; do
    ID=$(crcd_id "$snip"); [ -n "$ID" ] || { say "  WARN: unknown snippet $snip"; continue; }
    STAGED=$REPO/data/CRCD/$ID; GT=$STAGED/rgb
    # --- baseline: stage + preprocess + MoGe + run + Sim3 + ship (proven runbook) ---
    say "  CRCD $snip BASELINE"
    bash scripts/run_crcd_overnight.sh "$snip" 2>&1 | tee -a "$LOG" || say "  WARN: $snip baseline runbook nonzero"
    render_eval_crcd "$snip" "$REPO/output/CRCD/$ID/test" "$GT" "$ID" "$DRIVE/crcd/${ID}_baseline"
    # --- hipix A/B: data already staged/preprocessed/MoGe'd above -> just run + render + eval ---
    HCFG=configs/CRCD/${snip}_hipix_${DATE}.yaml
    if [ -f "$HCFG" ] && [ -d "$STAGED" ]; then
      say "  CRCD $snip HIPIX (6000/3000)"
      python -W ignore run.py "$HCFG" 2>&1 | tee -a "$LOG" || say "  WARN: $snip hipix run failed"
      render_eval_crcd "${snip}_hipix_${DATE}" "$REPO/output/CRCD/${snip}_hipix_${DATE}/test" "$GT" "$ID" "$DRIVE/crcd/${ID}_hipix"
    else say "  WARN: hipix config or staged data missing for $snip — skip A/B"; fi
  done
else say "PHASE B: CRCD skipped (CRCD_SNIPPETS empty)"; fi

# ---------------------------------------------------------------------
# PHASE C — summary
# ---------------------------------------------------------------------
say "=== SNI render-both done $(date -Iseconds) ==="
say "Replica : $DRIVE/replica/<scene>/rendered  + output shipped by run_replica_subset to MyDrive/Outputs/sni_slam_replica/<scene>"
say "          VIEW 3D LOCALLY: download that scene dir, then  python visualizer.py configs/Replica/<scene>.yaml --output <dir> --save_rendering"
say "CRCD    : $DRIVE/crcd/{<ID>_baseline,<ID>_hipix}/{rendered,render_eval.csv}  + Sim3 via run_crcd_overnight"
say "A/B     : compare _baseline vs _hipix render_eval.csv  -> does higher pixel density lift PSNR/SSIM/LPIPS?"
if [ -f "$DRIVE/crcd/_render_summary.csv" ]; then say "summary:"; cat "$DRIVE/crcd/_render_summary.csv" | tee -a "$LOG"; fi
