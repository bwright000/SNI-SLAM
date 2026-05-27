#!/bin/bash
# ============================================================
# run_replica_subset.sh — fire SNI-SLAM on a subset of Replica scenes,
# save each completed run to Drive, then evaluate ATE + reconstruction.
#
# Idempotent: if a scene's output is already on Drive, the SLAM step is
# skipped. If the eval results for a scene are already present in
# $DRIVE_RESULTS, eval is skipped. Lets you re-run safely after a session
# death.
#
# Per-scene runtime on T4: ~1.5-2.5 hr SLAM + a couple of minutes eval.
# With 4 scenes the full sweep is ~6-10 hr.
#
# Usage:
#   conda activate sni
#   cd sni-slam
#   bash scripts/run_replica_subset.sh
#
# Env overrides:
#   SCENES="room0 room1"   bash scripts/run_replica_subset.sh    # custom subset
#   SKIP_EVAL=1            bash scripts/run_replica_subset.sh    # SLAM only
# ============================================================
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Default subset chosen for spread across room/office families.
# Override with SCENES env var if the Google Drive subset has different
# scenes available (the README says "subset" — confirm via ls before running).
SCENES_DEFAULT="room0 room1 office0 office1"
SCENES=${SCENES:-$SCENES_DEFAULT}

DRIVE_OUT=/content/drive/MyDrive/Outputs/sni_slam_replica
DRIVE_RESULTS=$DRIVE_OUT/_eval
MASTER_LOG=$REPO_ROOT/output/run_replica_subset.log
SKIP_EVAL=${SKIP_EVAL:-0}

mkdir -p "$DRIVE_OUT" "$DRIVE_RESULTS"
mkdir -p "$(dirname "$MASTER_LOG")"
touch "$MASTER_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

# Pre-flight: verify the scenes we want actually exist in data/replica/
log "=== Replica subset sweep starting ==="
log "Scenes requested:  $SCENES"
log "Drive output root: $DRIVE_OUT"
ACTUAL_SCENES=""
for s in $SCENES; do
  # Configs use 'room0' / 'office0' as keys but the data folder is 'room_0' / 'office_0'
  data_name="${s/room/room_}"
  data_name="${data_name/office/office_}"
  if [ -d "data/replica/$data_name" ]; then
    ACTUAL_SCENES="$ACTUAL_SCENES $s"
  else
    log "  WARN  $s skipped — data/replica/$data_name not present"
  fi
done
ACTUAL_SCENES=$(echo $ACTUAL_SCENES | xargs)   # trim
if [ -z "$ACTUAL_SCENES" ]; then
  log "FATAL: no scenes available in data/replica/. Run colab_setup_sni.sh first."
  exit 1
fi
log "Scenes to run:     $ACTUAL_SCENES"
log ""

# ---------------------------------------------------------------------
# SLAM loop
# ---------------------------------------------------------------------
for s in $ACTUAL_SCENES; do
  CFG="configs/Replica/${s}.yaml"
  if [ ! -f "$CFG" ]; then
    log "SKIP  $s (no config at $CFG)"
    continue
  fi

  DRIVE_SCENE_OUT=$DRIVE_OUT/$s
  if [ -d "$DRIVE_SCENE_OUT" ] && [ "$(ls -A "$DRIVE_SCENE_OUT" 2>/dev/null)" ]; then
    log "SKIP  $s SLAM (already on Drive at $DRIVE_SCENE_OUT)"
    # Make sure local output is in place for eval
    if [ ! -d "output/Replica/$s" ]; then
      mkdir -p "output/Replica/$s"
      cp -r "$DRIVE_SCENE_OUT/"* "output/Replica/$s/" 2>/dev/null || true
    fi
  else
    log "START $s"
    t0=$(date +%s)
    python -W ignore run.py "$CFG" >> "$MASTER_LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    h=$(( elapsed / 3600 ))
    m=$(( (elapsed % 3600) / 60 ))
    if [ $rc -eq 0 ]; then
      log "DONE  $s SLAM (${h}h${m}m) — copying to Drive..."
      mkdir -p "$DRIVE_SCENE_OUT"
      cp -r "output/Replica/$s/"* "$DRIVE_SCENE_OUT/" 2>>"$MASTER_LOG"
      log "SAVED $s -> $DRIVE_SCENE_OUT"
    else
      log "FAIL  $s SLAM (rc=$rc, ${h}h${m}m) — continuing"
      continue
    fi
  fi

  # -----------------------------------------------------------------
  # Per-scene evaluation
  # -----------------------------------------------------------------
  if [ "$SKIP_EVAL" = "1" ]; then
    log "  eval skipped (SKIP_EVAL=1)"
    continue
  fi

  EVAL_OUT=$DRIVE_RESULTS/${s}_metrics.txt
  if [ -f "$EVAL_OUT" ]; then
    log "  eval already present at $EVAL_OUT — skipping"
    continue
  fi

  log "EVAL  $s — ATE + reconstruction"
  : > "$EVAL_OUT"
  echo "# $s — eval generated $(date)" >> "$EVAL_OUT"
  echo "" >> "$EVAL_OUT"

  # ATE
  echo "## ATE (eval_ate.py)" >> "$EVAL_OUT"
  python src/tools/eval_ate.py "$CFG" >> "$EVAL_OUT" 2>&1 || \
    echo "(eval_ate.py exited with error)" >> "$EVAL_OUT"
  echo "" >> "$EVAL_OUT"

  # Reconstruction (depends on GT mesh being on disk)
  echo "## Reconstruction (eval_recon.py)" >> "$EVAL_OUT"
  python src/tools/eval_recon.py "$CFG" >> "$EVAL_OUT" 2>&1 || \
    echo "(eval_recon.py exited with error — GT mesh may be missing)" >> "$EVAL_OUT"

  log "  eval written to $EVAL_OUT"
done

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
log ""
log "=== sweep complete ==="
log "Per-scene Drive outputs:"
ls -d "$DRIVE_OUT"/*/ 2>/dev/null | tee -a "$MASTER_LOG"
log ""
log "Per-scene eval files:"
ls "$DRIVE_RESULTS" 2>/dev/null | tee -a "$MASTER_LOG"
log ""
log "Final mesh per scene (paths):"
find "$DRIVE_OUT" -name "final_mesh_eval_rec_culled.ply" 2>/dev/null | tee -a "$MASTER_LOG"
cp "$MASTER_LOG" "$DRIVE_RESULTS/run_replica_subset.log" 2>/dev/null
