# SNI-SLAM — Overnight Run Handover (2026-06-03)

Operator: human or a fresh Claude session driving a Colab tunnel.
Two goals for the night, on **ONE A100** (see hard requirement below):

- **Goal A** — get *better* SNI-SLAM Replica numbers (close the gap to paper).
- **Goal B** — get SNI-SLAM *running on CRCD* (first attempt; CRCD specifics are the operator's domain).

Assumed budget: ~10–12 h on one A100. Split ≈ 6 h Goal A + ≈ 4–5 h Goal B. Adjust to actual hours.

---

## 0. HARD requirements & current state (read first)

- **GPU: A100 high-RAM ONLY.** SNI-SLAM's mapper grows ~28 MB/frame; on Colab T4 high-RAM (51.5 GB cgroup) it is OOM-killed around frame ~1980 and the parent then deadlocks for hours. The 8-scene sweep that produced our numbers ran on A100 (~83 GB). Do **not** attempt the full Replica sequences on T4.
- **Repo**: `bwright000/sni-slam` (fork of `irmvlab/sni-slam`). Our config tree is byte-identical to upstream **except two path lines** — we introduced **no hyperparameter changes** (verified via `git diff upstream/main`). Env = authors' exact `environment.yaml` (Py3.7.11 / PyTorch1.11+cu113 / pytorch3d 0.7.1).
- **What already ran** (8-scene Replica, single seed each, A100): all 8 scenes completed, semantics validated. See the result table in §3. Headline: ATE Mean **0.825** cm vs paper **0.397** (2.08×); Comp.Ratio **90.07** vs **96.62**.
- **Data source**: vMAP `vmap.zip` (all 8 scenes, label-compatible) + authors' `seg/` weights. Staged + cached on Drive by `scripts/colab_setup_sni.sh`.
- **The one config-vs-paper discrepancy**: upstream `replica.yaml` sets `tracking.lr_T: 0.002`, but the paper text says camera-pose LR **0.001** for Replica (`lr_R` already 0.001). This is upstream's own; it is a Goal-A experiment (§3, A1).

---

## 1. Launch from a fresh Colab tunnel

```bash
# 1) Mount Drive first (in a Colab NOTEBOOK cell, not the terminal):
#    from google.colab import drive; drive.mount('/content/drive')
# 2) Then in the tunnel terminal — one-liner (self-clones, builds env, stages data):
curl -fsSL https://raw.githubusercontent.com/bwright000/sni-slam/main/scripts/colab_setup_sni.sh | bash
```

Warm cache (env tar + extracted data on Drive) → ~5–10 min. Cold (no cache) → env build ~25–35 min + vMAP 44.8 GB download/extract ~30–60 min. Setup is **idempotent** and leaves the env + all 8 scenes ready.

Then activate and sanity-check:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate sni
cd /content/sni-slam
python -c "import torch; print('cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # must say A100
```

---

## 2. The runner you'll use (resumable, ships to Drive)

`scripts/run_replica_subset.sh` runs SLAM + eval per scene. It:
- skips scenes already complete on Drive (gated on the culled final mesh, not rc==0),
- ships each finished scene to `MyDrive/Outputs/sni_slam_replica/<scene>/`,
- writes ATE + reconstruction metrics to `MyDrive/Outputs/sni_slam_replica/_eval/<scene>_metrics.txt`,
- is safe to relaunch after a session death.

```bash
# Full 8-scene sweep:
SCENES='room0 room1 room2 office0 office1 office2 office3 office4' \
  bash scripts/run_replica_subset.sh
# Subset / single scene:
SCENES='room0' bash scripts/run_replica_subset.sh
```

Per-scene SLAM time on A100 ≈ 40–50 min. Full 8 ≈ 5–6 h.

---

## 3. GOAL A — better Replica results

### Where we stand vs paper (verified per-scene, paper = suppl Table 9)

| Scene | ATE RMSE o/p | ATE Mean o/p | Comp.Ratio o/p | Paper DepthL1 |
|---|---|---|---|---|
| office0 | 0.780 / 0.33 | 0.633 / 0.28 | 95.46 / 98.70 | 0.55 |
| office1 | 0.624 / 0.41 | 0.513 / 0.35 | 92.21 / 97.70 | 0.97 |
| office2 | 0.906 / 0.32 | 0.681 / 0.28 | 88.03 / 95.20 | 0.89 |
| office3 | 1.600 / 0.62 | 1.029 / 0.56 | 87.93 / 95.40 | 0.75 |
| office4 | 1.253 / 0.47 | 0.839 / 0.40 | 86.67 / 94.40 | 0.97 |
| room0 | 1.816 / 0.50 | 1.347 / 0.43 | 90.21 / 97.80 | 0.55 |
| room1 | 0.995 / 0.55 | 0.756 / 0.48 | 91.03 / 96.74 | 0.58 |
| room2 | 2.505 / 0.45 | 0.801 / 0.38 | 88.99 / 97.05 | 0.87 |
| **avg** | **1.310 / 0.456** | **0.825 / 0.395** | **90.07 / 96.62** | **0.766** |

### Diagnosis (what "better" should attack)
- **The gap is mostly systematic + single-seed variance, not a code/config bug** (config is the authors').
- **room0** is the only genuinely hard scene (worst on Mean 3.13×; median already > paper RMSE).
- **room2's 2.505 RMSE is a single 53.4 cm frame** (median 0.558). Its Mean ratio is only 2.11×. Re-rolling it (variance) or outlier-filtering kills most of that.
- **Comp.Ratio is uniformly −3 to −8 pts low everywhere** → systematic (eval-protocol / data-fidelity / single-seed), not scene-specific.
- The paper averages **5 runs/scene**; we ran **1**. That alone explains a meaningful slice of the gap.

### Prioritized levers (pick by available hours)

**A1 — `lr_T` A/B on room0 (~1.5 h, do this first; it's a gate).**
Tests the config-vs-paper translation-LR discrepancy on the hardest scene.
```bash
cat > configs/Replica/room0_lrT001.yaml <<'YAML'
inherit_from: configs/Replica/room0.yaml
tracking:
  lr_T: 0.001          # paper-text value (upstream replica.yaml uses 0.002)
data:
  output: output/Replica/room0_lrT001/test
YAML
python -W ignore run.py configs/Replica/room0_lrT001.yaml
python src/tools/eval_ate.py configs/Replica/room0_lrT001.yaml
# Compare RMSE/Mean to room0 baseline (1.816 / 1.347).
```
**Gate:** if lr_T=0.001 clearly beats 0.002 on room0 → adopt it for the sweep (A2a). If not → keep 0.002, go to A2b.

**A2a — full sweep with the winning lr_T (~5–6 h).**
Set `tracking.lr_T: 0.001` in `configs/Replica/replica.yaml` (affects all scenes) then:
```bash
SCENES='room0 room1 room2 office0 office1 office2 office3 office4' \
  bash scripts/run_replica_subset.sh
```

**A2b — multi-seed the hardest scenes (~4–5 h), report best + mean.**
SNI-SLAM is nondeterministic across reruns (CUDA atomics + multiprocessing), so a plain rerun IS a new seed — no code change needed. Target the variance-dominated scenes:
```bash
for s in room0 room2 office3 office4; do
  for k in 2 3; do
    cat > configs/Replica/${s}_seed${k}.yaml <<YAML
inherit_from: configs/Replica/${s}.yaml
data:
  output: output/Replica/${s}_seed${k}/test
YAML
    python -W ignore run.py configs/Replica/${s}_seed${k}.yaml
    python src/tools/eval_ate.py configs/Replica/${s}_seed${k}.yaml
  done
done
```
Then for each scene report **mean over seeds** (matches the paper's protocol) and **best** (room2's outlier frame won't recur every seed). This is the most honest "better" because it's the paper's own methodology.

**A3 — sanity check that won't cost a run:** confirm the run used **mesh eval resolution 0.01** (paper's), not a memory-forced downgrade. Check `meshing.resolution` is 0.01 in the effective config (it is in `replica.yaml`) and that no OOM forced a coarser mesh in the log. A coarse eval mesh alone depresses Comp.Ratio.

**A4 — (optional, completes the table, not a "better number" lever):** capture **Depth L1**. Our `eval_recon.py` emits only Acc/Comp/Comp.Ratio. Paper protocol (suppl §2.3): average abs error over **1000 randomly sampled views**, rendered depth vs GT mesh; culling follows Co-SLAM. Needs a small eval script (port from `neural_slam_eval`, the repo the README cites). Per-scene targets are in the table above (0.55–0.97). A big overshoot ⇒ depth-scale/data-fidelity problem. Treat as follow-up if time is short.

### Recommended Goal-A sequence for the night
A1 (1.5 h) → gate → **A2a if lr_T wins, else A2b** (4–5 h) → leave A4 as morning follow-up. Collect all `_eval/*.txt` + the new sweep outputs on Drive.

---

## 4. GOAL B — SNI-SLAM on CRCD (first run)

You own the CRCD data details. This section is the **SNI-SLAM-side contract** + decisions, so the integration is correct rather than guesswork.

### 4.1 What SNI-SLAM requires per sequence
A dataset folder the loader can read, providing:
- **RGB** + **depth** (depth as uint16 PNG; you must set `cam.png_depth_scale` to match your encoding),
- **camera intrinsics** (`fx, fy, cx, cy`, `H, W`) for the CRCD camera,
- **poses / trajectory** (for ATE; if no GT poses, tracking still runs but ATE is meaningless),
- a scene **`bound`** + **`marching_cubes_bound`** (metric extent of the scene; endoscopic scenes are tiny — e.g. ~0.05–0.5 m — so bounds must be small, unlike Replica's metre-scale),
- **semantic masks** `semantic_class/…` (or TUM `semantic/<ts>.png`) — see the head constraint below.

### 4.2 Loader choice (one of)
- **Reuse `tumrgbd`** (least code): `datasets.py:TUM_RGBD` already reads `rgb.txt`, `depth.txt`, `groundtruth.txt`, and **semantics from `semantic/<timestamp>.png`**. If your CRCD prep (the DDS-SLAM CRCD pipeline) can emit TUM layout + a `semantic/` folder, set `dataset: 'tumrgbd'` and you're done — no code change.
- **Add a `crcd` loader**: copy the `Replica` class, point the globs at your CRCD layout, register `"crcd": CRCD` in `dataset_dict`. Use if your data is already in a Replica-like `rgb/ depth/ semantic_class/ traj.txt` layout.

### 4.3 THE load-bearing constraint — the semantic head (decide before running)
`ModelManager.__init__` **always** builds the DINOv2 head and runs `load_state_dict(dinov2_replica.pth)` with `num_cls = model.cnn.n_classes` — **even if `use_gt_semantic: True`** (`model_manager.py:26,42-43`). Consequences:
- `dinov2_replica.pth` is a **52-class** Replica head. If you set `n_classes ≠ 52`, the `load_state_dict` **will mismatch and crash**.
- The semantic decoder output is `Linear(256, n_classes)` and the GT-label remap assumes the dataset's classes; CRCD's classes differ from Replica's.

**Three options, recommended order for a first overnight run:**
1. **(Fastest — get SLAM running, semantics nominal)** Keep `n_classes: 52`, keep `dinov2_replica.pth`, set `use_gt_semantic: False`. Feed CRCD `semantic_class` images that are valid for the loader (e.g. a single-class all-zeros mask, or the Replica head's own predictions). Semantics will be meaningless on surgical images, but **RGB-D tracking + mapping run** and you get ATE/recon. This proves the pipeline on CRCD tonight. Flag the semantics as not-yet-meaningful.
2. **(GT semantics, needs a matching head)** If CRCD GT masks exist: set `n_classes` = CRCD class count, set `use_gt_semantic: True`, and **replace the head load** — either skip loading `dinov2_replica.pth` (random-init head, trained on the fly via GT supervision) or train/provide a CRCD seg checkpoint. Requires a small code edit in `get_dinov2` to allow a fresh head when no compatible checkpoint exists. More faithful, more work.
3. **(Disable semantics)** Not recommended for a first pass — semantics is woven through both `Tracker` and `Mapper` losses; cleanly removing it is non-trivial.

> Recommendation: do **Option 1** tonight to confirm CRCD runs end-to-end, and leave Option 2 (real CRCD semantics) as the next iteration.

### 4.4 Config template (fill the CRCD specifics)
```yaml
# configs/CRCD/<seq>.yaml
inherit_from: configs/SNI-SLAM.yaml      # base; override what CRCD needs
dataset: 'tumrgbd'                        # or 'crcd' if you add a loader
cam:
  H: <rows>
  W: <cols>
  fx: <fx>
  fy: <fy>
  cx: <cx>
  cy: <cy>
  png_depth_scale: <scale>               # match your depth PNG encoding!
  crop_edge: 0
mapping:
  bound: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]            # SMALL for endoscopy
  marching_cubes_bound: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
model:
  cnn:
    n_classes: 52                        # keep 52 for Option 1 (see 4.3)
func:
  use_gt_semantic: False                 # Option 1
data:
  input_folder: data/CRCD/<seq>/
  output: output/CRCD/<seq>/test
```
Run + eval:
```bash
python -W ignore run.py configs/CRCD/<seq>.yaml
python src/tools/eval_ate.py configs/CRCD/<seq>.yaml      # only if GT poses exist
```

### 4.5 Where it will likely break (pre-empt)
- **`png_depth_scale` wrong** → depth collapses; tracking diverges immediately. Verify unique depth values are sensible after scaling (this bit DDS-SLAM before).
- **Bounds too large** (Replica-scale on an endoscopic scene) → empty/!exploded planes. Set bounds to the actual metric extent (tens of cm).
- **`load_state_dict` crash** → you changed `n_classes` off 52 without replacing the head (see 4.3).
- **RAM** → endoscopic snippets are short, so likely fine even on smaller GPUs, but keep A100 to be safe.
- **No GT poses** → SLAM runs but ATE is undefined; rely on rendered RGB/depth + mesh for qualitative check.

### 4.6 Goal-B success criterion for tonight
SNI-SLAM completes a CRCD sequence end-to-end (no crash), writes a mesh + checkpoint, and — if GT poses exist — an ATE number. Semantics meaningful is a **follow-up**, not tonight's bar.

---

## 5. Artifacts to collect (both goals)
Ship to Drive and note in the morning summary:
- `MyDrive/Outputs/sni_slam_replica/_eval/<scene>_metrics.txt` (+ any new `*_lrT001` / `*_seedK`),
- new per-scene SLAM outputs (`mesh/…culled.ply`, `ckpts/…tar`),
- CRCD: `output/CRCD/<seq>/test/` (mesh, ckpt, eval),
- the runner master log `MyDrive/Outputs/sni_slam_replica/_eval/run_replica_subset.log`.

Morning summary table: per-scene ATE RMSE/Mean (+ seed mean if A2b) and Comp.Ratio, vs the §3 paper column; CRCD: completed? ATE? where the mesh landed.

---

## 6. Reference for a fresh Claude session
- Memory index: `MEMORY.md` → start with `project_sni_slam_8scene_20260603` (results + corrected analysis + **verified per-scene paper targets** + eval protocol), `project_colab_env_gotchas_20260531`, `project_sni_slam_reproduction_state_20260531`, `project_sni_slam_setup_bug_20260602`.
- Paper: `sni-slam/paper/` (suppl PDF + `suppl_layout.txt`; Table 9 page rendered to `suppl_page_07.png`).
- Config audit conclusion: our configs == upstream except path lines; only config-vs-paper gap is `tracking.lr_T 0.002` vs paper 0.001.
- Scripts: `scripts/colab_setup_sni.sh` (setup), `scripts/run_replica_subset.sh` (run+eval, resumable).
- Personality/rules: caveman-direct, no assumptions, reproduce-first; do NOT cite the Acc "win" (it's a global eval offset, not a real win — see memory).
