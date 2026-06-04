"""
Extended trajectory metrics for SNI-SLAM — Sim3 + est_path/GT_path + per-axis Pearson.

The stock eval_ate.py is SE3-only (Horn alignment without scale) — matches paper
Replica convention but is misleading on monocular CRCD where scale is up-to-scale.

This script adds:
  - Sim3 ATE (Horn-with-scale)
  - est_path / GT_path ratio  (catches "tracker is jittering, not tracking")
  - Per-axis Pearson correlation between aligned-est and GT (catches shape decoupling)
  - WARN if ratio > 3.0 OR max|Pearson| < 0.1

Usage:
  python src/tools/eval_traj_extended.py configs/Replica/room0.yaml

Loads the same ckpt-based pose data as eval_ate.py (no timestamp dependency).
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.append('.')
from src import config
from src.common import matrix_to_cam_pose


def parse_c2w_list_to_xyz(c2w_list, N):
    """Extract translation [N, 3] from c2w pose list.  Skips NaN/Inf rows."""
    xyz = []
    for idx in range(N + 1):
        m = c2w_list[idx]
        if torch.isinf(m).any() or torch.isnan(m).any():
            xyz.append(None)
            continue
        xyz.append(m[:3, 3].cpu().numpy())
    return xyz


def horn_align(est, gt, with_scale=False):
    """
    Horn closed-form alignment of est -> gt.
    est, gt: [N, 3] numpy arrays.
    Returns aligned est [N, 3] and the scale factor s (1.0 if with_scale=False).
    """
    mc = est.mean(0)
    dc = gt.mean(0)
    mm = est - mc
    dd = gt - dc
    H = mm.T @ dd
    U, S, Vt = np.linalg.svd(H)
    ds = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, ds])
    R = Vt.T @ D @ U.T
    if with_scale:
        s = (S * np.array([1, 1, ds])).sum() / (mm * mm).sum()
    else:
        s = 1.0
    t = dc - s * R @ mc
    aligned = (s * (R @ est.T)).T + t
    return aligned, s


def main():
    parser = argparse.ArgumentParser(description='Extended ATE metrics: Sim3 + path-ratio + Pearson')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, help='Override output folder')
    parser.add_argument('--csv', type=str, help='Append summary row to CSV')
    parser.add_argument('--name', type=str, default='', help='Method/run label for CSV row')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    scale_cfg = cfg.get('scale', 1.0)
    output = cfg['data']['output'] if args.output is None else args.output
    ckptsdir = f'{output}/ckpts'
    if not os.path.exists(ckptsdir):
        print(f"FATAL: no ckpts dir at {ckptsdir}")
        return

    ckpts = sorted(os.path.join(ckptsdir, f) for f in os.listdir(ckptsdir) if 'tar' in f)
    if not ckpts:
        print(f"FATAL: no ckpt files in {ckptsdir}")
        return
    ckpt_path = ckpts[-1]
    print(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    N = ckpt['idx']
    est_c2w = ckpt['estimate_c2w_list']
    gt_c2w  = ckpt['gt_c2w_list']

    # Undo dataset.py scale=cfg['scale'] (divide by it to recover natural metres)
    est_xyz_list = []
    gt_xyz_list  = []
    for idx in range(N + 1):
        em, gm = est_c2w[idx], gt_c2w[idx]
        if torch.isinf(gm).any() or torch.isnan(gm).any():
            continue
        if torch.isinf(em).any() or torch.isnan(em).any():
            continue
        est_xyz_list.append((em[:3, 3] / scale_cfg).cpu().numpy())
        gt_xyz_list .append((gm[:3, 3] / scale_cfg).cpu().numpy())

    if len(est_xyz_list) < 10:
        print(f"FATAL: only {len(est_xyz_list)} valid pose pairs — cannot evaluate")
        return

    est_xyz = np.stack(est_xyz_list)
    gt_xyz  = np.stack(gt_xyz_list)
    n = len(est_xyz)
    print(f"Valid pose pairs: {n}")

    # SE3 (paper convention)
    est_se3, _      = horn_align(est_xyz, gt_xyz, with_scale=False)
    se3_err_per_pose = np.linalg.norm(est_se3 - gt_xyz, axis=1) * 100  # cm
    se3_rmse  = np.sqrt((se3_err_per_pose ** 2).mean())
    se3_mean  = se3_err_per_pose.mean()
    se3_median = np.median(se3_err_per_pose)

    # Sim3 (with-scale)
    est_sim3, s_factor = horn_align(est_xyz, gt_xyz, with_scale=True)
    sim3_err_per_pose  = np.linalg.norm(est_sim3 - gt_xyz, axis=1) * 100  # cm
    sim3_rmse  = np.sqrt((sim3_err_per_pose ** 2).mean())
    sim3_mean  = sim3_err_per_pose.mean()
    sim3_median = np.median(sim3_err_per_pose)

    # Path-length ratio
    est_path = np.linalg.norm(np.diff(est_xyz, axis=0), axis=1).sum() * 100  # cm
    gt_path  = np.linalg.norm(np.diff(gt_xyz,  axis=0), axis=1).sum() * 100
    path_ratio = est_path / max(gt_path, 1e-12)

    # Per-axis Pearson on Sim3-aligned positions
    pearson = []
    for axis in range(3):
        try:
            r = np.corrcoef(est_sim3[:, axis], gt_xyz[:, axis])[0, 1]
            pearson.append(r if np.isfinite(r) else 0.0)
        except Exception:
            pearson.append(0.0)
    max_abs_pearson = max(abs(p) for p in pearson)

    print()
    print(f"=== Extended trajectory metrics ===")
    print(f"SE3-aligned ATE (paper convention):")
    print(f"  RMSE   : {se3_rmse:.4f} cm")
    print(f"  Mean   : {se3_mean:.4f} cm")
    print(f"  Median : {se3_median:.4f} cm")
    print(f"Sim3-aligned ATE (with-scale, monocular-aware):")
    print(f"  RMSE   : {sim3_rmse:.4f} cm")
    print(f"  Mean   : {sim3_mean:.4f} cm")
    print(f"  Median : {sim3_median:.4f} cm")
    print(f"  Scale s: {s_factor:.4f}  (1.0 = perfectly metric)")
    print(f"Path lengths:")
    print(f"  est_path : {est_path:.2f} cm")
    print(f"  GT_path  : {gt_path:.2f} cm")
    print(f"  ratio    : {path_ratio:.2f}  (1.0 = ideal; >3 = tracker jittering)")
    print(f"Per-axis Pearson (est_sim3 vs GT):")
    print(f"  x: {pearson[0]:.3f}")
    print(f"  y: {pearson[1]:.3f}")
    print(f"  z: {pearson[2]:.3f}")
    print(f"  max|r|: {max_abs_pearson:.3f}  (>0.5 = shape-tracking; <0.1 = decoupled)")

    warnings = []
    if path_ratio > 3.0:
        warnings.append(f"path-ratio {path_ratio:.2f} > 3 — tracker jittering, NOT tracking")
    if max_abs_pearson < 0.1:
        warnings.append(f"max|Pearson| {max_abs_pearson:.3f} < 0.1 — trajectory shape uncorrelated with GT")
    if abs(np.log(s_factor)) > 0.1:
        warnings.append(f"sim3 scale {s_factor:.3f} > 10%% off 1.0 — depth-scale calibration likely off")
    if warnings:
        print()
        print("WARNINGS:")
        for w in warnings:
            print(f"  * {w}")

    if args.csv:
        import csv
        header = ['name', 'n_poses', 'se3_rmse_cm', 'se3_mean_cm', 'se3_median_cm',
                  'sim3_rmse_cm', 'sim3_mean_cm', 'sim3_median_cm', 'sim3_scale',
                  'est_path_cm', 'gt_path_cm', 'path_ratio',
                  'pearson_x', 'pearson_y', 'pearson_z', 'max_abs_pearson']
        row = [args.name or os.path.basename(output), n,
               f'{se3_rmse:.4f}', f'{se3_mean:.4f}', f'{se3_median:.4f}',
               f'{sim3_rmse:.4f}', f'{sim3_mean:.4f}', f'{sim3_median:.4f}', f'{s_factor:.4f}',
               f'{est_path:.2f}', f'{gt_path:.2f}', f'{path_ratio:.3f}',
               f'{pearson[0]:.4f}', f'{pearson[1]:.4f}', f'{pearson[2]:.4f}', f'{max_abs_pearson:.4f}']
        write_header = not os.path.isfile(args.csv)
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        print(f"\nAppended row to {args.csv}")


if __name__ == '__main__':
    main()
