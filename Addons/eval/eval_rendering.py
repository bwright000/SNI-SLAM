"""
SNI-SLAM render-quality evaluation — PSNR, SSIM, LPIPS.

Ported from DDS-SLAM/Addons/eval/eval_rendering.py (2026-06-04) for parity
between the two systems on shared CRCD benchmark.

Computes:
  - PSNR (peak signal-to-noise ratio)
  - SSIM (structural similarity)
  - LPIPS (learned perceptual; falls back to PSNR+SSIM only if `lpips` package missing)

Usage:
  python Addons/eval/eval_rendering.py \\
      --gt_dir data/CRCD/C1_001/rgb \\
      --render_dir output/CRCD/c1_001/test/rendered \\
      --name "SNI-SLAM c1_001" \\
      --output_csv output/CRCD/c1_001/render_eval.csv \\
      --summary_csv output/CRCD/_render_summary.csv \\
      --sequence "CRCD (C1_001)"

GT filename patterns supported (in order):
  *-left.png   (SemSup convention)
  *_left.png   (alternate)
  rgb_*.png    (SNI-SLAM Replica/CRCD convention from preprocess_crcd_for_sni.py)
  *l.png       (StereoMIS / DDS-SLAM CRCD convention)
"""

import argparse
import csv
import glob
import os
import cv2
import numpy as np


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


# Paper references where applicable.  SNI-SLAM paper does NOT report PSNR/SSIM/LPIPS
# on CRCD (CRCD didn't exist when paper was written); DDS-SLAM does report on StereoMIS.
PAPER_REFERENCES = {
    'StereoMIS (P2_1)': {'PSNR': 22.513, 'SSIM': 0.592, 'LPIPS': 0.496, 'source': 'DDS-SLAM Table II'},
    'CRCD (C1_001)':    {'PSNR': None,   'SSIM': None,  'LPIPS': None,  'source': 'no paper baseline'},
    'CRCD (C2_001)':    {'PSNR': None,   'SSIM': None,  'LPIPS': None,  'source': 'no paper baseline'},
    'CRCD (F1_002)':    {'PSNR': None,   'SSIM': None,  'LPIPS': None,  'source': 'no paper baseline'},
    'CRCD (F3_007)':    {'PSNR': None,   'SSIM': None,  'LPIPS': None,  'source': 'no paper baseline'},
}


def find_gt_images(gt_dir):
    """Try common naming conventions in order."""
    for pattern in ('*-left.png', '*_left.png', 'rgb_*.png'):
        files = sorted(glob.glob(os.path.join(gt_dir, pattern)))
        if files:
            return files, pattern
    # StereoMIS pattern: *l.png (exclude *r.png)
    all_l = sorted(glob.glob(os.path.join(gt_dir, '*l.png')))
    files = [f for f in all_l if not f.endswith('r.png')]
    if files:
        return files, '*l.png'
    return [], None


def main():
    parser = argparse.ArgumentParser(description='SNI-SLAM rendering quality eval (PSNR/SSIM/LPIPS)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth RGB directory')
    parser.add_argument('--render_dir', type=str, required=True,
                        help='Rendered output directory (e.g., output/CRCD/c1_001/test/rendered)')
    parser.add_argument('--name', type=str, default='', help='Method/run label')
    parser.add_argument('--output_csv', type=str, default='',
                        help='Save per-frame metrics to CSV')
    parser.add_argument('--summary_csv', type=str, default='',
                        help='Append summary row to cross-method CSV')
    parser.add_argument('--sequence', type=str, default='CRCD (C1_001)',
                        choices=list(PAPER_REFERENCES.keys()),
                        help='Sequence label (selects paper reference values)')
    parser.add_argument('--gt_offset', type=int, default=0,
                        help='Offset added to numeric-named render indices before GT lookup')
    args = parser.parse_args()

    # Rendered images (skip *_gt.* sidecars if present)
    rendered = []
    for ext in ('*.png', '*.jpg'):
        rendered.extend(p for p in sorted(glob.glob(os.path.join(args.render_dir, ext)))
                          if '_gt' not in os.path.basename(p))
    rendered = sorted(rendered)
    if not rendered:
        print(f"No rendered images in {args.render_dir}")
        return

    gt_images, gt_pattern = find_gt_images(args.gt_dir)
    if not gt_images:
        print(f"No GT images in {args.gt_dir}")
        return

    print(f"Method  : {args.name}")
    print(f"GT      : {len(gt_images)} images ({gt_pattern} in {args.gt_dir})")
    print(f"Render  : {len(rendered)} images in {args.render_dir}")

    try:
        import torch
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
        has_lpips = True
        print("LPIPS   : available")
    except ImportError:
        has_lpips = False
        print("LPIPS   : NOT available (pip install lpips to enable)")

    psnr_list, ssim_list, lpips_list, frame_indices = [], [], [], []

    def _parse_idx(path):
        # Try to extract numeric index from filename (e.g., 0042.jpg -> 42)
        stem = os.path.splitext(os.path.basename(path))[0]
        # Strip common prefixes
        for prefix in ('rgb_', 'render_', 'frame_'):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        try:
            return int(stem)
        except ValueError:
            return None

    numeric_render = all(_parse_idx(p) is not None for p in rendered)
    if numeric_render:
        pairs = [(p, _parse_idx(p) + args.gt_offset) for p in rendered
                 if _parse_idx(p) + args.gt_offset < len(gt_images)]
        print(f"Pairing : by filename index, gt_offset={args.gt_offset}, {len(pairs)} pairs")
    else:
        n_eval = min(len(rendered), len(gt_images))
        pairs = [(rendered[i], i) for i in range(n_eval)]
        print(f"Pairing : positional, {n_eval} pairs")

    for render_path, gt_idx in pairs:
        render = cv2.imread(render_path)
        if render is None:
            continue
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.imread(gt_images[gt_idx])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_indices.append(gt_idx)
        if render.shape != gt.shape:
            render = cv2.resize(render, (gt.shape[1], gt.shape[0]))
        psnr_list.append(compute_psnr(render, gt))
        ssim_list.append(compute_ssim(render, gt))
        if has_lpips:
            import torch
            r_t = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            g_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            if torch.cuda.is_available():
                r_t = r_t.cuda(); g_t = g_t.cuda()
            lpips_list.append(lpips_fn(r_t, g_t).item())

    n = len(psnr_list)
    if n == 0:
        print("No frames evaluated.")
        return

    print()
    print("=" * 50)
    print(f"Results: {args.name} ({n} frames)")
    print("=" * 50)
    print(f"PSNR  : {np.mean(psnr_list):.3f} (std {np.std(psnr_list):.3f})")
    print(f"SSIM  : {np.mean(ssim_list):.3f} (std {np.std(ssim_list):.3f})")
    if lpips_list:
        print(f"LPIPS : {np.mean(lpips_list):.3f} (std {np.std(lpips_list):.3f})")
    print("=" * 50)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['frame', 'psnr', 'ssim']
            if lpips_list:
                header.append('lpips')
            w.writerow(header)
            for i in range(n):
                row = [frame_indices[i], f"{psnr_list[i]:.4f}", f"{ssim_list[i]:.4f}"]
                if lpips_list:
                    row.append(f"{lpips_list[i]:.4f}")
                w.writerow(row)
        print(f"Per-frame CSV: {args.output_csv}")

    ref = PAPER_REFERENCES[args.sequence]
    if ref['PSNR'] is not None:
        print(f"\nPaper reference ({ref['source']}, {args.sequence}):")
        print(f"  PSNR : {ref['PSNR']}")
        print(f"  SSIM : {ref['SSIM']}")
        print(f"  LPIPS: {ref['LPIPS']}")
    else:
        print(f"\nNo paper reference for {args.sequence} ({ref['source']}).")

    if args.summary_csv:
        write_header = not os.path.exists(args.summary_csv)
        os.makedirs(os.path.dirname(args.summary_csv) or '.', exist_ok=True)
        with open(args.summary_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                header = ['method', 'sequence', 'n_frames',
                          'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std']
                if lpips_list:
                    header += ['lpips_mean', 'lpips_std']
                w.writerow(header)
            row = [args.name, args.sequence, n,
                   f"{np.mean(psnr_list):.3f}", f"{np.std(psnr_list):.3f}",
                   f"{np.mean(ssim_list):.3f}", f"{np.std(ssim_list):.3f}"]
            if lpips_list:
                row += [f"{np.mean(lpips_list):.3f}", f"{np.std(lpips_list):.3f}"]
            w.writerow(row)
        print(f"Summary CSV : {args.summary_csv}")


if __name__ == '__main__':
    main()
