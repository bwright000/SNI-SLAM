"""
Post-hoc per-frame renderer for SNI-SLAM (DDS-SLAM-parity).

PROBLEM
-------
SNI-SLAM's Frame_Visualizer.save_imgs (src/utils/Frame_Visualizer.py:55) saves
a 6-subplot matplotlib figure (GT/Generated/Residual for color+depth, plus
semantic if available) at sparse `vis_freq` intervals.  It does NOT save:
  - clean per-frame RGB renders (for PSNR/SSIM/LPIPS eval)
  - clean per-frame depth renders (for 6-panel video Panel 6)

DDS-SLAM does this natively at ddsslam.py:806-828, saving every frame to
$OUTPUT/{frame:04d}.jpg + $OUTPUT/depth/{frame:04d}.png.

SOLUTION
--------
Post-hoc renderer: load the final ckpt + dataset, iterate over all frames,
render at each frame's estimated pose, save in DDS-SLAM-compatible format
so generate_video.py + eval_rendering.py work unchanged.

Output structure:
  <output_dir>/rendered/{idx:04d}.jpg          # rendered RGB (BGR uint8)
  <output_dir>/rendered/depth/{idx:04d}.png    # rendered depth (uint16 at png_depth_scale)

Usage:
  python Addons/viz/render_all_frames_sni.py \
    --config configs/CRCD/c1_001.yaml \
    --skip 1
  # Auto-detects ckpt from cfg['data']['output']/ckpts/<last>.tar
  # Writes to cfg['data']['output']/rendered/

  python Addons/viz/render_all_frames_sni.py \
    --config configs/CRCD/c1_001.yaml \
    --checkpoint output/CRCD/c1_001/test/ckpts/00359.tar \
    --output_dir output/CRCD/c1_001/test/rendered \
    --skip 1
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Post-hoc per-frame renderer for SNI-SLAM')
    parser.add_argument('config', type=str, help='Per-snippet SNI-SLAM yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to ckpt .tar.  Auto-detects from cfg.output/ckpts/ if omitted.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for renders.  Defaults to cfg.output/rendered/.')
    parser.add_argument('--skip', type=int, default=1, help='Render every N-th frame')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Ensure repo root on sys.path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from src import config
    from src.utils.datasets import get_dataset
    from src.utils.Renderer import Renderer

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    output_root = cfg['data']['output']

    # Auto-detect ckpt
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = os.path.join(output_root, 'ckpts')
        if not os.path.isdir(ckpt_dir):
            print(f'FATAL: no ckpt dir at {ckpt_dir}'); sys.exit(1)
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.tar'))
        if not ckpts:
            print(f'FATAL: no .tar files in {ckpt_dir}'); sys.exit(1)
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f'Loading ckpt: {ckpt_path}')

    # Output dirs (DDS-SLAM-compatible layout: rendered/ + rendered/depth/)
    out_dir = args.output_dir or os.path.join(output_root, 'rendered')
    os.makedirs(out_dir, exist_ok=True)
    depth_out_dir = os.path.join(out_dir, 'depth')
    os.makedirs(depth_out_dir, exist_ok=True)
    print(f'Writing renders to: {out_dir}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Build minimal SNI-SLAM-like shim for Renderer ---
    # Avoids the multiprocessing forks that SNI_SLAM.__init__ does.
    class SniShim:
        pass
    sni = SniShim()
    sni.device = device
    sni.H = cfg['cam']['H']
    sni.W = cfg['cam']['W']
    sni.fx = cfg['cam']['fx']
    sni.fy = cfg['cam']['fy']
    sni.cx = cfg['cam']['cx']
    sni.cy = cfg['cam']['cy']
    # crop_edge handling (matches SNI_SLAM.update_cam)
    crop = cfg['cam'].get('crop_edge', 0)
    if crop > 0:
        sni.H -= crop * 2; sni.W -= crop * 2
        sni.cx -= crop;    sni.cy -= crop

    # Bound (matches SNI_SLAM.load_bound — bound is on device)
    sni.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])).float().to(device)

    # Model manager (semantic / DINOv2 backbone if use_gt_semantic)
    from src.networks.model_manager import ModelManager
    sni.model_manager = ModelManager(cfg)
    sni.model_manager.get_share_memory()

    # Decoders via config.get_model
    decoders = config.get_model(cfg).to(device)
    decoders.bound = sni.bound
    sni.shared_decoders = decoders

    # --- Load ckpt ---
    # weights_only kwarg is torch>=2.0 only; SNI-SLAM env is torch 1.11 (Py 3.7)
    ckpt = torch.load(ckpt_path, map_location=device)
    decoders.load_state_dict(ckpt['decoder_state_dict'])
    decoders.eval()

    # Tri-planes: lists of tensors at multiple resolutions.
    all_planes = (
        [p.to(device) for p in ckpt['planes_xy']],
        [p.to(device) for p in ckpt['planes_xz']],
        [p.to(device) for p in ckpt['planes_yz']],
        [p.to(device) for p in ckpt['c_planes_xy']],
        [p.to(device) for p in ckpt['c_planes_xz']],
        [p.to(device) for p in ckpt['c_planes_yz']],
        [p.to(device) for p in ckpt['s_planes_xy']],
        [p.to(device) for p in ckpt['s_planes_xz']],
        [p.to(device) for p in ckpt['s_planes_yz']],
    )
    est_c2w_list = ckpt['estimate_c2w_list'].to(device)
    n_img = ckpt['idx'] + 1
    print(f'Ckpt has {n_img} frames; est_c2w_list shape: {tuple(est_c2w_list.shape)}')

    # --- Renderer ---
    truncation = cfg['model']['truncation']
    renderer = Renderer(cfg, sni)

    # --- Dataset for GT depth ---
    sni.scale = cfg.get('scale', 1.0)
    args_obj = argparse.Namespace(input_folder=None, output=None)
    dataset = get_dataset(cfg, args_obj, sni.scale)
    if len(dataset) < n_img:
        print(f'WARN: dataset has {len(dataset)} frames but ckpt has {n_img}')
        n_img = min(n_img, len(dataset))

    png_depth_scale = float(cfg['cam'].get('png_depth_scale', 10000.0))
    print(f'png_depth_scale={png_depth_scale} (for uint16 depth save)')

    max_n = args.max_frames or n_img
    frame_indices = list(range(0, max_n, args.skip))
    print(f'Rendering {len(frame_indices)} frames (skip={args.skip}) ...')

    for idx in tqdm(frame_indices, desc='render'):
        # Get GT depth (needed for guided ray sampling in render_img).
        # SNI-SLAM datasets.py:126 returns 5 values: (index, color, depth, pose, semantic)
        try:
            item = dataset[idx]
        except Exception as e:
            print(f'  frame {idx}: dataset access failed: {e}'); continue
        gt_color = item[1]
        gt_depth = item[2]
        gt_depth = gt_depth.to(device).float()
        c2w = est_c2w_list[idx]
        if torch.isnan(c2w).any() or torch.isinf(c2w).any():
            print(f'  frame {idx}: invalid c2w (NaN/Inf), skipping'); continue

        with torch.no_grad():
            depth, color, _semantic = renderer.render_img(
                all_planes, decoders, c2w, truncation, device, gt_depth=gt_depth
            )

        # Save RGB (BGR for cv2)
        color_np = np.clip(color.cpu().numpy(), 0.0, 1.0)
        color_uint8 = (color_np * 255).astype(np.uint8)
        bgr = cv2.cvtColor(color_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f'{idx:04d}.jpg'), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save depth (uint16 at png_depth_scale)
        depth_np = depth.cpu().numpy()
        depth_uint16 = np.clip(depth_np * png_depth_scale, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_out_dir, f'{idx:04d}.png'), depth_uint16)

    n_rgb = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    n_depth = len([f for f in os.listdir(depth_out_dir) if f.endswith('.png')])
    print(f'\nDone. rgb={n_rgb} depth={n_depth}')
    print(f'  rendered RGB : {out_dir}/{{:04d}}.jpg')
    print(f'  rendered depth: {depth_out_dir}/{{:04d}}.png')


if __name__ == '__main__':
    main()
