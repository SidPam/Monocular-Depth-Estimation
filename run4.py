# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import numpy as np
import os
import torch
import gc
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

def save_colorbar_legend(output_path, cmap_name='inferno', height=300, width=60, min_label="Closest", max_label="Farthest", dpi=100):
    """Save depth color legend"""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    fig.subplots_adjust(left=0.4, right=0.7)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap_name), norm=norm, orientation='vertical')
    cb.set_ticks([0, 1])
    cb.set_ticklabels([min_label, max_label])
    cb.ax.tick_params(labelsize=10, length=0)
    cb.outline.set_linewidth(1)
    ax.set_title("Depth Legend", fontsize=12, pad=10)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close(fig)

def create_range_mask(depth_map, min_dist=0.0, max_dist=50.0):
    """Downweight only distant objects (beyond max_dist)"""
    mask = np.ones_like(depth_map)
    mask[depth_map > max_dist] = 0.1
    return mask

def enhance_range(depth_map, min_val=0.0, max_val=50.0):
    """Apply contrast stretching to target range"""
    in_range_mask = (depth_map >= min_val) & (depth_map <= max_val)
    if np.any(in_range_mask):
        in_range = depth_map[in_range_mask]
        p2, p98 = np.percentile(in_range, [2, 98])
        depth_map = np.clip((depth_map - p2) / (p98 - p2 + 1e-8), 0, 1)
    return depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1920)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--save_npz', action='store_true')
    parser.add_argument('--save_exr', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_cpu_offload', action='store_true')
    parser.add_argument('--focus_range', action='store_true')
    parser.add_argument('--min_range', type=float, default=0.0)
    parser.add_argument('--max_range', type=float, default=50.0)
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Memory optimization
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.75)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    # Initialize model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()
    
    # Read video frames
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    
    # Process video depth
    depths, fps = model.infer_video_depth(
        frames,
        target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
        batch_size=args.batch_size,
        use_cpu_offload=args.use_cpu_offload
    )
    
    # Debug: Print depth statistics
    if args.debug:
        print("\nRaw depth stats (frame 0):")
        print(f"Min: {depths[0].min():.2f}m, Max: {depths[0].max():.2f}m")
        print(f"Mean: {depths[0].mean():.2f}m, Median: {np.median(depths[0]):.2f}m")

    # Apply range optimization if requested
    if args.focus_range:
        print(f"Applying range optimization with min={args.min_range}m, max={args.max_range}m")
        for i in range(len(depths)):
            # Downweight distant objects only
            depths[i] = depths[i] * create_range_mask(depths[i], args.min_range, args.max_range)
            
            # Apply contrast enhancement (from run3.py)
            depths[i] = enhance_range(depths[i], args.min_range, args.max_range)
            
            # Debug prints
            if args.debug and i == 0:
                print("\nAfter range optimization (frame 0):")
                print(f"Min: {depths[i].min():.2f}, Max: {depths[i].max():.2f}")
                print(f"Dashboard region: {depths[i][100:110, 100:110].mean():.2f}")
                print(f"Road region: {depths[i][300:310, 500:510].mean():.2f}")

    # Prepare output paths
    video_name = os.path.basename(args.input_video)
    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_vis.mp4')

    # Save videos using the original method from run3.py
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
    
    # Save colorbar legend
    legend_path = os.path.join(args.output_dir, "depth_colorbar_legend.png")
    save_colorbar_legend(legend_path)
    print(f"Saved depth color legend to {legend_path}")

    # Save additional formats
    if args.save_npz:
        np.savez_compressed(
            os.path.join(args.output_dir, f"{os.path.splitext(video_name)[0]}_depths.npz"),
            depths=depths
        )
    
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, f"{os.path.splitext(video_name)[0]}_depths_exr")
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR, Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType.FLOAT)}
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

