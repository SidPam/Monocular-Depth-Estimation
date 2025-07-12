# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import os
import torch
import gc
import cv2
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

# Import for legend generation and enhanced visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def apply_traffic_zones(depth, critical_max=10, traffic_max=30):
    """
    Traffic-optimized multi-zone visualization:
    - Critical zone (0-10m): Reds colormap (high contrast for immediate objects)
    - Traffic zone (10-30m): Viridis colormap (clear differentiation for road objects)
    - Background (30m+): Uniform dark gray
    """
    # Initialize output image
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    # Critical zone: 0-10m (immediate threats)
    critical_mask = depth <= critical_max
    if np.any(critical_mask):
        critical_depths = depth[critical_mask]
        critical_norm = critical_depths / critical_max
        # Reds colormap: 0=dark red, 1=bright yellow
        critical_colors = plt.get_cmap('Reds')(critical_norm)[:, :3]
        result[critical_mask] = (critical_colors * 255).astype(np.uint8)
    
    # Traffic zone: 10-30m (road-level objects)
    traffic_mask = (depth > critical_max) & (depth <= traffic_max)
    if np.any(traffic_mask):
        traffic_depths = depth[traffic_mask]
        traffic_norm = (traffic_depths - critical_max) / (traffic_max - critical_max)
        # Viridis colormap: 0=purple, 1=yellow
        traffic_colors = plt.get_cmap('viridis')(traffic_norm)[:, :3]
        result[traffic_mask] = (traffic_colors * 255).astype(np.uint8)
    
    # Background: >30m (uniform dark gray)
    bg_mask = depth > traffic_max
    result[bg_mask] = [64, 64, 64]  # Dark gray
    
    return result

def generate_traffic_legend(critical_max=10, traffic_max=30):
    """Generate traffic-optimized legend showing all three zones"""
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.subplots_adjust(bottom=0.3)
    
    # Create custom colormap for legend
    colors = []
    zones = [
        ("Critical (0-10m)", 'Reds', (0, critical_max)),
        ("Traffic (10-30m)", 'viridis', (critical_max, traffic_max)),
        ("Background (30m+)", 'gray', (traffic_max, traffic_max + 10))
    ]
    
    # Create colorbars for each zone
    for i, (label, cmap, (vmin, vmax)) in enumerate(zones):
        cax = fig.add_axes([0.05 + i*0.3, 0.6, 0.25, 0.3])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        
        if cmap == 'gray':
            # Uniform color for background
            cax.imshow([[[64, 64, 64]]], extent=[0, 1, 0, 1])
        else:
            # Gradient for critical and traffic zones
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cax.imshow(gradient, aspect='auto', cmap=cmap_obj)
        
        cax.set_title(label, fontsize=10)
        cax.set_xticks([])
        cax.set_yticks([])
    
    plt.figtext(0.5, 0.1, "Traffic-Optimized Depth Visualization", ha='center', fontsize=12)
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1920)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--save_npz', action='store_true')
    parser.add_argument('--save_exr', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_cpu_offload', action='store_true')
    parser.add_argument('--save_legend', action='store_true')
    # Traffic zone parameters
    parser.add_argument('--critical_max', type=float, default=10.0, 
                       help='Maximum distance for critical zone (0-10m)')
    parser.add_argument('--traffic_max', type=float, default=30.0, 
                       help='Maximum distance for traffic zone (10-30m)')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cpu':
        args.fp32 = True
        print("Running on CPU: forcing fp32 precision")

    # Memory optimization for low VRAM GPUs
    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        
        if available_memory < 6:
            args.batch_size = max(1, args.batch_size // 4)
            print(f"Optimizing for low VRAM: batch_size={args.batch_size}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', 
                                                   map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(
        frames, target_fps, input_size=args.input_size, 
        device=DEVICE, fp32=args.fp32, 
        batch_size=args.batch_size, use_cpu_offload=args.use_cpu_offload
    )
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Calculate global depth range
    all_depths = np.concatenate([d.flatten() for d in depths])
    vmin = np.nanmin(all_depths)
    vmax = np.nanmax(all_depths)
    print(f"Depth range: {vmin:.2f}m to {vmax:.2f}m")
    
    # Apply traffic-optimized visualization
    print("🚗 Applying traffic-optimized multi-zone visualization")
    traffic_frames = []
    for depth in depths:
        frame = apply_traffic_zones(depth, args.critical_max, args.traffic_max)
        traffic_frames.append(frame)
    
    # Convert to array and save video
    traffic_frames_array = np.array(traffic_frames)
    depth_vis_path = os.path.join(
        args.output_dir, 
        os.path.splitext(video_name)[0] + f'_traffic_vis.mp4'
    )
    save_video(traffic_frames_array, depth_vis_path, fps=fps, is_depths=False)
    print(f"✅ Saved traffic-optimized visualization to {depth_vis_path}")
    
    # Generate traffic-specific legend
    if args.save_legend:
        try:
            fig = generate_traffic_legend(args.critical_max, args.traffic_max)
            legend_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_traffic_legend.png')
            plt.savefig(legend_path, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"✅ Saved traffic legend to {legend_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate legend: {str(e)}")
    
    # Save original video
    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_src.mp4')
    save_video(frames, processed_video_path, fps=fps)
    
    # Save depth data formats
    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
        print(f"✅ Saved depth maps to {depth_npz_path}")
    
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()
        print(f"✅ Saved EXR depth maps to {depth_exr_dir}")

