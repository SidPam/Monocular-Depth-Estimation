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

def apply_histogram_equalization(depth, vmin, vmax, colormap='turbo'):
    """
    Apply histogram equalization to depth values for better object differentiation
    This distributes colors evenly across the actual depth values present in the scene
    """
    # Convert depth to 8-bit range for histogram equalization
    depth_normalized = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # Apply OpenCV histogram equalization
    depth_equalized = cv2.equalizeHist(depth_normalized)
    
    # Convert back to [0,1] range for colormap
    depth_final = depth_equalized.astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_final)[:, :, :3]  # Extract RGB channels
    return (colored * 255).astype(np.uint8)

def apply_contrast_stretching(depth, percentile_low=2, percentile_high=98, colormap='turbo'):
    """
    Apply contrast stretching using percentiles to focus on actual object ranges
    This removes extreme outliers and enhances the main depth range
    """
    # Calculate percentiles to ignore extreme values
    depth_flat = depth.flatten()
    depth_flat = depth_flat[~np.isnan(depth_flat)]
    
    vmin_stretch = np.percentile(depth_flat, percentile_low)
    vmax_stretch = np.percentile(depth_flat, percentile_high)
    
    # Clip and normalize
    depth_clipped = np.clip(depth, vmin_stretch, vmax_stretch)
    depth_normalized = (depth_clipped - vmin_stretch) / (vmax_stretch - vmin_stretch)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def apply_distance_capping(depth, vmin, vmax, cap, colormap='turbo'):
    """
    Apply distance capping to depth visualization:
    - Colors vary normally between vmin and cap
    - Everything beyond cap uses the max color value
    """
    # Create mask for beyond-cap regions
    beyond_cap_mask = depth >= cap
    
    # Normalize within [vmin, cap]
    depth_normalized = np.clip(depth, vmin, cap)
    depth_normalized = (depth_normalized - vmin) / (cap - vmin + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_normalized)[:, :, :3]
    
    # Set beyond-cap regions to max color
    max_color = np.array(cmap(1.0))[:3]
    colored[beyond_cap_mask] = max_color
    
    return (colored * 255).astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1920)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for video processing to reduce memory usage')
    parser.add_argument('--use_cpu_offload', action='store_true', help='enable CPU offloading to reduce GPU memory usage')
    parser.add_argument('--save_legend', action='store_true', help='generate depth color legend')
    # Enhanced visualization parameters
    parser.add_argument('--enhance_contrast', action='store_true', help='enhance contrast for better object differentiation')
    parser.add_argument('--colormap', type=str, default='turbo', help='colormap for visualization (turbo, viridis, plasma, etc.)')
    parser.add_argument('--equalization_method', type=str, default='histogram', choices=['histogram', 'contrast'],
                       help='method for contrast enhancement')
    # Distance capping parameter
    parser.add_argument('--far_distance_cap', type=float, default=None,
                       help='Maximum distance (meters) to visualize; farther depths are shown as max color')

    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cpu':
        args.fp32 = True
        print("Running on CPU: forcing fp32 precision to avoid compatibility issues")

    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        
        torch.cuda.set_per_process_memory_fraction(0.85)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        
        if available_memory < 6:
            args.batch_size = 1
            print(f"Auto-optimizing for low VRAM: batch_size=1, input_size={args.input_size}, max_res={args.max_res}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32, batch_size=args.batch_size, use_cpu_offload=args.use_cpu_offload)
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Calculate global depth range
    all_depths = np.concatenate([d.flatten() for d in depths])
    vmin = np.nanmin(all_depths)
    vmax = np.nanmax(all_depths)
    
    print(f"Depth range: {vmin:.2f}m to {vmax:.2f}m")
    
    # Enhanced visualization for better object differentiation
    if args.enhance_contrast or args.far_distance_cap is not None:
        print("🎨 Applying enhanced visualization")
        enhanced_frames = []
        
        for depth in depths:
            # Apply contrast enhancement if requested
            if args.enhance_contrast:
                if args.equalization_method == 'histogram':
                    frame = apply_histogram_equalization(depth, vmin, vmax, args.colormap)
                else:
                    frame = apply_contrast_stretching(depth, colormap=args.colormap)
            else:
                # If no enhancement, create basic visualization
                depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
                cmap = plt.get_cmap(args.colormap)
                frame = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
            
            # Apply distance capping if requested
            if args.far_distance_cap is not None:
                frame = apply_distance_capping(depth, vmin, vmax, args.far_distance_cap, args.colormap)
            
            enhanced_frames.append(frame)
        
        # Convert list to NumPy array
        enhanced_frames_array = np.array(enhanced_frames)
        
        # Determine output path based on processing
        if args.far_distance_cap is not None:
            cap_str = f"_cap{int(args.far_distance_cap)}"
        else:
            cap_str = ""
            
        if args.enhance_contrast:
            method_str = f"_{args.equalization_method}"
        else:
            method_str = ""
            
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + f'_enhanced{method_str}{cap_str}_vis.mp4')
        save_video(enhanced_frames_array, depth_vis_path, fps=fps, is_depths=False)
        print(f"✅ Saved enhanced visualization to {depth_vis_path}")
    else:
        # Standard visualization
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_vis.mp4')
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
    
    # Enhanced depth legend generation
    if args.save_legend:
        try:
            # Create enhanced colormap visualization
            cmap = plt.get_cmap(args.colormap)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.subplots_adjust(bottom=0.25)
            
            # Determine visualization range
            if args.far_distance_cap is not None:
                vmax_legend = args.far_distance_cap
                range_label = f'Depth Range (0-{vmax_legend:.1f}m)'
                cap_note = f'Beyond {vmax_legend:.1f}m: Uniform Color'
            else:
                vmax_legend = vmax
                range_label = f'Depth Range ({vmin:.1f}-{vmax_legend:.1f}m)'
                cap_note = None
            
            # Create colorbar
            norm = plt.Normalize(vmin=0, vmax=vmax_legend)
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax,
                orientation='horizontal',
                label=range_label
            )
            
            # Add distance cap annotation if used
            if cap_note:
                plt.figtext(0.5, 0.05, cap_note, ha='center', fontsize=10, color='red')
            
            legend_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_depth_legend.png')
            plt.savefig(legend_path, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"✅ Saved depth legend to {legend_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate depth legend: {str(e)}")
    
    # Save original processed video
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
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()
        print(f"✅ Saved EXR depth maps to {depth_exr_dir}")

