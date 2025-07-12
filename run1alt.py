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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def apply_histogram_equalization(depth, vmin, vmax, colormap='turbo'):
    """Global histogram equalization."""
    depth_normalized = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    depth_equalized = cv2.equalizeHist(depth_normalized)
    depth_final = depth_equalized.astype(np.float32) / 255.0
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_final)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def apply_contrast_stretching(depth, percentile_low=2, percentile_high=98, colormap='turbo'):
    """Contrast stretching using percentiles."""
    depth_flat = depth.flatten()
    depth_flat = depth_flat[~np.isnan(depth_flat)]
    vmin_stretch = np.percentile(depth_flat, percentile_low)
    vmax_stretch = np.percentile(depth_flat, percentile_high)
    depth_clipped = np.clip(depth, vmin_stretch, vmax_stretch)
    depth_normalized = (depth_clipped - vmin_stretch) / (vmax_stretch - vmin_stretch)
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def apply_adaptive_histogram_equalization(depth, vmin, vmax, colormap='turbo', clip_limit=2.0, tile_grid_size=(8,8)):
    """Adaptive histogram equalization (CLAHE) for local contrast enhancement."""
    depth_normalized = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    depth_clahe = clahe.apply(depth_normalized)
    depth_final = depth_clahe.astype(np.float32) / 255.0
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_final)[:, :, :3]
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
    parser.add_argument('--enhance_contrast', action='store_true', help='enhance contrast for better object differentiation')
    parser.add_argument('--colormap', type=str, default='turbo', help='colormap for visualization (turbo, viridis, plasma, etc.)')
    parser.add_argument('--equalization_method', type=str, default='histogram',
                        choices=['histogram', 'contrast', 'adaptive'],
                        help='method for contrast enhancement')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='CLAHE clip limit for adaptive histogram equalization')
    parser.add_argument('--clahe_tile_grid', type=int, nargs=2, default=[8,8], help='CLAHE tile grid size, e.g. 8 8')

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
    video_depth_anything.load_state_dict(torch.load(
        f'./checkpoints/video_depth_anything_{args.encoder}.pth',
        map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    frames, target_fps = read_video_frames(
        args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(
        frames, target_fps, input_size=args.input_size, device=DEVICE,
        fp32=args.fp32, batch_size=args.batch_size, use_cpu_offload=args.use_cpu_offload)

    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Calculate global depth range
    all_depths = np.concatenate([d.flatten() for d in depths])
    vmin = np.nanmin(all_depths)
    vmax = np.nanmax(all_depths)
    print(f"Depth range: {vmin:.2f}m to {vmax:.2f}m")

    # Enhanced visualization for better object differentiation
    if args.enhance_contrast:
        print("🎨 Applying enhanced contrast visualization for better object differentiation")
        enhanced_frames = []
        for depth in depths:
            if args.equalization_method == 'histogram':
                enhanced_frame = apply_histogram_equalization(depth, vmin, vmax, args.colormap)
            elif args.equalization_method == 'contrast':
                enhanced_frame = apply_contrast_stretching(depth, colormap=args.colormap)
            elif args.equalization_method == 'adaptive':
                enhanced_frame = apply_adaptive_histogram_equalization(
                    depth, vmin, vmax, args.colormap,
                    clip_limit=args.clahe_clip_limit,
                    tile_grid_size=tuple(args.clahe_tile_grid)
                )
            else:
                raise ValueError("Unknown equalization method.")
            enhanced_frames.append(enhanced_frame)
        enhanced_frames_array = np.array(enhanced_frames)
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_enhanced_vis.mp4')
        save_video(enhanced_frames_array, depth_vis_path, fps=fps, is_depths=False)
        print(f"✅ Saved enhanced visualization to {depth_vis_path}")
    else:
        # Standard visualization
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_vis.mp4')
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    # Enhanced depth legend generation
    if args.save_legend:
        try:
            cmap = plt.get_cmap(args.colormap)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
            fig.subplots_adjust(hspace=0.4)
            norm_original = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar1 = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm_original, cmap=cmap),
                cax=ax1,
                orientation='horizontal',
                label='Original Depth Range (meters)'
            )
            if args.enhance_contrast:
                ax2.text(0.5, 0.5, f'Enhanced using {args.equalization_method} equalization\nColormap: {args.colormap}',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                ax2.set_title('Enhancement Method', fontsize=10)
                ax2.axis('off')
            else:
                ax2.axis('off')
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

