# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
import argparse
import numpy as np
import os
import torch
import gc
import cv2
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

import matplotlib.pyplot as plt
def create_colorbar_legend(cmap_name='inferno', width=50, height=300):
    """
    Create a vertical colorbar legend image showing depth from closest (bottom) to farthest (top).
    Returns a uint8 RGB image.
    """
    fig, ax = plt.subplots(figsize=(2, 6))
    gradient = np.linspace(0, 1, height).reshape(-1, 1)
    gradient = np.flipud(gradient)  # So bottom is closest (dark) and top farthest (bright)
    
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap_name))
    ax.set_axis_off()
    
    # Add labels for closest and farthest
    ax.text(1.5, height - 10, 'Closest', va='bottom', ha='left', fontsize=10, color='black', transform=ax.transData)
    ax.text(1.5, 10, 'Farthest', va='top', ha='left', fontsize=10, color='black', transform=ax.transData)
    
    # Save to buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Resize width to desired size
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def create_range_mask(depth_map, min_dist=3.0, max_dist=50.0):
    """Create mask prioritizing 3-50m range"""
    mask = np.ones_like(depth_map)
    mask[(depth_map < min_dist) | (depth_map > max_dist)] = 0.1  # Downweight irrelevant areas
    return mask

def enhance_range(depth_map, min_val=3.0, max_val=50.0):
    """Apply contrast stretching to target range"""
    in_range_mask = (depth_map >= min_val) & (depth_map <= max_val)
    if np.any(in_range_mask):
        in_range = depth_map[in_range_mask]
        p2, p98 = np.percentile(in_range, [2, 98])
        depth_map = np.clip((depth_map - p2) / (p98 - p2 + 1e-8), 0, 1)
    return depth_map

def range_optimized_processing(depths, min_range=3.0, max_range=50.0):
    """Apply range optimization to depth sequence"""
    optimized_depths = []
    for depth in depths:
        # Apply masking to downweight irrelevant areas
        masked_depth = depth * create_range_mask(depth, min_range, max_range)
        
        # Enhance contrast in critical range
        optimized_depth = enhance_range(masked_depth, min_range, max_range)
        optimized_depths.append(optimized_depth)
    return optimized_depths

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
    
    # Range optimization parameters
    parser.add_argument('--focus_range', action='store_true', help='Optimize for specific depth range')
    parser.add_argument('--min_range', type=float, default=3.0, help='Min distance for optimization (meters)')
    parser.add_argument('--max_range', type=float, default=50.0, help='Max distance for optimization (meters)')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Memory optimization for low VRAM GPUs
    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        
        '''
        if available_memory < 6 and args.encoder == 'vitl':
            print("Switching to vits model due to low VRAM")
            args.encoder = 'vits'
        '''
        
        # Memory optimization settings
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
    
    # Clear cache after model loading
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # Read video frames
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    
    # Process video depth
    depths, fps = video_depth_anything.infer_video_depth(
        frames,
        target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
        batch_size=args.batch_size,
        use_cpu_offload=args.use_cpu_offload
    )

    # Apply range optimization if requested
    if args.focus_range:
        print(f"Applying range optimization (3-50m focus) with min={args.min_range}m, max={args.max_range}m")
        # Process in-place to avoid memory spike
        for i in range(len(depths)):
            # Apply masking to downweight irrelevant areas
            masked_depth = depths[i] * create_range_mask(depths[i], args.min_range, args.max_range)
            # Enhance contrast in critical range
            depths[i] = enhance_range(masked_depth, args.min_range, args.max_range)
            
            # Clear cache periodically during processing
            if DEVICE == 'cuda' and i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # Prepare output
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    
    # Save processed videos
    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
    
    # Save the colorbar legend as a separate image
    legend_path = os.path.join(args.output_dir, "depth_colorbar_legend.png")
    save_colorbar_legend(legend_path, cmap_name="inferno", min_label="Closest", max_label="Farthest")
    print(f"Saved colorbar legend to {legend_path}")


    # Save additional formats
    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
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

