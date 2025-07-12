import argparse
import numpy as np
import os
import torch
import gc
import cv2
import psutil
from scipy.interpolate import interp1d

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, (width, height)

def compute_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def warp_depth(prev_depth, flow):
    h, w = flow.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1) + flow
    warped_depth = cv2.remap(
        prev_depth.astype(np.float32),
        flow_map[...,0].astype(np.float32),
        flow_map[...,1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped_depth

def apply_scaling_and_refinement(depths, frames, lidar_refs, temporal_refine, prev_depth=None):
    scale_factors = np.ones(len(depths))
    for i, depth in enumerate(depths):
        ref = lidar_refs.get(i, lidar_refs.get('global', None))
        if ref:
            x, y, lidar_depth = ref
            if x is None or y is None:
                h, w = depth.shape
                x, y = w // 2, h // 2
            x = min(int(x), depth.shape[1] - 1)
            y = min(int(y), depth.shape[0] - 1)
            pred_depth = depth[y, x]
            if pred_depth > 0:
                scale_factors[i] = lidar_depth / pred_depth
    # Temporal propagation of scaling factors
    valid_indices = np.where(scale_factors != 1.0)[0]
    if len(valid_indices) > 0:
        if len(valid_indices) >= 2:
            f = interp1d(valid_indices, scale_factors[valid_indices], kind='linear', fill_value="extrapolate")
            for i in range(len(scale_factors)):
                if scale_factors[i] == 1.0:
                    scale_factors[i] = f(i)
        else:
            single_scale = scale_factors[valid_indices[0]]
            scale_factors = np.ones(len(depths)) * single_scale
    scaled_depths = [depth * scale_factors[i] for i, depth in enumerate(depths)]
    refined_depths = [scaled_depths[0]]
    if temporal_refine:
        for i in range(1, len(scaled_depths)):
            if i == 1 and prev_depth is not None:
                warped_depth = prev_depth
            else:
                flow = compute_flow(frames[i-1], frames[i])
                warped_depth = warp_depth(refined_depths[-1], flow)
            blended = 0.7 * scaled_depths[i] + 0.3 * warped_depth
            refined_depths.append(blended)
    return scaled_depths, refined_depths

def read_video_frames_streaming(video_path, max_len=-1, target_fps=-1, max_res=1280):
    """
    Memory-optimized frame reading - returns video info and frame generator
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if target_fps <= 0:
        target_fps = original_fps
    
    # Calculate frame sampling
    frame_step = max(1, int(original_fps / target_fps))
    
    # Calculate resize parameters
    if max_res > 0:
        if width > height:
            if width > max_res:
                new_width = max_res
                new_height = int(height * max_res / width)
            else:
                new_width, new_height = width, height
        else:
            if height > max_res:
                new_height = max_res
                new_width = int(width * max_res / height)
            else:
                new_width, new_height = width, height
    else:
        new_width, new_height = width, height
    
    def frame_generator():
        frame_idx = 0
        read_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_step == 0:
                if new_width != width or new_height != height:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                yield frame
                read_count += 1
                
                if max_len > 0 and read_count >= max_len:
                    break
            
            frame_idx += 1
        
        cap.release()
    
    return frame_generator(), target_fps, (new_width, new_height)

def process_chunk_streaming(frames_chunk, video_depth_anything, target_fps, args, lidar_refs, prev_depth=None):
    # Force garbage collection before processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    depths_chunk, _ = video_depth_anything.infer_video_depth(
        frames_chunk, target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
        batch_size=1,  # Force batch_size=1 for maximum memory efficiency
        use_cpu_offload=True  # Force CPU offload
    )
    
    scaled_depths, refined_depths = apply_scaling_and_refinement(
        depths_chunk,
        frames_chunk,
        lidar_refs,
        args.temporal_refine,
        prev_depth
    )
    
    # Immediate cleanup
    del depths_chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return scaled_depths, refined_depths, frames_chunk[-1], refined_depths[-1] if args.temporal_refine else scaled_depths[-1]

def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    return memory.percent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything - Memory Optimized')
    parser.add_argument('--input_video', type=str, default='../assets/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--save_npz', action='store_true', help='Save depth maps as .npz files')
    parser.add_argument('--save_exr', action='store_true', help='Save depth maps as .exr files')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='Scale factor for metric depth')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--lidar_ref', type=str, help='CSV/path with frame_idx,x,y,lidar_depth (or single value)')
    parser.add_argument('--temporal_refine', action='store_true', help='Apply temporal refinement for depth consistency')
    parser.add_argument('--chunk_size', type=int, default=16, help='Frames per processing chunk (reduced for memory)')
    parser.add_argument('--overlap_frames', type=int, default=1, help='Overlap frames between chunks')
    parser.add_argument('--memory_limit', type=int, default=85, help='Stop processing if memory usage exceeds this percentage')
    args = parser.parse_args()

    # Force memory-optimized settings for RTX 4050
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1  # Always use batch_size=1
    use_cpu_offload = True  # Always use CPU offload
    
    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        
        # Set conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.75)  # Use only 75% of VRAM
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'  # Smaller memory blocks
        
        # Reduce chunk size for low VRAM
        if available_memory < 8:
            args.chunk_size = min(args.chunk_size, 8)
            print(f"Low VRAM optimization: chunk_size={args.chunk_size}, batch_size=1, cpu_offload=True")

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    lidar_refs = {}
    if args.lidar_ref:
        if args.lidar_ref.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.lidar_ref)
            for _, row in df.iterrows():
                lidar_refs[row['frame_idx']] = (row['x'], row['y'], row['lidar_depth'])
        else:
            lidar_depth = float(args.lidar_ref)
            lidar_refs['global'] = (None, None, lidar_depth)

    print("Loading model...")
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    # Initial cleanup
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    print("Setting up streaming video reader...")
    frame_generator, target_fps, (frame_width, frame_height) = read_video_frames_streaming(
        args.input_video, args.max_len, args.target_fps, args.max_res
    )

    # Prepare output paths
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')

    # Open video writers
    import imageio
    src_writer = imageio.get_writer(processed_video_path, fps=target_fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    depth_writer = imageio.get_writer(depth_vis_path, fps=target_fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])

    print("Starting streaming processing...")
    
    chunk_frames = []
    frame_count = 0
    prev_depth = None
    depth_storage = []  # For NPZ/EXR saving
    
    try:
        for frame in frame_generator:
            # Monitor memory usage
            memory_usage = monitor_memory()
            if memory_usage > args.memory_limit:
                print(f"Memory usage ({memory_usage:.1f}%) exceeds limit ({args.memory_limit}%), forcing cleanup...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            chunk_frames.append(frame)
            frame_count += 1
            
            # Process when chunk is full
            if len(chunk_frames) >= args.chunk_size:
                print(f"Processing frames {frame_count-len(chunk_frames)+1}-{frame_count} (Memory: {memory_usage:.1f}%)")
                
                scaled_depths, refined_depths, last_frame, prev_depth = process_chunk_streaming(
                    chunk_frames, video_depth_anything, target_fps, args, lidar_refs, prev_depth
                )
                
                current_depths = refined_depths if args.temporal_refine else scaled_depths
                
                # Save video frames incrementally
                for i, frame in enumerate(chunk_frames):
                    src_writer.append_data(frame.astype(np.uint8))
                    
                    # Depth visualization
                    depth = current_depths[i]
                    d_min, d_max = depth.min(), depth.max()
                    depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
                    
                    if args.grayscale:
                        depth_vis = depth_norm
                    else:
                        import matplotlib.pyplot as plt
                        colormap = np.array(plt.get_cmap("inferno").colors)
                        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
                    
                    depth_writer.append_data(depth_vis)
                
                # Store depths for later saving if needed
                if args.save_npz or args.save_exr:
                    depth_storage.extend(current_depths)
                
                # Cleanup
                del chunk_frames, scaled_depths, refined_depths, current_depths
                chunk_frames = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Process remaining frames
        if chunk_frames:
            print(f"Processing final {len(chunk_frames)} frames")
            scaled_depths, refined_depths, _, _ = process_chunk_streaming(
                chunk_frames, video_depth_anything, target_fps, args, lidar_refs, prev_depth
            )
            
            current_depths = refined_depths if args.temporal_refine else scaled_depths
            
            for i, frame in enumerate(chunk_frames):
                src_writer.append_data(frame.astype(np.uint8))
                
                depth = current_depths[i]
                d_min, d_max = depth.min(), depth.max()
                depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
                
                if args.grayscale:
                    depth_vis = depth_norm
                else:
                    import matplotlib.pyplot as plt
                    colormap = np.array(plt.get_cmap("inferno").colors)
                    depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
                
                depth_writer.append_data(depth_vis)
            
            if args.save_npz or args.save_exr:
                depth_storage.extend(current_depths)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Attempting to save partial results...")
    
    finally:
        src_writer.close()
        depth_writer.close()
        print("Video processing completed.")

    # Save NPZ/EXR files if requested
    if args.save_npz and depth_storage:
        print("Saving NPZ files...")
        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npy')
        os.makedirs(depth_npz_dir, exist_ok=True)
        for i, depth in enumerate(depth_storage):
            scale_factor = 1.0
            ref = lidar_refs.get(i, lidar_refs.get('global', None))
            if ref:
                x, y, lidar_depth = ref
                if x is None or y is None:
                    h, w = depth.shape
                    x, y = w // 2, h // 2
                x = min(int(x), depth.shape[1] - 1)
                y = min(int(y), depth.shape[0] - 1)
                pred_depth = depth[y, x]
                if pred_depth > 0:
                    scale_factor = lidar_depth / pred_depth
            scaled_depth = depth * scale_factor
            output_npy = f"{depth_npz_dir}/frame_{i:05d}.npy"
            np.save(output_npy, scaled_depth)

    if args.save_exr and depth_storage:
        print("Saving EXR files...")
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        try:
            import OpenEXR
            import Imath
            for i, depth in enumerate(depth_storage):
                scale_factor = 1.0
                ref = lidar_refs.get(i, lidar_refs.get('global', None))
                if ref:
                    x, y, lidar_depth = ref
                    if x is None or y is None:
                        h, w = depth.shape
                        x, y = w // 2, h // 2
                    x = min(int(x), depth.shape[1] - 1)
                    y = min(int(y), depth.shape[0] - 1)
                    pred_depth = depth[y, x]
                    if pred_depth > 0:
                        scale_factor = lidar_depth / pred_depth
                scaled_depth = depth * scale_factor
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": scaled_depth.tobytes()})
                exr_file.close()
        except ImportError:
            print("OpenEXR not available, skipping EXR export")

    print("All processing completed successfully!")
