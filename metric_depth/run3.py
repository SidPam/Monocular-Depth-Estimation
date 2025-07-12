import argparse
import numpy as np
import os
import torch
import gc
import cv2
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

def compute_global_scale_factors(frames, lidar_refs, video_depth_anything, target_fps, args):
    """Compute scaling factors for all LiDAR reference frames first"""
    print("Computing global scale factors for all LiDAR references...")
    
    global_scales = {}
    reference_frames = []
    
    # Collect all LiDAR reference frame indices
    for frame_idx in lidar_refs.keys():
        if frame_idx != 'global' and isinstance(frame_idx, int):
            reference_frames.append(frame_idx)
    
    reference_frames.sort()
    print(f"Found LiDAR references at frames: {reference_frames}")
    
    # Process reference frames to get actual scale factors
    for frame_idx in reference_frames:
        print(f"Processing reference frame {frame_idx}...")
        frame = frames[frame_idx:frame_idx+1]  # Single frame as numpy array
        
        depth, _ = video_depth_anything.infer_video_depth(
            frame, target_fps,
            input_size=args.input_size,
            device=DEVICE,
            fp32=args.fp32,
            batch_size=1,
            use_cpu_offload=use_cpu_offload
        )
        
        ref = lidar_refs[frame_idx]
        x, y, lidar_depth = ref
        
        if x is None or y is None:
            h, w = depth[0].shape
            x, y = w // 2, h // 2
        
        x = min(int(x), depth[0].shape[1] - 1)
        y = min(int(y), depth[0].shape[0] - 1)
        pred_depth = depth[0][y, x]
        
        if pred_depth > 0:
            scale_factor = lidar_depth / pred_depth
            global_scales[frame_idx] = scale_factor
            print(f"Frame {frame_idx}: LiDAR={lidar_depth:.2f}, Pred={pred_depth:.2f}, Scale={scale_factor:.4f}")
        
        del depth, frame
        torch.cuda.empty_cache()
    
    return global_scales

def interpolate_global_scales(total_frames, global_scales):
    """Create smooth interpolation between all reference frames"""
    if len(global_scales) == 0:
        return np.ones(total_frames)
    
    frame_indices = sorted(global_scales.keys())
    scale_values = [global_scales[idx] for idx in frame_indices]
    
    print(f"Creating global interpolation from frames {frame_indices}")
    print(f"Scale values: {scale_values}")
    
    # Create interpolation function
    if len(frame_indices) == 1:
        # Single reference - use constant scaling
        return np.full(total_frames, scale_values[0])
    
    # Extend interpolation to cover entire video
    extended_indices = []
    extended_scales = []
    
    # Add first frame if not already a reference
    if frame_indices[0] != 0:
        extended_indices.append(0)
        extended_scales.append(scale_values[0])
    
    # Add all reference frames
    extended_indices.extend(frame_indices)
    extended_scales.extend(scale_values)
    
    # Add last frame if not already a reference
    if frame_indices[-1] != total_frames - 1:
        extended_indices.append(total_frames - 1)
        extended_scales.append(scale_values[-1])
    
    # Create smooth interpolation across entire video
    f = interp1d(extended_indices, extended_scales, kind='linear', fill_value="extrapolate")
    interpolated_scales = f(np.arange(total_frames))
    
    return interpolated_scales

def process_chunk_with_scales(frames_chunk, scale_factors, video_depth_anything, target_fps, args, prev_depth=None):
    """Process chunk with pre-computed scale factors"""
    depths_chunk, _ = video_depth_anything.infer_video_depth(
        frames_chunk, target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
        batch_size=batch_size,
        use_cpu_offload=use_cpu_offload
    )
    
    # Apply pre-computed scaling
    scaled_depths = [depth * scale_factors[i] for i, depth in enumerate(depths_chunk)]
    
    # Temporal refinement
    refined_depths = [scaled_depths[0]]
    if args.temporal_refine:
        for i in range(1, len(scaled_depths)):
            if i == 1 and prev_depth is not None:
                warped_depth = prev_depth
            else:
                flow = compute_flow(frames_chunk[i-1], frames_chunk[i])
                warped_depth = warp_depth(refined_depths[-1], flow)
            blended = 0.7 * scaled_depths[i] + 0.3 * warped_depth
            refined_depths.append(blended)
    
    del depths_chunk
    torch.cuda.empty_cache()
    gc.collect()
    
    return scaled_depths, refined_depths, frames_chunk[-1], refined_depths[-1] if args.temporal_refine else scaled_depths[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
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
    parser.add_argument('--chunk_size', type=int, default=50, help='Frames per processing chunk')
    parser.add_argument('--overlap_frames', type=int, default=1, help='Overlap frames between chunks')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    use_cpu_offload = False
    
    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        torch.cuda.set_per_process_memory_fraction(0.85)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        
        if available_memory < 6:
            batch_size = 1
            use_cpu_offload = True
            print(f"Auto-optimizing for low VRAM: batch_size=1, input_size={args.input_size}, max_res={args.max_res}, cpu_offload=True")
        elif available_memory < 8:
            batch_size = 2
            print(f"Auto-optimizing for limited VRAM: batch_size=2, input_size={args.input_size}, max_res={args.max_res}")

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Load LiDAR references
    lidar_refs = {}
    if args.lidar_ref:
        if args.lidar_ref.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.lidar_ref)
            for _, row in df.iterrows():
                lidar_refs[int(row['frame_idx'])] = (row['x'], row['y'], row['lidar_depth'])
            print(f"Loaded {len(lidar_refs)} LiDAR references from CSV")
        else:
            lidar_depth = float(args.lidar_ref)
            lidar_refs['global'] = (None, None, lidar_depth)
            print(f"Using global LiDAR reference: {lidar_depth}")

    # Initialize model
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # Read video frames
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    total_frames = len(frames)
    chunk_size = args.chunk_size
    overlap = args.overlap_frames
    print(f"Processing {total_frames} frames in chunks of {chunk_size} with {overlap} frame overlap")

    # Step 1: Compute global scale factors for all LiDAR references
    global_scales = compute_global_scale_factors(frames, lidar_refs, video_depth_anything, target_fps, args)

    # Step 2: Create smooth interpolation across entire video
    global_scale_interpolation = interpolate_global_scales(total_frames, global_scales)

    print(f"Global scale interpolation: min={global_scale_interpolation.min():.4f}, max={global_scale_interpolation.max():.4f}")

    # Debug: Print scale factors for verification
    print("\nScale factor progression:")
    for i in range(0, min(total_frames, 200), 20):
        print(f"Frame {i}: scale={global_scale_interpolation[i]:.4f}")

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

    # Step 3: Process video in chunks using pre-computed scales
    all_depths = []
    prev_frame = None
    prev_depth = None

    for start_idx in range(0, total_frames, chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_frames = frames[start_idx:end_idx]
        
        if prev_frame is not None and start_idx > 0:
            # Use numpy concatenation instead of list conversion
            prev_frame_expanded = np.expand_dims(prev_frame, axis=0)
            chunk_frames = np.concatenate([prev_frame_expanded, chunk_frames], axis=0)
            chunk_start_global_idx = start_idx - 1
        else:
            chunk_start_global_idx = start_idx
        
        print(f"Processing chunk {start_idx}-{end_idx} (global start: {chunk_start_global_idx})")
        
        # Get pre-computed scale factors for this chunk
        chunk_scales = global_scale_interpolation[chunk_start_global_idx:chunk_start_global_idx + len(chunk_frames)]
        
        # Process chunk with pre-computed scales
        scaled_depths, refined_depths, prev_frame, prev_depth = process_chunk_with_scales(
            chunk_frames,
            chunk_scales,
            video_depth_anything,
            target_fps,
            args,
            prev_depth
        )
        
        # Remove overlap for non-first chunks
        if start_idx > 0:
            scaled_depths = scaled_depths[1:]
            refined_depths = refined_depths[1:] if args.temporal_refine else scaled_depths
            chunk_frames = chunk_frames[1:]
        
        current_depths = refined_depths if args.temporal_refine else scaled_depths
        all_depths.extend(current_depths)
        
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
                import matplotlib.cm as cm
                colormap = cm.get_cmap("inferno")
                depth_vis = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            
            depth_writer.append_data(depth_vis)
        
        del chunk_frames, scaled_depths, refined_depths, current_depths
        torch.cuda.empty_cache()
        gc.collect()

    src_writer.close()
    depth_writer.close()

    # Debug: Print final depth values for verification
    print("\nFinal depth values at specific pixels:")
    for i in range(0, min(len(all_depths), 200), 20):
        pixel_depth = all_depths[i][100, 100]
        scale_factor = global_scale_interpolation[i]
        if i in lidar_refs:
            print(f"Frame {i}: depth={pixel_depth:.2f}m, scale={scale_factor:.4f} (LiDAR reference)")
        else:
            print(f"Frame {i}: depth={pixel_depth:.2f}m, scale={scale_factor:.4f}")

    # Save NPY/EXR with global scaling applied
    if args.save_npz:
        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npy')
        os.makedirs(depth_npz_dir, exist_ok=True)
        
        for i, depth in enumerate(all_depths):
            output_npy = f"{depth_npz_dir}/frame_{i:05d}.npy"
            np.save(output_npy, depth)  # Depth already has global scaling applied

    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        
        try:
            import OpenEXR
            import Imath
            
            for i, depth in enumerate(all_depths):
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()
        except ImportError:
            print("OpenEXR not available, skipping EXR export")

    print(f"Processing complete! Output saved to {args.output_dir}")

