import argparse
import numpy as np
import os
import torch
import gc
import cv2
from scipy.interpolate import interp1d

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

class GlobalDepthState:
    def __init__(self):
        self.global_scale_factors = {}
        self.last_valid_scale = 1.0
        
    def update_scale_factor(self, global_frame_idx, scale_factor):
        self.global_scale_factors[global_frame_idx] = scale_factor
        self.last_valid_scale = scale_factor
    
    def get_interpolated_scale(self, global_frame_idx):
        if global_frame_idx in self.global_scale_factors:
            return self.global_scale_factors[global_frame_idx]
        
        # Find nearest valid scale factors
        valid_indices = sorted(self.global_scale_factors.keys())
        if not valid_indices:
            return self.last_valid_scale
            
        # Linear interpolation between nearest references
        before = [i for i in valid_indices if i <= global_frame_idx]
        after = [i for i in valid_indices if i > global_frame_idx]
        
        if before and after:
            before_idx, after_idx = before[-1], after[0]
            before_scale = self.global_scale_factors[before_idx]
            after_scale = self.global_scale_factors[after_idx]
            weight = (global_frame_idx - before_idx) / (after_idx - before_idx)
            return before_scale + weight * (after_scale - before_scale)
        elif before:
            return self.global_scale_factors[before[-1]]
        else:
            return self.global_scale_factors[after[0]]

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

def apply_scaling_and_refinement(depths, frames, lidar_refs, temporal_refine, prev_depth=None, start_global_idx=0, global_state=None):
    scale_factors = np.ones(len(depths))
    
    # Handle global frame indexing for LiDAR references
    for i, depth in enumerate(depths):
        global_frame_idx = start_global_idx + i
        ref = lidar_refs.get(global_frame_idx, lidar_refs.get('global', None))
        
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
                scale_factors[i] = scale_factor
                # Update global state
                if global_state:
                    global_state.update_scale_factor(global_frame_idx, scale_factor)
                    print(f"Frame {global_frame_idx}: LiDAR={lidar_depth:.2f}, Pred={pred_depth:.2f}, Scale={scale_factor:.4f}")
        else:
            # Use interpolated scale from global state
            if global_state:
                scale_factors[i] = global_state.get_interpolated_scale(global_frame_idx)
                print(f"Frame {global_frame_idx}: Interpolated scale={scale_factors[i]:.4f}")
    
    # Apply scaling
    scaled_depths = [depth * scale_factors[i] for i, depth in enumerate(depths)]
    
    # Temporal refinement
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

def process_chunk(frames_chunk, video_depth_anything, target_fps, args, lidar_refs, prev_depth=None, start_global_idx=0, global_state=None):
    depths_chunk, _ = video_depth_anything.infer_video_depth(
        frames_chunk, target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
        batch_size=batch_size,
        use_cpu_offload=use_cpu_offload
    )
    
    scaled_depths, refined_depths = apply_scaling_and_refinement(
        depths_chunk,
        frames_chunk,
        lidar_refs,
        args.temporal_refine,
        prev_depth,
        start_global_idx,  # Pass global frame index
        global_state       # Pass global state
    )
    
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

    # Initialize global state
    global_state = GlobalDepthState()

    # Pre-populate global state with all LiDAR references
    for frame_idx, ref in lidar_refs.items():
        if frame_idx != 'global' and isinstance(frame_idx, int):
            global_state.global_scale_factors[frame_idx] = 1.0

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

    all_depths = []
    prev_frame = None
    prev_depth = None

    # Process video in chunks
    for start_idx in range(0, total_frames, chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_frames = frames[start_idx:end_idx]
        
        if prev_frame is not None and start_idx > 0:
            # Use numpy concatenation instead of list conversion
            prev_frame_expanded = np.expand_dims(prev_frame, axis=0)
            chunk_frames = np.concatenate([prev_frame_expanded, chunk_frames], axis=0)
            chunk_start_global_idx = start_idx - 1  # Account for overlap
        else:
            chunk_start_global_idx = start_idx
        
        print(f"Processing chunk {start_idx}-{end_idx} (global start: {chunk_start_global_idx})")
        
        # Process chunk with global indexing
        scaled_depths, refined_depths, prev_frame, prev_depth = process_chunk(
            chunk_frames,
            video_depth_anything,
            target_fps,
            args,
            lidar_refs,
            prev_depth,
            chunk_start_global_idx,  # Pass global start index
            global_state             # Pass global state
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
    print("\nFinal depth values check:")
    for i in range(min(10, len(all_depths))):
        if i in lidar_refs:
            print(f"Frame {i}: {all_depths[i][100, 100]:.2f} (LiDAR reference frame)")
        else:
            print(f"Frame {i}: {all_depths[i][100, 100]:.2f}")

    # Save NPY/EXR if requested
    if args.save_npz:
        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npy')
        os.makedirs(depth_npz_dir, exist_ok=True)
        
        for i, depth in enumerate(all_depths):
            output_npy = f"{depth_npz_dir}/frame_{i:05d}.npy"
            np.save(output_npy, depth)

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
