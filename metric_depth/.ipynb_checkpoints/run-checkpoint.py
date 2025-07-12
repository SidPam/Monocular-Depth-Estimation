# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

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

#
import cv2
from scipy.interpolate import interp1d
#

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

'''
'''
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
'''
'''



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
    #parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')

    #parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    #parser.add_argument('--save_exr', action='store_true', help='save depths as exr')

    parser.add_argument('--lidar_ref', type=str, 
                   help='CSV/path with frame_idx,x,y,lidar_depth (or single value)')
    parser.add_argument('--temporal_refine', action='store_true', 
                   help='Apply temporal refinement for depth consistency')


    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Aggressive memory optimization for low VRAM GPUs
    batch_size = 16  # Default batch size
    use_cpu_offload = False
    
    if DEVICE == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name()} with {available_memory:.1f}GB VRAM")
        
        # Memory optimization settings
        torch.cuda.set_per_process_memory_fraction(0.85)  # Leave more room
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # Smaller split size
        
        # Force very conservative settings for low VRAM
        if available_memory < 6:
            batch_size = 1
            use_cpu_offload = True
            #args.input_size = min(args.input_size, 280)
            #args.max_res = min(args.max_res, 720)
            print(f"Auto-optimizing for low VRAM: batch_size=1, input_size={args.input_size}, max_res={args.max_res}, cpu_offload=True")
        elif available_memory < 8:
            batch_size = 2
            #args.input_size = min(args.input_size, 392)
            #args.max_res = min(args.max_res, 1280)
            print(f"Auto-optimizing for limited VRAM: batch_size=2, input_size={args.input_size}, max_res={args.max_res}")

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
        else:  # Single global reference
            lidar_depth = float(args.lidar_ref)
            lidar_refs['global'] = (None, None, lidar_depth)

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    # Clear cache after model loading
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32, batch_size=batch_size, use_cpu_offload=use_cpu_offload)
     # Convert to float32 to save memory
    frames = [f.astype(np.float32) for f in frames]
    depths = [d.astype(np.float32) for d in depths]


    '''
    '''
    # After: depths, fps = video_depth_anything.infer_video_depth(...)
    
    # Initialize scale_factors array
    scale_factors = np.ones(len(depths))
    
    # Compute scale factors for reference frames
    for i in range(len(depths)):
        ref = lidar_refs.get(i, lidar_refs.get('global', None))
        if ref:
            x, y, lidar_depth = ref
            if x is None or y is None:
                h, w = depths[i].shape
                x, y = w // 2, h // 2
            # Ensure coordinates are within bounds
            x = min(int(x), depths[i].shape[1] - 1)
            y = min(int(y), depths[i].shape[0] - 1)
            pred_depth = depths[i][y, x]
            if pred_depth > 0:
                scale_factors[i] = lidar_depth / pred_depth
    
    # Propagate scaling factors temporally
    valid_indices = np.where(scale_factors != 1.0)[0]
    if len(valid_indices) > 0:
        if len(valid_indices) >= 2:
            f = interp1d(valid_indices, scale_factors[valid_indices], 
                         kind='linear', fill_value="extrapolate")
            for i in range(len(scale_factors)):
                if scale_factors[i] == 1.0:
                    scale_factors[i] = f(i)
        else:
            single_scale = scale_factors[valid_indices[0]]
            scale_factors = np.ones(len(depths)) * single_scale
    
    # Apply scaling to all depths
    scaled_depths = [depth * scale_factors[i] for i, depth in enumerate(depths)]

    if args.temporal_refine:
            # Apply temporal refinement
        refined_depths = [scaled_depths[0]]
        
        for i in range(1, len(scaled_depths)):
            flow = compute_flow(frames[i-1], frames[i])
            warped_depth = warp_depth(refined_depths[-1], flow)
            blended = 0.7 * scaled_depths[i] + 0.3 * warped_depth
            refined_depths.append(blended)
        
        # Use refined_depths for saving
        final_depths = refined_depths
    else:
        final_depths = scaled_depths

   


    '''
    '''



    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    
    # Ensure frames and depths are numpy arrays for save_video function
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    if isinstance(final_depths, list):
        final_depths = np.stack(final_depths, axis=0)
    elif isinstance(depths, list):
        depths = np.stack(depths, axis=0)
    
    save_video(frames, processed_video_path, fps=fps)
    # Use final_depths if temporal refinement was applied, otherwise use depths
    depth_output = final_depths if args.temporal_refine else depths
    save_video(depth_output, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    


    '''if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)'''

    if args.save_npz:
        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npy')
        os.makedirs(depth_npz_dir, exist_ok=True)
        '''for i, depth in enumerate(depths):
            output_npy = f"{depth_npz_dir}/frame_{i:05d}.npy"
            np.save(output_npy, depth)'''
        #MODIFIED     for i, depth in enumerate(depths):
        for i, depth in enumerate(final_depths):
            scale_factor = 1.0
            # Check for frame-specific or global LiDAR reference
            ref = lidar_refs.get(i, lidar_refs.get('global', None))
            if ref:
                x, y, lidar_depth = ref
                if x is None or y is None:
                    # Use center pixel if coordinates not provided
                    h, w = depth.shape
                    x, y = w // 2, h // 2
                # Ensure coordinates are within bounds
                x = min(int(x), depth.shape[1] - 1)
                y = min(int(y), depth.shape[0] - 1)
                pred_depth = depth[y, x]
                if pred_depth > 0:
                    scale_factor = lidar_depth / pred_depth
            scaled_depth = depth * scale_factor
            output_npy = f"{depth_npz_dir}/frame_{i:05d}.npy"
            np.save(output_npy, scaled_depth)


        
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        '''for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()'''
        #MODIFIED  for i, depth in enumerate(depths):
        for i, depth in enumerate(final_depths):
            scale_factor = 1.0
            ref = lidar_refs.get(i, lidar_refs.get('global', None))
            if ref:
                x, y, lidar_depth = ref
                if x is None or y is None:
                    h, w = depth.shape
                    x, y = w // 2, h // 2
                # Ensure coordinates are within bounds
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
            # ... (rest of your EXR saving code, but use scaled_depth)


    


