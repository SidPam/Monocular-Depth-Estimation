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
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False, batch_size=16, use_cpu_offload=False):
        # Clear cache before processing
        if device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        
        # Memory optimization for limited VRAM (without reducing input size)
        if device == 'cuda':
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if available_memory < 6:  # Less than 6GB VRAM (RTX 4050)
                batch_size = 1
                print(f"Low VRAM detected ({available_memory:.1f}GB): maintaining input_size={input_size}, reducing batch_size to 1")
            elif available_memory < 8:  # Less than 8GB VRAM
                batch_size = min(batch_size, 2)
                print(f"Limited VRAM detected ({available_memory:.1f}GB): maintaining input_size={input_size}, reducing batch_size to {batch_size}")
        
        # Ensure input_size is compatible with the model (multiple of 14)
        input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(len(frames))]
        
        # Use smaller batch size for memory optimization
        effective_infer_len = min(INFER_LEN, batch_size)
        effective_overlap = min(OVERLAP, effective_infer_len // 2)
        
        frame_step = effective_infer_len - effective_overlap
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (effective_infer_len - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None
        
        # CPU offloading setup
        if use_cpu_offload:
            self.cpu()
        
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            # Clear cache before each batch
            if device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                
            cur_list = []
            for i in range(effective_infer_len):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1)
            
            # Move to device only when processing
            if use_cpu_offload:
                self.to(device)
            cur_input = cur_input.to(device)
            
            try:
                if pre_input is not None and effective_overlap > 0:
                    # Adjust keyframes for smaller batch size
                    effective_keyframes = [k for k in KEYFRAMES if k < effective_infer_len][:effective_overlap]
                    if len(effective_keyframes) > 0:
                        cur_input[:, :len(effective_keyframes), ...] = pre_input[:, effective_keyframes, ...]

                with torch.no_grad():
                    with torch.autocast(device_type=device, enabled=(not fp32)):
                        depth = self.forward(cur_input) # depth shape: [1, T, H, W]

                depth = depth.to(cur_input.dtype)
                depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

                pre_input = cur_input.detach()
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at frame {frame_id}, trying with reduced batch processing...")
                # Fallback: process with even smaller chunks or frame-by-frame
                torch.cuda.empty_cache()
                gc.collect()
                
                # Try processing in smaller sub-batches first
                chunk_depths = []
                try:
                    # Try processing in chunks of 4
                    chunk_size = min(4, effective_infer_len)
                    for chunk_start in range(0, effective_infer_len, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, effective_infer_len)
                        chunk_input = torch.cat(cur_list[chunk_start:chunk_end], dim=1).to(device)
                        
                        with torch.no_grad():
                            with torch.autocast(device_type=device, enabled=(not fp32)):
                                chunk_depth = self.forward(chunk_input)
                        
                        chunk_depth = chunk_depth.to(chunk_input.dtype)
                        chunk_depth = F.interpolate(chunk_depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                        chunk_depths += [chunk_depth[i][0].cpu().numpy() for i in range(chunk_depth.shape[0])]
                        
                        del chunk_input, chunk_depth
                        torch.cuda.empty_cache()
                        
                except torch.cuda.OutOfMemoryError:
                    print("Still OOM with small chunks, falling back to frame-by-frame processing...")
                    # Last resort: process frames individually
                    chunk_depths = []
                    for i in range(effective_infer_len):
                        try:
                            single_frame = cur_list[i].to(device)
                            with torch.no_grad():
                                with torch.autocast(device_type=device, enabled=(not fp32)):
                                    single_depth = self.forward(single_frame)
                            single_depth = single_depth.to(single_frame.dtype)
                            single_depth = F.interpolate(single_depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                            chunk_depths.append(single_depth[0][0].cpu().numpy())
                            del single_frame, single_depth
                            torch.cuda.empty_cache()
                        except torch.cuda.OutOfMemoryError:
                            print(f"Critical OOM at frame {frame_id + i}, using zero placeholder")
                            # Create a zero depth map as placeholder
                            placeholder = np.zeros((frame_height, frame_width), dtype=np.float32)
                            chunk_depths.append(placeholder)
                            torch.cuda.empty_cache()
                
                depth_list += chunk_depths
                pre_input = None  # Reset for next iteration
                
            # Move model back to CPU if using offloading
            if use_cpu_offload:
                self.cpu()
                
            # Delete intermediate tensors
            del cur_input
            if 'depth' in locals():
                del depth
            if device == 'cuda':
                torch.cuda.empty_cache()

        del frame_list
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        # Move model back to device if using CPU offloading
        if use_cpu_offload:
            self.to(device)

        # Simplified processing for memory-constrained scenarios
        if effective_infer_len < INFER_LEN or batch_size < INFER_LEN:
            # Skip complex alignment for memory optimization
            depth_list_aligned = depth_list
        else:
            # Use original alignment logic for full batch processing
            depth_list_aligned = []
            ref_align = []
            align_len = OVERLAP - INTERP_LEN
            kf_align_list = KEYFRAMES[:align_len]

            for frame_id in range(0, len(depth_list), INFER_LEN):
                if len(depth_list_aligned) == 0:
                    depth_list_aligned += depth_list[:INFER_LEN]
                    for kf_id in kf_align_list:
                        ref_align.append(depth_list[frame_id+kf_id])
                else:
                    curr_align = []
                    for i in range(len(kf_align_list)):
                        curr_align.append(depth_list[frame_id+i])
                        
                    scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                           np.concatenate(ref_align),
                                                           np.concatenate(np.ones_like(ref_align)==1))

                    pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                    post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                    for i in range(len(post_depth_list)):
                        post_depth_list[i] = post_depth_list[i] * scale + shift
                        post_depth_list[i][post_depth_list[i]<0] = 0
                    depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                    for i in range(OVERLAP, INFER_LEN):
                        new_depth = depth_list[frame_id+i] * scale + shift
                        new_depth[new_depth<0] = 0
                        depth_list_aligned.append(new_depth)

                    ref_align = ref_align[:1]
                    for kf_id in kf_align_list[1:]:
                        new_depth = depth_list[frame_id+kf_id] * scale + shift
                        new_depth[new_depth<0] = 0
                        ref_align.append(new_depth)
        
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
        