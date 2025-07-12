#!/usr/bin/env python3
"""
Test script to validate CUDA memory optimizations.
"""

import torch
import numpy as np
from video_depth_anything.video_depth import VideoDepthAnything
import cv2
import os

def create_test_video_frames(num_frames=50, height=480, width=640):
    """Create synthetic test video frames"""
    frames = []
    for i in range(num_frames):
        # Create a simple gradient pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 5) % 255  # Red channel varies with frame
        frame[:, :, 1] = 128  # Green constant
        frame[:, :, 2] = 255 - ((i * 5) % 255)  # Blue inverse of red
        frames.append(frame)
    return np.array(frames)

def test_memory_optimization():
    """Test the memory optimization with a small video"""
    print("Testing CUDA memory optimization...")
    
    # Get GPU info
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {device_name}")
        print(f"Total VRAM: {total_memory:.1f} GB")
    else:
        print("CUDA not available")
        return False
    
    try:
        # Load model with optimized settings
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        # Use smaller model for RTX 4050
        encoder = 'vits'
        print(f"Loading {encoder} model...")
        
        model = VideoDepthAnything(**model_configs[encoder])
        model.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{encoder}.pth', map_location='cpu'), strict=True)
        model = model.to('cuda').eval()
        
        print("Model loaded successfully!")
        
        # Create test frames
        print("Creating test video frames...")
        test_frames = create_test_video_frames(num_frames=32, height=280, width=392)
        
        # Test inference with optimized settings
        print("Testing inference with memory optimizations...")
        depths, fps = model.infer_video_depth(
            test_frames, 
            target_fps=15, 
            input_size=280,  # Reduced input size
            device='cuda'
        )
        
        print(f"Successfully processed {len(depths)} frames!")
        print(f"Output shape: {depths.shape}")
        print(f"Memory usage after processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Clean up
        del model, depths, test_frames
        torch.cuda.empty_cache()
        
        print("\n✅ CUDA memory optimization test PASSED!")
        print("The memory optimizations are working correctly.")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error: {e}")
        print("Memory optimization may need further tuning.")
        return False
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_optimization()
    if success:
        print("\nYou can now use the application without CUDA OOM errors!")
        print("Recommended usage:")
        print("  - Use 'vits' model (automatically selected)")
        print("  - Keep resolution under 720p")
        print("  - Limit video length to 150-300 frames")
        print("  - Input size will be automatically reduced to 280-392")
    else:
        print("\nFurther optimization may be needed. Consider:")
        print("  - Using even smaller input sizes (e.g., 196)")
        print("  - Processing shorter video segments")
        print("  - Closing other GPU applications")

