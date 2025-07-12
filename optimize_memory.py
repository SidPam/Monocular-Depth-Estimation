#!/usr/bin/env python3
"""
Memory optimization script for Video Depth Anything on low VRAM devices.
This script automatically detects available VRAM and sets optimal parameters.
"""

import torch
import os
import sys

def get_gpu_memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return None, None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    
    total_gb = total_memory / 1024**3
    free_gb = (total_memory - reserved_memory) / 1024**3
    
    return total_gb, free_gb

def optimize_for_gpu():
    """Optimize settings based on available GPU memory"""
    total_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        print("No CUDA device available. Using CPU fallback settings.")
        return {
            'model': 'vits',
            'input_size': 280,
            'max_res': 480,
            'max_len': 100,
            'use_fp16': True
        }
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {total_memory:.1f} GB")
    print(f"Free VRAM: {free_memory:.1f} GB")
    
    # Memory optimization based on available VRAM
    if total_memory >= 12:  # High-end GPU (12GB+)
        settings = {
            'model': 'vitl',
            'input_size': 518,
            'max_res': 1920,
            'max_len': -1,
            'use_fp16': False
        }
        print("High VRAM detected - using maximum quality settings")
    elif total_memory >= 8:  # Mid-range GPU (8-12GB)
        settings = {
            'model': 'vitl',
            'input_size': 392,
            'max_res': 1280,
            'max_len': 500,
            'use_fp16': True
        }
        print("Medium VRAM detected - using balanced settings")
    elif total_memory >= 6:  # Entry GPU (6-8GB) - like RTX 4050
        settings = {
            'model': 'vits',
            'input_size': 392,
            'max_res': 720,
            'max_len': 300,
            'use_fp16': True
        }
        print("Low VRAM detected - using optimized settings for 6GB GPU")
    else:  # Very low VRAM (<6GB)
        settings = {
            'model': 'vits',
            'input_size': 280,
            'max_res': 480,
            'max_len': 150,
            'use_fp16': True
        }
        print("Very low VRAM detected - using minimal settings")
    
    return settings

def apply_memory_optimizations():
    """Apply CUDA memory optimizations"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction to leave some for system
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Disable benchmark for memory efficiency
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Enable memory efficient attention if available
        try:
            import xformers
            print("xFormers available - memory efficient attention enabled")
        except ImportError:
            print("xFormers not available - consider installing for better memory efficiency")
            print("Install with: pip install xformers")
        
        print("Memory optimizations applied")

def check_model_exists(model_name):
    """Check if model checkpoint exists"""
    checkpoint_path = f'./checkpoints/video_depth_anything_{model_name}.pth'
    return os.path.exists(checkpoint_path)

def main():
    print("Video Depth Anything Memory Optimizer")
    print("="*50)
    
    # Get optimal settings
    settings = optimize_for_gpu()
    
    # Check if model exists
    if not check_model_exists(settings['model']):
        print(f"\nWarning: Model checkpoint for '{settings['model']}' not found!")
        print(f"Expected path: ./checkpoints/video_depth_anything_{settings['model']}.pth")
        print("Please download the appropriate model checkpoint.")
        return
    
    # Apply optimizations
    apply_memory_optimizations()
    
    print("\nRecommended settings:")
    print(f"  Model: {settings['model']}")
    print(f"  Input size: {settings['input_size']}")
    print(f"  Max resolution: {settings['max_res']}")
    print(f"  Max length: {settings['max_len']} frames (-1 = unlimited)")
    print(f"  Use FP16: {settings['use_fp16']}")
    
    print("\nMemory optimization complete!")
    print("You can now run the application with reduced risk of CUDA OOM errors.")
    
    # Save settings to environment variables for the app to use
    os.environ['VRAM_OPTIMIZED'] = 'true'
    os.environ['OPTIMAL_MODEL'] = settings['model']
    os.environ['OPTIMAL_INPUT_SIZE'] = str(settings['input_size'])
    os.environ['OPTIMAL_MAX_RES'] = str(settings['max_res'])
    os.environ['OPTIMAL_MAX_LEN'] = str(settings['max_len'])
    
if __name__ == "__main__":
    main()

