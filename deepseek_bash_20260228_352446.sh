#!/bin/bash
# scripts/optimize_gpu.sh - GPU optimization for maximum performance

echo "========================================="
echo "GPU Optimization for Cup Tracker Pro"
echo "========================================="

# NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - applying optimizations"
    
    # Enable persistence mode
    sudo nvidia-smi -pm 1
    
    # Set power limit to maximum
    sudo nvidia-smi -pl 250
    
    # Set compute mode to default
    sudo nvidia-smi -c 0
    
    # Enable GPU statistics
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_CACHE_DISABLE=0
    export CUDA_CACHE_MAXSIZE=1073741824
    
    # Set compute capabilities
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
    
    # Enable Tensor Cores for mixed precision
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
    
    echo "✓ NVIDIA optimizations applied"
fi

# AMD GPU
if command -v rocminfo &> /dev/null; then
    echo "AMD GPU detected - applying optimizations"
    
    # Set ROCm options
    export ROCM_PATH=/opt/rocm
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/tensile
    
    # Enable HIP optimizations
    export HIP_VISIBLE_DEVICES=0
    export HIP_LAUNCH_BLOCKING=0
    
    echo "✓ AMD optimizations applied"
fi

# Intel GPU
if command -v intel_gpu_top &> /dev/null; then
    echo "Intel GPU detected - applying optimizations"
    
    # Enable performance mode
    echo performance | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
    
    # Set GPU frequency
    echo 1550 | sudo tee /sys/class/drm/card0/gt_max_freq_mhz
    
    echo "✓ Intel optimizations applied"
fi

# OpenCL optimizations
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors
export PYOPENCL_CTX='0'

# PyTorch optimizations
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

# Enable CUDA graphs for PyTorch
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set memory allocation strategy
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072

echo "✓ General GPU optimizations applied"

# Test GPU performance
echo ""
echo "Testing GPU performance..."
python -c "
import torch
import time

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'GPU: {torch.cuda.get_device_name(device)}')
    
    # Benchmark
    a = torch.randn(5000, 5000).cuda()
    b = torch.randn(5000, 5000).cuda()
    
    start = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f'Matrix multiply (10x): {end-start:.3f}s')
    print(f'Memory: {torch.cuda.memory_allocated(device)/1024**2:.0f}MB allocated')
"