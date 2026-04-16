#!/bin/bash
# Setup script for AI Compliance Agent with AMD ROCm compatibility
# This script installs dependencies, checks ROCm availability, and configures GPU settings

set -e  # Exit on error

echo "=========================================="
echo "AI Compliance Agent - Setup Script"
echo "=========================================="

# Step 1: Install Python dependencies
echo "[1/3] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Note: PyTorch with ROCm support should be installed separately if not already present
# pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Step 2: Check ROCm availability
echo "[2/3] Checking ROCm availability..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ ROCm detected! Running rocm-smi to check GPU status..."
    rocm-smi
    
    # Step 3: Configure HIP_VISIBLE_DEVICES for AMD Cloud
    echo "[3/3] Configuring HIP_VISIBLE_DEVICES for AMD GPU..."
    
    # Get available GPU IDs (assuming all detected GPUs should be visible)
    # In production, you may want to filter specific GPU IDs based on your setup
    GPU_IDS=$(rocm-smi --showid 2>/dev/null | grep -oP '\d+' | head -n 1 || echo "0")
    
    echo "Setting HIP_VISIBLE_DEVICES=$GPU_IDS"
    export HIP_VISIBLE_DEVICES=$GPU_IDS
    
    # Add to .bashrc for persistence (optional, uncomment if needed)
    # echo "export HIP_VISIBLE_DEVICES=$GPU_IDS" >> ~/.bashrc
    
    echo "✓ HIP_VISIBLE_DEVICES configured successfully!"
    echo "  Current value: $HIP_VISIBLE_DEVICES"
else
    echo "⚠ ROCm not detected (rocm-smi command not found)"
    echo "  If running on AMD hardware, ensure ROCm drivers are installed."
    echo "  For AMD Cloud instances, ROCm should be pre-installed."
    echo ""
    echo "  To install ROCm on supported systems:"
    echo "    Ubuntu/Debian: sudo apt install rocm-smi-lib"
    echo "    Or follow: https://rocm.docs.amd.com/projects/install-on-linux/"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a .env file with your API keys and configuration"
echo "2. Run: python src/main.py"
echo ""
echo "AMD ROCm Notes:"
echo "- HIP_VISIBLE_DEVICES controls which GPUs are visible to the application"
echo "- optimum-amd provides AMD-specific optimizations for transformers"
echo "- vllm supports ROCm for high-performance LLM inference"
echo "- Ensure PyTorch is installed with ROCm support for GPU acceleration"
