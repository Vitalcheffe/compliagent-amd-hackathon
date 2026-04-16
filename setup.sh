#!/bin/bash
# =============================================================================
# CompliAgent Setup Script - AMD ROCm Optimized
# 🏆 AMD Developer Hackathon 2026 - AI Compliance Agent
# =============================================================================
# This script:
#   1. Checks for AMD ROCm availability (rocm-smi command)
#   2. Installs system dependencies if running on AMD Cloud
#   3. Sets HIP_VISIBLE_DEVICES environment variable for AMD GPUs
#   4. Creates virtual environment and installs Python dependencies
# =============================================================================

set -e  # Exit on error

echo "🚀 CompliAgent Setup - AMD ROCm Edition"
echo "========================================"

# -----------------------------------------------------------------------------
# Step 1: Check for AMD ROCm availability
# -----------------------------------------------------------------------------
echo ""
echo "🔍 Checking for AMD ROCm installation..."

ROCM_AVAILABLE=false
AMD_CLOUD=false

# Check if rocm-smi command exists (ROCm System Management Interface)
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected: rocm-smi command found"
    ROCM_AVAILABLE=true
    
    # Check if running on AMD Cloud by looking for AMD-specific paths
    if [ -d "/opt/rocm" ] || [ -d "/usr/lib/rocm" ]; then
        echo "✅ AMD Cloud environment detected"
        AMD_CLOUD=true
    fi
else
    echo "⚠️  ROCm not detected: rocm-smi command not found"
    echo "   Will proceed with CPU fallback mode or install ROCm deps if possible"
fi

# -----------------------------------------------------------------------------
# Step 2: Install system dependencies (if on AMD Cloud with sudo access)
# -----------------------------------------------------------------------------
if [ "$AMD_CLOUD" = true ]; then
    echo ""
    echo "📦 Installing AMD ROCm system dependencies..."
    
    # Check if we have sudo access
    if command -v sudo &> /dev/null; then
        # Update package list
        sudo apt-get update -qq || true
        
        # Install ROCm libraries (common packages for MI300X)
        sudo apt-get install -y \
            rocm-libs \
            hipblas \
            rccl \
            roctracer-dev \
            hipfft-dev \
            hipsparse-dev \
            hipsolver-dev \
            || echo "⚠️  Some ROCm packages may already be installed or unavailable"
        
        echo "✅ ROCm system libraries installed/verified"
    else
        echo "⚠️  No sudo access - skipping system package installation"
        echo "   Assuming ROCm is pre-installed on AMD Cloud instance"
    fi
else
    echo ""
    echo "ℹ️  Not running on AMD Cloud - skipping system dependency installation"
fi

# -----------------------------------------------------------------------------
# Step 3: Set AMD GPU environment variables
# -----------------------------------------------------------------------------
echo ""
echo "🔧 Configuring AMD GPU environment variables..."

# Create .env file for persistent configuration
ENV_FILE=".env"

# Set HIP_VISIBLE_DEVICES for AMD GPU visibility
# This tells ROCm which GPU(s) to use (0 = first GPU)
if [ "$ROCM_AVAILABLE" = true ]; then
    echo "export HIP_VISIBLE_DEVICES=0" >> "$ENV_FILE"
    echo "✅ Set HIP_VISIBLE_DEVICES=0 in .env"
    
    # Set HSA_OVERRIDE_GFX_VERSION for MI300X compatibility
    # MI300X uses GFX version 10.3.0
    echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" >> "$ENV_FILE"
    echo "✅ Set HSA_OVERRIDE_GFX_VERSION=10.3.0 in .env"
    
    # Export for current session
    export HIP_VISIBLE_DEVICES=0
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    
    echo ""
    echo "🎯 AMD GPU Configuration:"
    echo "   - HIP_VISIBLE_DEVICES=0 (using GPU 0)"
    echo "   - HSA_OVERRIDE_GFX_VERSION=10.3.0 (MI300X compatible)"
else
    echo "⚠️  ROCm not available - setting CPU fallback mode"
    echo "export HIP_VISIBLE_DEVICES=" >> "$ENV_FILE"
    echo "   Running in CPU fallback mode"
fi

# -----------------------------------------------------------------------------
# Step 4: Create virtual environment and install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "🐍 Setting up Python virtual environment..."

# Check if venv already exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created: venv/"
else
    echo "✅ Virtual environment already exists: venv/"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch with ROCm support (if ROCm is available)
if [ "$ROCM_AVAILABLE" = true ]; then
    echo ""
    echo "🔥 Installing PyTorch with ROCm support..."
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0 -q || \
    pip install torch --index-url https://download.pytorch.org/whl/rocm5.7 -q || \
    echo "⚠️  Using default PyTorch (may not have ROCm support)"
fi

# Install project dependencies
echo ""
echo "📦 Installing project dependencies from requirements.txt..."
pip install -r requirements.txt -q

echo ""
echo "✅ All Python dependencies installed successfully"

# -----------------------------------------------------------------------------
# Step 5: Final verification and success message
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
if [ "$ROCM_AVAILABLE" = true ] && [ "$AMD_CLOUD" = true ]; then
    echo "✅ ✅ ✅  READY FOR AMD MI300X  ✅ ✅ ✅"
    echo ""
    echo "🎉 Setup complete! Your environment is optimized for AMD ROCm."
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv: source venv/bin/activate"
    echo "  2. Run the agent: python src/main.py --demo"
    echo "  3. Launch demo UI: python app.py"
    echo ""
    echo "GPU Info:"
    if command -v rocm-smi &> /dev/null; then
        rocm-smi --showproductname 2>/dev/null || echo "   (Run 'rocm-smi' for GPU details)"
    fi
else
    echo "⚠️  RUNNING IN CPU FALLBACK MODE"
    echo ""
    echo "🔧 Setup complete, but ROCm was not detected."
    echo "   The application will run on CPU or use CUDA if available."
    echo ""
    echo "To enable AMD ROCm:"
    echo "  - Deploy to AMD Developer Cloud with MI300X instances"
    echo "  - Ensure ROCm 6.0+ is installed"
    echo "  - Re-run this setup script"
fi

echo ""
echo "========================================"
echo "🏆 CompliAgent - AMD Developer Hackathon 2026"
echo "========================================"
