"""
AI Compliance Agent - Source Package

This package provides an AI-powered compliance agent built with:
- CrewAI for multi-agent orchestration
- LangChain/LangGraph for workflow management
- Qdrant for vector storage and retrieval
- AMD ROCm-compatible ML backends (transformers, optimum-amd, vllm)

AMD ROCm Compatibility:
- This package is optimized for AMD GPUs via ROCm
- Requires PyTorch with ROCm support installed
- HIP_VISIBLE_DEVICES environment variable controls GPU visibility
- optimum-amd provides AMD-specific performance optimizations
"""

__version__ = "0.1.0"
__author__ = "AI Compliance Team"

# AMD ROCm Configuration
# When running on AMD hardware, ensure the following:
# 1. PyTorch is installed with ROCm support
# 2. HIP_VISIBLE_DEVICES is set to appropriate GPU IDs
# 3. optimum-amd is used for transformer optimizations

import os

# Auto-detect and configure AMD GPU if available
def _configure_amd_gpu():
    """Configure AMD ROCm settings if running on AMD hardware."""
    try:
        # Check if rocm-smi is available (indicates AMD GPU system)
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            # AMD GPU detected, set HIP_VISIBLE_DEVICES if not already set
            if "HIP_VISIBLE_DEVICES" not in os.environ:
                os.environ["HIP_VISIBLE_DEVICES"] = "0"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Not an AMD system or rocm-smi not available
        pass

_configure_amd_gpu()
