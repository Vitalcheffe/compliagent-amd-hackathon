"""
CompliAgent Utilities Module
🏆 AMD Developer Hackathon 2026

Common utility functions:
- Logging setup with AMD GPU info
- GPU detection and monitoring
- File I/O helpers
- Error handling decorators

AMD ROCm Utilities:
    - get_gpu_info(): Detect AMD GPUs via rocm-smi
    - monitor_gpu_usage(): Real-time GPU utilization tracking
    - setup_logging(): Configure logging with GPU context
"""

import logging
import sys
from typing import Optional, Dict, Any


def setup_logging(level: str = "INFO", include_gpu_info: bool = True) -> logging.Logger:
    """
    Set up application logging with optional AMD GPU information.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        include_gpu_info: Whether to log AMD GPU detection status
        
    Returns:
        Configured logger instance
        
    AMD ROCm Note:
        When running on AMD Cloud, this will log GPU detection
        and HIP_VISIBLE_DEVICES configuration.
    """
    # Create logger
    logger = logging.getLogger("compliagent")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    # Log AMD GPU info if requested
    if include_gpu_info:
        gpu_info = get_gpu_info()
        if gpu_info.get("available"):
            logger.info(f"✅ AMD ROCm detected: {gpu_info.get('output', 'GPU available')}")
            logger.info(f"   HIP_VISIBLE_DEVICES={gpu_info.get('hip_visible_devices', 'Not set')}")
            logger.info(f"   HSA_OVERRIDE_GFX_VERSION={gpu_info.get('hsa_version', 'Not set')}")
        else:
            logger.warning("⚠️  Running in CPU fallback mode (ROCm not detected)")
    
    return logger


def get_gpu_info() -> Dict[str, Any]:
    """
    Get AMD GPU information if available.
    
    Returns:
        dict with GPU details:
        - available: bool indicating if ROCm is available
        - output: rocm-smi product name output
        - hip_visible_devices: Current HIP_VISIBLE_DEVICES value
        - hsa_version: HSA_OVERRIDE_GFX_VERSION value
        
    AMD ROCm Note:
        Uses rocm-smi command to detect AMD GPUs.
        Returns available=False if ROCm is not installed.
    """
    import subprocess
    import os
    
    result = {
        "available": False,
        "output": None,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", "Not set"),
        "hsa_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION", "Not set")
    }
    
    try:
        # Try rocm-smi for GPU detection
        smi_result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if smi_result.returncode == 0:
            result["available"] = True
            result["output"] = smi_result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return result


def ensure_dirs(*paths: str) -> None:
    """
    Ensure directories exist, creating them if necessary.
    
    Args:
        paths: One or more directory paths to create
    """
    import os
    
    for path in paths:
        os.makedirs(path, exist_ok=True)


__all__ = ["setup_logging", "get_gpu_info", "ensure_dirs"]
