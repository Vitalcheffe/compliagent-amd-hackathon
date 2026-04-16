"""
CompliAgent Configuration Module
🏆 AMD Developer Hackathon 2026

Centralized configuration management:
- Environment variable loading via python-dotenv
- AMD ROCm settings (HIP_VISIBLE_DEVICES, HSA_OVERRIDE_GFX_VERSION)
- LLM model configurations for MI300X
- Vector database (Qdrant) settings

AMD ROCm Configuration:
    The following environment variables are used:
    - HIP_VISIBLE_DEVICES: Controls GPU visibility (default: 0)
    - HSA_OVERRIDE_GFX_VERSION: GFX version for MI300X (10.3.0)
    - VLLM_USE_ROCM: Enable vLLM ROCm backend (1)
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with AMD ROCm support."""
    
    # === AMD ROCm Settings ===
    hip_visible_devices: str = "0"
    hsa_override_gfx_version: str = "10.3.0"
    vllm_use_rocm: bool = True
    
    # === LLM Settings ===
    llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_model_70b: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.1
    
    # === Vector Database (Qdrant) ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "compliance_docs"
    
    # === Regulatory Sources ===
    regulatory_sources: dict = {
        "AMF": "https://www.amf-france.org",
        "ECB": "https://www.ecb.europa.eu",
        "SEC": "https://www.sec.gov",
        "ESMA": "https://www.esma.europa.eu"
    }
    
    # === Paths ===
    data_dir: str = "data"
    output_dir: str = "output"
    cache_dir: str = ".cache"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-detect AMD Cloud environment
        if os.path.exists("/opt/rocm") or os.path.exists("/usr/lib/rocm"):
            # Running on AMD Cloud - ensure proper GPU settings
            self.hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "0")
            self.hsa_override_gfx_version = os.environ.get(
                "HSA_OVERRIDE_GFX_VERSION", "10.3.0"
            )


# Global settings instance
settings = Settings()

__all__ = ["Settings", "settings"]
