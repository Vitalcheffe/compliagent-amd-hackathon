"""
CompliAgent - Autonomous Regulatory Compliance Agent
🏆 AMD Developer Hackathon 2026 Submission

A multi-agent AI system for regulatory compliance monitoring,
analysis, and reporting optimized for AMD ROCm/MI300X.

AMD ROCm Compatibility:
    This package is optimized for AMD GPUs via ROCm:
    - Uses optimum-amd for transformer optimizations
    - Supports vLLM with ROCm backend for high-throughput inference
    - Configures HIP_VISIBLE_DEVICES for GPU visibility on AMD Cloud
    
Usage:
    >>> from src import CompliAgent
    >>> agent = CompliAgent()
    >>> agent.analyze_policy("policy.pdf", regulation="AMF")

Environment Variables (set in .env or via setup.sh):
    HIP_VISIBLE_DEVICES: Controls which AMD GPUs are visible (default: 0)
    HSA_OVERRIDE_GFX_VERSION: GFX version for MI300X compatibility (10.3.0)
"""

__version__ = "0.1.0"
__author__ = "CompliAgent Team"
__hackathon__ = "AMD Developer Hackathon 2026"

# AMD ROCm Configuration
# These are set by setup.sh when running on AMD Cloud
import os

# Auto-detect and configure AMD GPU settings if on AMD Cloud
if os.path.exists("/opt/rocm") or os.path.exists("/usr/lib/rocm"):
    # Running on AMD Cloud - ensure GPU visibility
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Import main components
from src.agents import ScraperAgent, AnalystAgent, ReporterAgent
from src.config import settings
from src.utils import setup_logging

__all__ = [
    "CompliAgent",
    "ScraperAgent",
    "AnalystAgent", 
    "ReporterAgent",
    "settings",
    "setup_logging",
    "__version__",
]


class CompliAgent:
    """
    Main entry point for the CompliAgent system.
    
    Orchestrates multiple AI agents for regulatory compliance:
    1. ScraperAgent: Monitors regulatory sources (AMF, ECB, SEC, ESMA)
    2. AnalystAgent: Analyzes impact using RAG + Llama-3-70B on MI300X
    3. ReporterAgent: Generates audit-ready compliance reports
    
    Example:
        >>> agent = CompliAgent()
        >>> result = agent.analyze_policy("policy.pdf", regulation="AMF")
        >>> print(result.summary)
    """
    
    def __init__(self, config=None):
        """Initialize the CompliAgent with optional custom config."""
        self.config = config or settings
        self.scraper = None
        self.analyst = None
        self.reporter = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all agents (lazy loading)."""
        if not self._initialized:
            from src.agents import ScraperAgent, AnalystAgent, ReporterAgent
            
            self.scraper = ScraperAgent(config=self.config)
            self.analyst = AnalystAgent(config=self.config)
            self.reporter = ReporterAgent(config=self.config)
            self._initialized = True
        
        return self
    
    def analyze_policy(self, policy_path: str, regulation: str = "AMF"):
        """
        Analyze a policy document against regulatory requirements.
        
        Args:
            policy_path: Path to PDF policy document
            regulation: Regulatory body code (AMF, ECB, SEC, ESMA)
            
        Returns:
            ComplianceReport with gap analysis and recommendations
        """
        self.initialize()
        
        # Step 1: Scrape latest regulations
        regulations = self.scraper.scrape(regulation=regulation)
        
        # Step 2: Analyze policy against regulations
        analysis = self.analyst.analyze(
            policy_path=policy_path,
            regulations=regulations
        )
        
        # Step 3: Generate report
        report = self.reporter.generate(analysis=analysis)
        
        return report


def get_gpu_info():
    """
    Get AMD GPU information if available.
    
    Returns:
        dict with GPU details or None if ROCm not available
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return {
                "available": True,
                "output": result.stdout.strip(),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", "Not set")
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return {"available": False, "message": "ROCm not detected"}


# Print GPU info on import (useful for debugging)
if __name__ == "__main__":
    gpu_info = get_gpu_info()
    if gpu_info.get("available"):
        print(f"✅ AMD ROCm detected: {gpu_info['output']}")
        print(f"   HIP_VISIBLE_DEVICES={gpu_info['hip_visible_devices']}")
    else:
        print("⚠️  Running in CPU fallback mode (ROCm not detected)")
