"""
CompliAgent Agents Module
🏆 AMD Developer Hackathon 2026

Multi-agent system for regulatory compliance:
- ScraperAgent: Monitors regulatory sources (AMF, ECB, SEC, ESMA)
- AnalystAgent: Analyzes policy impact using RAG + Llama-3-70B on MI300X
- ReporterAgent: Generates audit-ready compliance reports

AMD ROCm Optimization:
    All agents leverage AMD MI300X GPUs via ROCm for:
    - Fast LLM inference with vLLM ROCm backend
    - Transformer optimizations via optimum-amd
    - Efficient vector search with Qdrant
"""

from src.agents.scraper import ScraperAgent
from src.agents.analyst import AnalystAgent
from src.agents.reporter import ReporterAgent

__all__ = ["ScraperAgent", "AnalystAgent", "ReporterAgent"]
