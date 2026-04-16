"""
ReporterAgent - Compliance Report Generator
🏆 AMD Developer Hackathon 2026

Generates audit-ready compliance reports:
- Markdown reports with executive summaries
- JSON audit trails for regulatory submissions
- Actionable remediation recommendations

AMD ROCm Note:
    Report generation runs on CPU but uses analysis results
    from GPU-accelerated AnalystAgent.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReporterAgent:
    """
    Agent responsible for generating compliance reports.
    
    Outputs:
    1. Markdown report (human-readable)
    2. JSON audit trail (machine-readable)
    3. Executive summary (for leadership)
    
    All reports include:
    - Citation tracking to source regulations
    - Gap severity ratings
    - Prioritized remediation steps
    """
    
    def __init__(self, config=None):
        """
        Initialize the ReporterAgent.
        
        Args:
            config: Application settings (uses src.config.settings if None)
        """
        from src.config import settings as default_config
        self.config = config or default_config
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReporterAgent initialized")
        logger.info(f"  Output directory: {self.config.output_dir}")
    
    def generate(self, analysis: Any) -> Dict[str, str]:
        """
        Generate compliance reports from analysis results.
        
        Args:
            analysis: AnalysisResult from AnalystAgent
            
        Returns:
            Dict with paths to generated reports:
            - markdown: Path to .md report
            - json: Path to .json audit trail
            - summary: Executive summary text
            
        Example:
            >>> reporter = ReporterAgent()
            >>> reports = reporter.generate(analysis_result)
            >>> print(f"Report: {reports['markdown']}")
        """
        logger.info("Generating compliance reports...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate Markdown report
        md_path = self._generate_markdown(analysis, timestamp)
        
        # Generate JSON audit trail
        json_path = self._generate_json(analysis, timestamp)
        
        # Generate executive summary
        summary = self._generate_summary(analysis)
        
        reports = {
            "markdown": md_path,
            "json": json_path,
            "summary": summary
        }
        
        logger.info(f"Reports generated: {len(reports)} files")
        return reports
    
    def _generate_markdown(self, analysis: Any, timestamp: str) -> str:
        """Generate Markdown compliance report."""
        filename = f"compliance_report_{timestamp}.md"
        filepath = Path(self.config.output_dir) / filename
        
        content = f"""# Compliance Analysis Report

**Generated:** {datetime.now().isoformat()}  
**Policy:** {analysis.policy_path}  
**Overall Compliance Score:** {analysis.overall_compliance_score:.1%}

---

## Executive Summary

{analysis.summary}

---

## Identified Gaps ({len(analysis.gaps)})

| Section | Issue | Severity | Regulation Reference |
|---------|-------|----------|---------------------|
"""
        
        for gap in analysis.gaps:
            severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(gap.severity, "⚪")
            content += f"| {gap.section} | {gap.issue} | {severity_emoji} {gap.severity} | {gap.regulation_ref} |\n"
        
        content += f"""
---

## Detailed Recommendations

"""
        for i, gap in enumerate(analysis.gaps, 1):
            content += f"""### {i}. {gap.section}: {gap.issue}

**Severity:** {gap.severity.upper()}  
**Regulation:** {gap.regulation_ref}

**Recommendation:** {gap.recommendation}

---

"""
        
        content += f"""## Citations ({len(analysis.citations)})

"""
        for citation in analysis.citations:
            content += f"- **{citation.get('policy_section', 'N/A')}** → {citation.get('regulation_id', 'N/A')} (relevance: {citation.get('relevance_score', 0):.2f})\n"
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Markdown report saved: {filepath}")
        return str(filepath)
    
    def _generate_json(self, analysis: Any, timestamp: str) -> str:
        """Generate JSON audit trail."""
        filename = f"audit_trail_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename
        
        # Convert dataclasses to dicts
        gaps_data = []
        for gap in analysis.gaps:
            gaps_data.append({
                "section": gap.section,
                "issue": gap.issue,
                "severity": gap.severity,
                "recommendation": gap.recommendation,
                "regulation_ref": gap.regulation_ref
            })
        
        audit_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "policy_path": analysis.policy_path,
                "tool": "CompliAgent v0.1.0",
                "hackathon": "AMD Developer Hackathon 2026"
            },
            "overall_compliance_score": analysis.overall_compliance_score,
            "summary": analysis.summary,
            "gaps": gaps_data,
            "citations": analysis.citations
        }
        
        # Write JSON with indentation
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Audit trail saved: {filepath}")
        return str(filepath)
    
    def _generate_summary(self, analysis: Any) -> str:
        """Generate executive summary text."""
        critical_gaps = sum(1 for g in analysis.gaps if g.severity == "critical")
        high_gaps = sum(1 for g in analysis.gaps if g.severity == "high")
        
        status = "✅ COMPLIANT" if analysis.overall_compliance_score >= 0.9 else \
                 "⚠️ NEEDS ATTENTION" if analysis.overall_compliance_score >= 0.7 else \
                 "❌ NON-COMPLIANT"
        
        summary = f"""
COMPLIANCE STATUS: {status}

Policy: {analysis.policy_path}
Compliance Score: {analysis.overall_compliance_score:.1%}

Key Findings:
- Total Gaps: {len(analysis.gaps)}
- Critical: {critical_gaps}
- High Severity: {high_gaps}

Immediate Actions Required:
"""
        # Add top 3 priority gaps
        priority_gaps = sorted(
            analysis.gaps,
            key=lambda g: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(g.severity, 0),
            reverse=True
        )[:3]
        
        for i, gap in enumerate(priority_gaps, 1):
            summary += f"{i}. [{gap.severity.upper()}] {gap.issue} → {gap.recommendation}\n"
        
        return summary


__all__ = ["ReporterAgent"]
