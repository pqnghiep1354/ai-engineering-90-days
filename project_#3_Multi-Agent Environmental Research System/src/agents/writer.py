"""
Writer Agent for report generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from loguru import logger

from .base import (
    BaseAgent,
    AgentRole,
    AgentState,
    AgentResponse,
)
from ..config import settings, WRITER_SYSTEM_PROMPT


class WriterAgent(BaseAgent):
    """
    Agent responsible for writing and formatting research reports.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
    ):
        """Initialize writer agent."""
        super().__init__(
            role=AgentRole.WRITER,
            system_prompt=WRITER_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
        )
    
    async def process(
        self,
        state: AgentState,
        report_format: str = "full",
        include_citations: bool = True,
        **kwargs,
    ) -> AgentResponse:
        """
        Process writing task.
        
        Args:
            state: Current agent state
            report_format: Report format (full, executive, brief)
            include_citations: Whether to include citations
            
        Returns:
            Agent response with report
        """
        logger.info(f"Writer agent creating {report_format} report")
        
        try:
            # Build content from state
            content_context = self._build_content_context(state)
            
            # Generate report based on format
            if report_format == "executive":
                report = await self._write_executive_summary(
                    state["topic"], content_context
                )
            elif report_format == "brief":
                report = await self._write_brief(
                    state["topic"], content_context
                )
            else:
                report = await self._write_full_report(
                    state["topic"], content_context, state
                )
            
            # Add citations if requested
            if include_citations and state.get("sources"):
                citations = self._format_citations(state["sources"])
                report = f"{report}\n\n---\n\n## References\n\n{citations}"
            
            # Add metadata
            report = self._add_report_metadata(report, state)
            
            return AgentResponse(
                agent=self.role,
                content=report,
                metadata={
                    "format": report_format,
                    "word_count": len(report.split()),
                    "sources_cited": len(state.get("sources", [])),
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Report writing failed: {e}")
            return AgentResponse(
                agent=self.role,
                content="",
                success=False,
                error=str(e),
            )
    
    def _build_content_context(self, state: AgentState) -> str:
        """Build context for report writing."""
        parts = []
        
        # Add findings
        if state.get("findings"):
            findings_text = "\n".join(
                f"- {f.get('content', '')}"
                for f in state["findings"]
            )
            parts.append(f"Key Findings:\n{findings_text}")
        
        # Add analysis
        if state.get("analysis"):
            parts.append(f"Analysis:\n{state['analysis']}")
        
        # Add fact-check results if available
        if state.get("fact_check_results"):
            verified = sum(1 for r in state["fact_check_results"] if r.get("verified", False))
            total = len(state["fact_check_results"])
            parts.append(f"Fact-Check: {verified}/{total} claims verified")
        
        return "\n\n".join(parts)
    
    async def _write_full_report(
        self,
        topic: str,
        context: str,
        state: AgentState,
    ) -> str:
        """Write a full research report."""
        prompt = f"""Write a comprehensive research report on "{topic}".

{context}

Structure the report with the following sections:

1. **Executive Summary** (2-3 paragraphs)
   - Key findings and conclusions

2. **Introduction**
   - Background and context
   - Scope of research
   - Research questions

3. **Methodology**
   - How research was conducted
   - Sources and tools used

4. **Key Findings**
   - Detailed findings organized by theme
   - Supporting evidence and data

5. **Analysis and Discussion**
   - Interpretation of findings
   - Implications
   - Limitations

6. **Conclusions and Recommendations**
   - Summary of key points
   - Actionable recommendations
   - Areas for further research

Write in a professional, clear style appropriate for a research report.
Use markdown formatting with headers, bullet points, and emphasis where appropriate.
Include specific data points and cite sources where available."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
    
    async def _write_executive_summary(
        self,
        topic: str,
        context: str,
    ) -> str:
        """Write an executive summary."""
        prompt = f"""Write an executive summary on "{topic}".

{context}

The executive summary should:
- Be 300-500 words
- Start with the most important conclusion
- Highlight 3-5 key findings
- Include actionable recommendations
- Be suitable for busy executives

Write in clear, direct language. Avoid jargon.
Use bullet points for key takeaways."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return f"# Executive Summary: {topic}\n\n{response.content}"
    
    async def _write_brief(
        self,
        topic: str,
        context: str,
    ) -> str:
        """Write a brief summary."""
        prompt = f"""Write a brief summary on "{topic}".

{context}

The brief should be:
- 150-250 words
- Focus on key takeaways only
- Include main conclusion
- List 3-4 key points
- Be scannable and easy to read

Write concisely and directly."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return f"# Brief: {topic}\n\n{response.content}"
    
    def _format_citations(self, sources: List[Dict[str, Any]]) -> str:
        """Format citations from sources."""
        citations = []
        
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Unknown Title")
            url = source.get("url", "")
            domain = source.get("domain", "")
            
            if url:
                citations.append(f"{i}. [{title}]({url})")
            else:
                citations.append(f"{i}. {title}")
        
        return "\n".join(citations)
    
    def _add_report_metadata(self, report: str, state: AgentState) -> str:
        """Add metadata to report."""
        metadata = f"""---
**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Topic:** {state['topic']}
**Sources Analyzed:** {len(state.get('sources', []))}
**Generated by:** Multi-Agent Environmental Research System
---

"""
        return metadata + report
    
    async def improve_draft(
        self,
        draft: str,
        feedback: str,
    ) -> str:
        """Improve a draft based on feedback."""
        prompt = f"""Improve the following draft based on the feedback provided.

Draft:
{draft}

Feedback:
{feedback}

Make the necessary improvements while maintaining the overall structure.
Return the improved version."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
