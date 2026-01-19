"""
Analysis Agent for data analysis and insights.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from loguru import logger

from .base import (
    BaseAgent,
    AgentRole,
    AgentState,
    AgentResponse,
    Finding,
)
from ..config import settings, ANALYST_SYSTEM_PROMPT


class AnalystAgent(BaseAgent):
    """
    Agent responsible for analyzing research data and generating insights.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
    ):
        """Initialize analyst agent."""
        super().__init__(
            role=AgentRole.ANALYST,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
        )
    
    async def process(
        self,
        state: AgentState,
        analysis_type: str = "comprehensive",
        **kwargs,
    ) -> AgentResponse:
        """
        Process analysis task.
        
        Args:
            state: Current agent state
            analysis_type: Type of analysis (comprehensive, trend, comparison)
            
        Returns:
            Agent response with analysis
        """
        logger.info(f"Analyst agent processing: {analysis_type} analysis")
        
        try:
            # Get context from state
            context = self._build_analysis_context(state)
            
            # Perform analysis based on type
            if analysis_type == "trend":
                analysis = await self._trend_analysis(state["topic"], context)
            elif analysis_type == "comparison":
                analysis = await self._comparative_analysis(state["topic"], context)
            else:
                analysis = await self._comprehensive_analysis(state["topic"], context)
            
            # Extract key insights
            insights = await self._extract_insights(state["topic"], analysis)
            
            return AgentResponse(
                agent=self.role,
                content=analysis,
                findings=insights,
                metadata={
                    "analysis_type": analysis_type,
                    "findings_analyzed": len(state.get("findings", [])),
                    "sources_analyzed": len(state.get("sources", [])),
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AgentResponse(
                agent=self.role,
                content="",
                success=False,
                error=str(e),
            )
    
    def _build_analysis_context(self, state: AgentState) -> str:
        """Build context for analysis from state."""
        context_parts = []
        
        # Add findings
        if state.get("findings"):
            findings_text = "\n".join(
                f"- {f.get('content', '')}" 
                for f in state["findings"]
            )
            context_parts.append(f"Research Findings:\n{findings_text}")
        
        # Add sources summary
        if state.get("sources"):
            sources_text = "\n".join(
                f"- {s.get('title', 'Unknown')}: {s.get('snippet', '')[:100]}..."
                for s in state["sources"][:10]
            )
            context_parts.append(f"\nKey Sources:\n{sources_text}")
        
        return "\n\n".join(context_parts)
    
    async def _comprehensive_analysis(
        self,
        topic: str,
        context: str,
    ) -> str:
        """Perform comprehensive analysis."""
        prompt = f"""Perform a comprehensive analysis of the research on "{topic}".

{context}

Your analysis should include:

1. **Key Themes**: What are the main themes and topics that emerge?

2. **Evidence Assessment**: How strong is the evidence? What are the key data points?

3. **Stakeholder Perspectives**: What different viewpoints exist on this topic?

4. **Gaps and Limitations**: What information is missing or unclear?

5. **Implications**: What are the broader implications of these findings?

6. **Recommendations**: What actions or further research would you suggest?

Provide a thorough, balanced analysis with specific references to the findings."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
    
    async def _trend_analysis(
        self,
        topic: str,
        context: str,
    ) -> str:
        """Perform trend analysis."""
        prompt = f"""Analyze the trends related to "{topic}" based on the research findings.

{context}

Your trend analysis should cover:

1. **Historical Context**: How has this issue evolved over time?

2. **Current State**: What is the current situation?

3. **Emerging Trends**: What new developments are appearing?

4. **Future Projections**: What trends are likely to continue or emerge?

5. **Driving Factors**: What factors are driving these trends?

6. **Regional Variations**: Are there geographic differences in trends?

Focus on identifying patterns and trajectories with supporting evidence."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
    
    async def _comparative_analysis(
        self,
        topic: str,
        context: str,
    ) -> str:
        """Perform comparative analysis."""
        prompt = f"""Conduct a comparative analysis related to "{topic}".

{context}

Your comparative analysis should include:

1. **Key Dimensions**: What aspects should be compared?

2. **Similarities**: What common themes or approaches exist?

3. **Differences**: What key differences emerge?

4. **Best Practices**: What approaches seem most effective?

5. **Trade-offs**: What trade-offs exist between different approaches?

6. **Synthesis**: What conclusions can be drawn from the comparison?

Create clear comparisons with specific evidence from the findings."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
    
    async def _extract_insights(
        self,
        topic: str,
        analysis: str,
    ) -> List[Finding]:
        """Extract key insights from analysis."""
        prompt = f"""From the following analysis about "{topic}", extract 5-7 key insights.

Analysis:
{analysis}

For each insight, provide:
INSIGHT: [A clear, actionable insight]
IMPORTANCE: [HIGH/MEDIUM/LOW]
EVIDENCE: [Brief supporting evidence]

Focus on the most significant and well-supported insights."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        # Parse insights
        insights = []
        current_insight = None
        importance = "MEDIUM"
        evidence = ""
        
        for line in response.content.split("\n"):
            line = line.strip()
            
            if line.startswith("INSIGHT:"):
                if current_insight:
                    confidence = 0.9 if importance == "HIGH" else (0.7 if importance == "MEDIUM" else 0.5)
                    insights.append(Finding(
                        content=current_insight,
                        confidence=confidence,
                        category="insight",
                    ))
                current_insight = line[8:].strip()
                importance = "MEDIUM"
                
            elif line.startswith("IMPORTANCE:"):
                importance = line[11:].strip().upper()
                
            elif line.startswith("EVIDENCE:"):
                evidence = line[9:].strip()
        
        # Don't forget the last insight
        if current_insight:
            confidence = 0.9 if importance == "HIGH" else (0.7 if importance == "MEDIUM" else 0.5)
            insights.append(Finding(
                content=current_insight,
                confidence=confidence,
                category="insight",
            ))
        
        return insights
