"""
Research Agent for information gathering.
"""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from loguru import logger

from .base import (
    BaseAgent,
    AgentRole,
    AgentState,
    AgentResponse,
    Source,
    Finding,
)
from ..config import settings, RESEARCHER_SYSTEM_PROMPT
from ..tools.web_search import WebSearchTool


class ResearcherAgent(BaseAgent):
    """
    Agent responsible for gathering information from various sources.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """Initialize researcher agent."""
        super().__init__(
            role=AgentRole.RESEARCHER,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
        )
        
        # Initialize search tool
        self.search_tool = WebSearchTool()
    
    async def process(
        self,
        state: AgentState,
        search_queries: Optional[List[str]] = None,
        max_sources: int = 10,
        **kwargs,
    ) -> AgentResponse:
        """
        Process research task.
        
        Args:
            state: Current agent state
            search_queries: Specific queries to search
            max_sources: Maximum sources to collect
            
        Returns:
            Agent response with findings and sources
        """
        logger.info(f"Researcher agent processing topic: {state['topic']}")
        
        try:
            # Generate search queries if not provided
            if not search_queries:
                search_queries = await self._generate_queries(state["topic"])
            
            # Perform searches
            all_sources = []
            for query in search_queries[:5]:  # Limit queries
                logger.info(f"Searching: {query}")
                sources = await self.search_tool.search(
                    query=query,
                    max_results=max_sources // len(search_queries) + 1,
                )
                all_sources.extend(sources)
            
            # Deduplicate sources
            unique_sources = self._deduplicate_sources(all_sources)
            logger.info(f"Found {len(unique_sources)} unique sources")
            
            # Extract findings from sources
            findings = await self._extract_findings(
                state["topic"],
                unique_sources[:max_sources],
            )
            
            # Generate summary
            summary = await self._generate_summary(state["topic"], findings)
            
            return AgentResponse(
                agent=self.role,
                content=summary,
                findings=findings,
                sources=unique_sources[:max_sources],
                metadata={
                    "queries": search_queries,
                    "total_sources": len(all_sources),
                    "unique_sources": len(unique_sources),
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return AgentResponse(
                agent=self.role,
                content="",
                success=False,
                error=str(e),
            )
    
    async def _generate_queries(self, topic: str) -> List[str]:
        """
        Generate search queries for a topic.
        
        Args:
            topic: Research topic
            
        Returns:
            List of search queries
        """
        prompt = f"""Generate 5 specific search queries to research the following topic comprehensively.
Focus on environmental, climate, and sustainability aspects.

Topic: {topic}

Return only the queries, one per line, without numbering or bullet points.
Make queries specific and searchable."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        # Parse queries from response
        queries = [
            q.strip() 
            for q in response.content.strip().split("\n") 
            if q.strip()
        ]
        
        # Always include the original topic as a query
        if topic not in queries:
            queries.insert(0, topic)
        
        return queries[:5]
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on URL."""
        seen_urls = set()
        unique = []
        
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique.append(source)
        
        return unique
    
    async def _extract_findings(
        self,
        topic: str,
        sources: List[Source],
    ) -> List[Finding]:
        """
        Extract key findings from sources.
        
        Args:
            topic: Research topic
            sources: List of sources
            
        Returns:
            List of findings
        """
        if not sources:
            return []
        
        # Prepare source summaries
        source_texts = []
        for i, source in enumerate(sources[:10], 1):  # Limit for context
            source_texts.append(
                f"[{i}] {source.title}\n"
                f"URL: {source.url}\n"
                f"Content: {source.snippet or source.content[:500]}"
            )
        
        sources_context = "\n\n".join(source_texts)
        
        prompt = f"""Based on the following sources, extract 5-10 key findings about the topic.
Each finding should be a specific, factual statement with clear attribution.

Topic: {topic}

Sources:
{sources_context}

Format each finding as:
FINDING: [The specific finding]
SOURCE: [Source number(s) that support this]
CONFIDENCE: [HIGH/MEDIUM/LOW]

Extract the most important and well-supported findings."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        # Parse findings
        findings = self._parse_findings(response.content, sources)
        
        return findings
    
    def _parse_findings(
        self,
        response_text: str,
        sources: List[Source],
    ) -> List[Finding]:
        """Parse findings from LLM response."""
        findings = []
        current_finding = None
        current_sources = []
        confidence = 0.8
        
        for line in response_text.split("\n"):
            line = line.strip()
            
            if line.startswith("FINDING:"):
                # Save previous finding
                if current_finding:
                    findings.append(Finding(
                        content=current_finding,
                        sources=current_sources,
                        confidence=confidence,
                    ))
                
                current_finding = line[8:].strip()
                current_sources = []
                confidence = 0.8
                
            elif line.startswith("SOURCE:"):
                # Parse source references
                source_refs = line[7:].strip()
                try:
                    for ref in source_refs.replace(",", " ").split():
                        ref = ref.strip("[]")
                        if ref.isdigit():
                            idx = int(ref) - 1
                            if 0 <= idx < len(sources):
                                current_sources.append(sources[idx])
                except:
                    pass
                    
            elif line.startswith("CONFIDENCE:"):
                conf_text = line[11:].strip().upper()
                if "HIGH" in conf_text:
                    confidence = 0.9
                elif "LOW" in conf_text:
                    confidence = 0.6
                else:
                    confidence = 0.8
        
        # Don't forget the last finding
        if current_finding:
            findings.append(Finding(
                content=current_finding,
                sources=current_sources,
                confidence=confidence,
            ))
        
        return findings
    
    async def _generate_summary(
        self,
        topic: str,
        findings: List[Finding],
    ) -> str:
        """Generate a summary of research findings."""
        if not findings:
            return "No findings were extracted from the sources."
        
        findings_text = "\n".join(
            f"- {f.content} (confidence: {f.confidence:.0%})"
            for f in findings
        )
        
        prompt = f"""Summarize the following research findings about "{topic}" in 2-3 paragraphs.
Highlight the most important points and any areas of consensus or disagreement.

Findings:
{findings_text}

Write a clear, informative summary."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return response.content
