"""
Fact-Checker Agent for verifying research claims.
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
from ..config import settings, FACT_CHECKER_SYSTEM_PROMPT
from ..tools.web_search import WebSearchTool


class FactCheckerAgent(BaseAgent):
    """
    Agent responsible for verifying claims and fact-checking research.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,  # Low temperature for accuracy
    ):
        """Initialize fact-checker agent."""
        super().__init__(
            role=AgentRole.FACT_CHECKER,
            system_prompt=FACT_CHECKER_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
        )
        
        # Initialize search tool for verification
        self.search_tool = WebSearchTool()
    
    async def process(
        self,
        state: AgentState,
        verify_all: bool = False,
        max_claims: int = 10,
        **kwargs,
    ) -> AgentResponse:
        """
        Process fact-checking task.
        
        Args:
            state: Current agent state
            verify_all: Whether to verify all claims
            max_claims: Maximum claims to verify
            
        Returns:
            Agent response with verification results
        """
        logger.info("Fact-checker agent verifying claims")
        
        try:
            # Extract claims to verify
            claims = await self._extract_claims(state)
            
            if not claims:
                return AgentResponse(
                    agent=self.role,
                    content="No specific claims found to verify.",
                    metadata={"claims_checked": 0},
                    success=True,
                )
            
            # Limit claims to verify
            claims_to_check = claims[:max_claims] if not verify_all else claims
            
            # Verify each claim
            verification_results = []
            for claim in claims_to_check:
                result = await self._verify_claim(claim)
                verification_results.append(result)
            
            # Generate verification summary
            summary = self._generate_verification_summary(verification_results)
            
            # Create findings from results
            findings = [
                Finding(
                    content=f"{r['claim']}: {'✓ Verified' if r['verified'] else '⚠ Needs review'}",
                    confidence=r.get("confidence", 0.5),
                    category="fact_check",
                )
                for r in verification_results
            ]
            
            return AgentResponse(
                agent=self.role,
                content=summary,
                findings=findings,
                metadata={
                    "claims_checked": len(verification_results),
                    "verified_count": sum(1 for r in verification_results if r["verified"]),
                    "details": verification_results,
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return AgentResponse(
                agent=self.role,
                content="",
                success=False,
                error=str(e),
            )
    
    async def _extract_claims(self, state: AgentState) -> List[str]:
        """Extract verifiable claims from state."""
        # Collect text to analyze
        text_parts = []
        
        # From findings
        for finding in state.get("findings", []):
            text_parts.append(finding.get("content", ""))
        
        # From analysis
        if state.get("analysis"):
            text_parts.append(state["analysis"])
        
        # From draft
        if state.get("draft"):
            text_parts.append(state["draft"])
        
        if not text_parts:
            return []
        
        combined_text = "\n\n".join(text_parts)
        
        prompt = f"""Extract specific, verifiable factual claims from the following text.
Focus on:
- Statistical claims (numbers, percentages, dates)
- Causal claims (X causes Y)
- Comparative claims (A is greater/better than B)
- Attribution claims (according to X, Y said)

Text:
{combined_text[:4000]}

List each claim on a separate line, without numbering.
Only include claims that can be fact-checked.
Maximum 10 claims."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        claims = [
            c.strip() 
            for c in response.content.strip().split("\n") 
            if c.strip() and len(c.strip()) > 10
        ]
        
        return claims
    
    async def _verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verify a single claim.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Verification result dictionary
        """
        # Search for verification
        search_query = f"verify fact: {claim}"
        
        try:
            sources = await self.search_tool.search(
                query=search_query,
                max_results=3,
            )
        except Exception:
            sources = []
        
        # Build context for verification
        if sources:
            source_context = "\n".join(
                f"- {s.title}: {s.snippet or s.content[:200]}"
                for s in sources
            )
        else:
            source_context = "No additional sources found."
        
        prompt = f"""Verify the following claim based on your knowledge and the provided sources.

Claim: {claim}

Sources found:
{source_context}

Provide your assessment:
1. Is the claim VERIFIED, PARTIALLY VERIFIED, UNVERIFIED, or FALSE?
2. What is your confidence level (0-100%)?
3. Brief explanation (1-2 sentences)

Format your response exactly as:
STATUS: [your assessment]
CONFIDENCE: [percentage]
EXPLANATION: [your explanation]"""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        # Parse response
        result = {
            "claim": claim,
            "verified": False,
            "status": "UNVERIFIED",
            "confidence": 0.5,
            "explanation": "",
            "sources": [s.to_dict() for s in sources] if sources else [],
        }
        
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                status = line[7:].strip().upper()
                result["status"] = status
                result["verified"] = status in ["VERIFIED", "PARTIALLY VERIFIED"]
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = line[11:].strip().replace("%", "")
                    result["confidence"] = float(conf) / 100
                except:
                    pass
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line[12:].strip()
        
        return result
    
    def _generate_verification_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """Generate summary of verification results."""
        if not results:
            return "No claims were verified."
        
        verified = sum(1 for r in results if r["verified"])
        total = len(results)
        
        summary_parts = [
            f"## Fact-Check Summary\n",
            f"**Claims Checked:** {total}",
            f"**Verified:** {verified} ({verified/total*100:.0f}%)",
            f"**Needs Review:** {total - verified}",
            "",
            "### Details:",
        ]
        
        for r in results:
            icon = "✓" if r["verified"] else "⚠"
            status = r["status"]
            conf = r["confidence"] * 100
            
            summary_parts.append(
                f"\n{icon} **{status}** (Confidence: {conf:.0f}%)\n"
                f"   Claim: {r['claim']}\n"
                f"   {r['explanation']}"
            )
        
        return "\n".join(summary_parts)
    
    async def cross_reference(
        self,
        claim: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Cross-reference a claim against multiple sources.
        
        Args:
            claim: Claim to verify
            sources: Existing sources to check against
            
        Returns:
            Cross-reference result
        """
        if not sources:
            return {
                "claim": claim,
                "supported_by": [],
                "contradicted_by": [],
                "inconclusive": True,
            }
        
        source_context = "\n".join(
            f"Source {i+1}: {s.get('title', 'Unknown')}\n{s.get('snippet', '')[:200]}"
            for i, s in enumerate(sources[:5])
        )
        
        prompt = f"""Analyze whether the following claim is supported by the given sources.

Claim: {claim}

Sources:
{source_context}

For each source, indicate if it:
- SUPPORTS the claim
- CONTRADICTS the claim
- Is NEUTRAL/INCONCLUSIVE

Format as:
Source 1: [SUPPORTS/CONTRADICTS/NEUTRAL] - [brief reason]
Source 2: ...

Then provide overall assessment."""

        messages = self._create_messages(prompt)
        response = await self._call_llm(messages)
        
        return {
            "claim": claim,
            "analysis": response.content,
        }
