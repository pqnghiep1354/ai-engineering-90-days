"""
Web Search Tool using Tavily API.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger

from ..config import settings
from ..agents.base import Source


class WebSearchTool:
    """
    Web search tool using Tavily API for environmental research.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: Optional[str] = None,
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: Tavily API key
            search_depth: Search depth (basic or advanced)
        """
        self.api_key = api_key or settings.tavily_api_key
        self.search_depth = search_depth or settings.tavily_search_depth
        self._client = None
        
        if not self.api_key:
            logger.warning("Tavily API key not configured. Using mock search.")
    
    def _get_client(self):
        """Get or create Tavily client."""
        if self._client is None and self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("Tavily package not installed")
        return self._client
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Source]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            include_domains: Domains to include
            exclude_domains: Domains to exclude
            
        Returns:
            List of Source objects
        """
        logger.info(f"Searching: {query}")
        
        # Try Tavily search
        client = self._get_client()
        
        if client:
            try:
                return await self._tavily_search(
                    client, query, max_results, include_domains, exclude_domains
                )
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")
        
        # Fallback to mock search for demo
        return self._mock_search(query, max_results)
    
    async def _tavily_search(
        self,
        client,
        query: str,
        max_results: int,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
    ) -> List[Source]:
        """Perform search using Tavily API."""
        # Run in executor since Tavily client is synchronous
        loop = asyncio.get_event_loop()
        
        search_params = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max_results,
        }
        
        if include_domains:
            search_params["include_domains"] = include_domains
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains
        
        response = await loop.run_in_executor(
            None,
            lambda: client.search(**search_params)
        )
        
        sources = []
        for result in response.get("results", []):
            domain = urlparse(result.get("url", "")).netloc
            
            sources.append(Source(
                title=result.get("title", "Unknown"),
                url=result.get("url", ""),
                content=result.get("content", ""),
                snippet=result.get("content", "")[:300],
                domain=domain,
                relevance_score=result.get("score", 0.5),
            ))
        
        return sources
    
    def _mock_search(self, query: str, max_results: int) -> List[Source]:
        """
        Mock search for demo/testing purposes.
        Returns pre-defined results for environmental topics.
        """
        # Mock data for common environmental queries
        mock_data = {
            "climate change": [
                Source(
                    title="IPCC Sixth Assessment Report",
                    url="https://www.ipcc.ch/assessment-report/ar6/",
                    content="The IPCC AR6 report confirms that human activities have unequivocally warmed the climate. Global surface temperature has increased faster since 1970 than in any other 50-year period over at least the last 2000 years.",
                    snippet="IPCC AR6 confirms human-caused climate change...",
                    domain="ipcc.ch",
                    relevance_score=0.95,
                ),
                Source(
                    title="NASA Climate Change Evidence",
                    url="https://climate.nasa.gov/evidence/",
                    content="The current warming trend is particularly significant because it is unequivocally the result of human activity since the mid-20th century and proceeding at a rate unprecedented over millennia.",
                    snippet="NASA evidence on climate change...",
                    domain="nasa.gov",
                    relevance_score=0.92,
                ),
            ],
            "renewable energy": [
                Source(
                    title="IEA Renewables 2023 Report",
                    url="https://www.iea.org/reports/renewables-2023",
                    content="Global renewable capacity additions are set to soar by 107 GW, the largest absolute increase ever, to more than 440 GW in 2023. Solar PV alone accounts for three-quarters of additions worldwide.",
                    snippet="IEA reports record renewable energy growth...",
                    domain="iea.org",
                    relevance_score=0.94,
                ),
                Source(
                    title="IRENA Renewable Energy Statistics",
                    url="https://www.irena.org/Statistics",
                    content="Renewable energy capacity reached 3,372 GW at the end of 2022, with solar and wind accounting for 90% of all new capacity added during the year.",
                    snippet="IRENA statistics on renewable capacity...",
                    domain="irena.org",
                    relevance_score=0.91,
                ),
            ],
            "esg": [
                Source(
                    title="Global ESG Disclosure Standards",
                    url="https://www.ifrs.org/groups/international-sustainability-standards-board/",
                    content="The ISSB has issued its first two standards: IFRS S1 on general sustainability disclosures and IFRS S2 on climate-related disclosures, creating a global baseline for sustainability reporting.",
                    snippet="ISSB releases global ESG standards...",
                    domain="ifrs.org",
                    relevance_score=0.93,
                ),
            ],
            "vietnam climate": [
                Source(
                    title="World Bank Vietnam Climate Assessment",
                    url="https://www.worldbank.org/en/country/vietnam/climate",
                    content="Vietnam is one of the top 5 countries most affected by climate change. The Mekong Delta, home to 18 million people, faces significant risks from sea level rise and saltwater intrusion.",
                    snippet="Vietnam climate vulnerability assessment...",
                    domain="worldbank.org",
                    relevance_score=0.94,
                ),
            ],
        }
        
        # Find matching results
        query_lower = query.lower()
        results = []
        
        for key, sources in mock_data.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                results.extend(sources)
        
        # Default results if no match
        if not results:
            results = [
                Source(
                    title=f"Search Results for: {query}",
                    url="https://example.com/search",
                    content=f"Mock search results for query: {query}. In production, this would return real web search results using the Tavily API.",
                    snippet="Mock search results...",
                    domain="example.com",
                    relevance_score=0.5,
                ),
            ]
        
        return results[:max_results]
    
    async def search_news(
        self,
        query: str,
        days: int = 7,
        max_results: int = 5,
    ) -> List[Source]:
        """
        Search for recent news articles.
        
        Args:
            query: Search query
            days: Number of days to look back
            max_results: Maximum results
            
        Returns:
            List of news sources
        """
        # Modify query for news
        news_query = f"{query} news recent {datetime.now().year}"
        return await self.search(news_query, max_results)
    
    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Source]:
        """
        Search for academic/scientific sources.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of academic sources
        """
        # Include academic domains
        academic_domains = [
            "sciencedirect.com",
            "nature.com",
            "springer.com",
            "nih.gov",
            "ipcc.ch",
            "researchgate.net",
        ]
        
        return await self.search(
            query,
            max_results,
            include_domains=academic_domains,
        )
