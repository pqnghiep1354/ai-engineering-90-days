"""
Web Search Tool for EIA Research.

Uses Tavily API for searching current environmental information.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import get_settings


class WebSearchTool:
    """
    Web search tool for researching environmental data.
    
    Uses Tavily API for:
    - Current environmental news
    - Latest regulations and updates
    - Location-specific environmental data
    - Best practices and case studies
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.tavily_api_key
        self._client = None
    
    @property
    def client(self):
        """Lazy load Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("Tavily not installed. Install with: pip install tavily-python")
                self._client = None
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")
                self._client = None
        return self._client
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",  # basic or advanced
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: Search depth (basic or advanced)
            include_domains: Only include results from these domains
            exclude_domains: Exclude results from these domains
            
        Returns:
            List of search results
        """
        if not self.client:
            logger.warning("Web search unavailable - returning empty results")
            return []
        
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def search_regulations(
        self,
        topic: str,
        country: str = "Vietnam",
    ) -> List[Dict[str, Any]]:
        """Search for environmental regulations."""
        query = f"{country} environmental regulations {topic}"
        
        include_domains = [
            "thuvienphapluat.vn",
            "monre.gov.vn",
            "moit.gov.vn",
        ]
        
        return self.search(
            query=query,
            max_results=5,
            include_domains=include_domains,
        )
    
    def search_environmental_data(
        self,
        location: str,
        data_type: str = "general",  # air, water, climate, etc.
    ) -> List[Dict[str, Any]]:
        """Search for environmental data by location."""
        query_map = {
            "air": f"air quality data {location} pollution",
            "water": f"water quality {location} river groundwater",
            "climate": f"climate weather {location} temperature rainfall",
            "general": f"environmental conditions {location}",
        }
        
        query = query_map.get(data_type, f"environmental {data_type} {location}")
        
        return self.search(query=query, max_results=5)
    
    def search_case_studies(
        self,
        project_type: str,
        country: str = "Vietnam",
    ) -> List[Dict[str, Any]]:
        """Search for EIA case studies."""
        query = f"environmental impact assessment {project_type} {country} case study"
        
        return self.search(
            query=query,
            max_results=5,
            search_depth="advanced",
        )
    
    def search_best_practices(
        self,
        topic: str,
    ) -> List[Dict[str, Any]]:
        """Search for environmental best practices."""
        query = f"environmental best practices {topic} mitigation measures"
        
        include_domains = [
            "ifc.org",
            "worldbank.org",
            "epa.gov",
            "unep.org",
        ]
        
        return self.search(
            query=query,
            max_results=5,
            include_domains=include_domains,
        )
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
    ) -> str:
        """Get context string from search results."""
        results = self.search(query, max_results=3)
        
        context_parts = []
        total_length = 0
        
        for result in results:
            content = result.get("content", "")
            if total_length + len(content) > max_tokens * 4:  # Rough estimate
                break
            
            context_parts.append(f"Source: {result.get('title', 'Unknown')}")
            context_parts.append(content)
            context_parts.append("---")
            total_length += len(content)
        
        return "\n".join(context_parts)
