"""
Citation Management Tool.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger


@dataclass
class Citation:
    """Represents a citation."""
    id: int
    title: str
    url: str
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    publisher: str = ""
    accessed_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    citation_type: str = "web"  # web, journal, report, book
    
    def to_apa(self) -> str:
        """Format citation in APA style."""
        authors_str = ", ".join(self.authors) if self.authors else self.publisher or "Unknown"
        year = self.publication_date[:4] if self.publication_date else "n.d."
        
        return f"{authors_str} ({year}). {self.title}. Retrieved from {self.url}"
    
    def to_mla(self) -> str:
        """Format citation in MLA style."""
        authors_str = ", ".join(self.authors) if self.authors else self.publisher or "Unknown"
        
        return f'{authors_str}. "{self.title}." Web. {self.accessed_date}. <{self.url}>.'
    
    def to_chicago(self) -> str:
        """Format citation in Chicago style."""
        authors_str = ", ".join(self.authors) if self.authors else self.publisher or "Unknown"
        year = self.publication_date[:4] if self.publication_date else "n.d."
        
        return f'{authors_str}. "{self.title}." Accessed {self.accessed_date}. {self.url}.'
    
    def to_markdown(self) -> str:
        """Format citation for markdown."""
        return f"[{self.id}] [{self.title}]({self.url})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "publisher": self.publisher,
            "accessed_date": self.accessed_date,
            "citation_type": self.citation_type,
        }


class CitationManager:
    """
    Manages citations for research reports.
    """
    
    def __init__(self):
        """Initialize citation manager."""
        self.citations: Dict[str, Citation] = {}
        self._counter = 0
    
    def add_citation(
        self,
        title: str,
        url: str,
        authors: Optional[List[str]] = None,
        publication_date: Optional[str] = None,
        publisher: Optional[str] = None,
        citation_type: str = "web",
    ) -> Citation:
        """
        Add a new citation.
        
        Args:
            title: Source title
            url: Source URL
            authors: List of authors
            publication_date: Publication date
            publisher: Publisher name
            citation_type: Type of citation
            
        Returns:
            Created citation
        """
        # Check if URL already exists
        if url in self.citations:
            return self.citations[url]
        
        self._counter += 1
        
        # Extract publisher from domain if not provided
        if not publisher:
            domain = urlparse(url).netloc
            publisher = domain.replace("www.", "")
        
        citation = Citation(
            id=self._counter,
            title=title,
            url=url,
            authors=authors or [],
            publication_date=publication_date,
            publisher=publisher,
            citation_type=citation_type,
        )
        
        self.citations[url] = citation
        logger.debug(f"Added citation {self._counter}: {title}")
        
        return citation
    
    def add_from_source(self, source: Dict[str, Any]) -> Citation:
        """
        Add citation from a source dictionary.
        
        Args:
            source: Source dictionary
            
        Returns:
            Created citation
        """
        return self.add_citation(
            title=source.get("title", "Unknown Title"),
            url=source.get("url", ""),
            publisher=source.get("domain", ""),
        )
    
    def get_citation(self, url: str) -> Optional[Citation]:
        """Get citation by URL."""
        return self.citations.get(url)
    
    def get_citation_by_id(self, citation_id: int) -> Optional[Citation]:
        """Get citation by ID."""
        for citation in self.citations.values():
            if citation.id == citation_id:
                return citation
        return None
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations sorted by ID."""
        return sorted(self.citations.values(), key=lambda c: c.id)
    
    def format_reference_list(
        self,
        style: str = "apa",
    ) -> str:
        """
        Format all citations as a reference list.
        
        Args:
            style: Citation style (apa, mla, chicago, markdown)
            
        Returns:
            Formatted reference list
        """
        citations = self.get_all_citations()
        
        if not citations:
            return "No citations available."
        
        lines = []
        for citation in citations:
            if style == "apa":
                lines.append(f"[{citation.id}] {citation.to_apa()}")
            elif style == "mla":
                lines.append(f"[{citation.id}] {citation.to_mla()}")
            elif style == "chicago":
                lines.append(f"[{citation.id}] {citation.to_chicago()}")
            else:  # markdown
                lines.append(citation.to_markdown())
        
        return "\n\n".join(lines)
    
    def create_inline_citation(self, url: str) -> str:
        """
        Create inline citation reference.
        
        Args:
            url: Source URL
            
        Returns:
            Inline citation string (e.g., "[1]")
        """
        citation = self.get_citation(url)
        if citation:
            return f"[{citation.id}]"
        return ""
    
    def insert_citations_in_text(
        self,
        text: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Insert citation references into text.
        
        Args:
            text: Text to add citations to
            sources: Source dictionaries
            
        Returns:
            Text with citation references
        """
        # Add all sources as citations
        for source in sources:
            self.add_from_source(source)
        
        # This is a simplified version - in practice you'd use
        # more sophisticated matching
        return text
    
    def export_to_bibtex(self) -> str:
        """Export citations in BibTeX format."""
        lines = []
        
        for citation in self.get_all_citations():
            entry_type = "misc" if citation.citation_type == "web" else citation.citation_type
            key = f"ref{citation.id}"
            
            entry = f"""@{entry_type}{{{key},
    title = {{{citation.title}}},
    url = {{{citation.url}}},
    year = {{{citation.publication_date[:4] if citation.publication_date else datetime.now().year}}},
    note = {{Accessed: {citation.accessed_date}}}
}}"""
            lines.append(entry)
        
        return "\n\n".join(lines)
    
    def clear(self):
        """Clear all citations."""
        self.citations.clear()
        self._counter = 0
