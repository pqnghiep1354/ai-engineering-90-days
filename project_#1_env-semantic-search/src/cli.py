#!/usr/bin/env python3
"""
Command Line Interface for Environmental Semantic Search Tool.

Usage:
    python src/cli.py "climate change impacts"
    python src/cli.py --interactive
    python src/cli.py --help
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.search_engine import SemanticSearchEngine, get_search_engine, format_results_for_display
from src.utils import setup_logging


# =============================================================================
# CLI Colors
# =============================================================================

class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.END):
    """Print colored text."""
    print(f"{color}{text}{Colors.END}")


def print_header():
    """Print CLI header."""
    header = """
╔═══════════════════════════════════════════════════════════════╗
║          🔍 Environmental Semantic Search Tool                 ║
║              AI-Powered Document Search                        ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(header, Colors.GREEN)


def print_help():
    """Print interactive mode help."""
    help_text = """
Commands:
  /help      - Show this help
  /stats     - Show index statistics
  /sources   - List indexed sources
  /top <n>   - Set number of results (e.g., /top 10)
  /quit      - Exit program
  
Just type your search query and press Enter.
    """
    print_colored(help_text, Colors.YELLOW)


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive(engine: SemanticSearchEngine, top_k: int = 5):
    """Run interactive search mode."""
    print_header()
    print_help()
    
    current_top_k = top_k
    
    while True:
        try:
            # Get input
            print_colored("\n🔍 Search: ", Colors.GREEN)
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower().split()
                cmd = command[0]
                
                if cmd == "/help":
                    print_help()
                elif cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    print_colored("\n👋 Goodbye!", Colors.CYAN)
                    break
                elif cmd == "/stats":
                    stats = engine.get_stats()
                    print_colored("\n📊 Index Statistics:", Colors.BLUE)
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                elif cmd == "/sources":
                    sources = engine.get_sources()
                    print_colored(f"\n📁 Indexed Sources ({len(sources)}):", Colors.BLUE)
                    for source in sources:
                        print(f"  • {source}")
                elif cmd == "/top" and len(command) > 1:
                    try:
                        current_top_k = int(command[1])
                        print_colored(f"✓ Results set to {current_top_k}", Colors.YELLOW)
                    except ValueError:
                        print_colored("Invalid number", Colors.RED)
                else:
                    print_colored(f"Unknown command: {cmd}", Colors.RED)
                continue
            
            # Perform search
            print_colored("\n⏳ Searching...", Colors.CYAN)
            response = engine.search(user_input, top_k=current_top_k)
            
            # Display results
            print_colored(
                f"\n📋 Found {response.total_results} results "
                f"in {response.search_time_ms:.0f}ms",
                Colors.BLUE
            )
            
            if response.results:
                for i, result in enumerate(response.results, 1):
                    print_colored(f"\n[{i}] Score: {result.score:.3f}", Colors.GREEN)
                    print_colored(f"    Source: {result.source}", Colors.CYAN)
                    print(f"    {result.content[:300]}...")
            else:
                print_colored("No results found.", Colors.YELLOW)
                
        except KeyboardInterrupt:
            print_colored("\n\n👋 Goodbye!", Colors.CYAN)
            break
        except EOFError:
            break


# =============================================================================
# Single Query Mode
# =============================================================================

def run_single_query(
    engine: SemanticSearchEngine,
    query: str,
    top_k: int = 5,
    output_format: str = "text",
    threshold: float = 0.3,
):
    """Run a single search query."""
    
    response = engine.search(query, top_k=top_k, threshold=threshold)
    
    if output_format == "json":
        # JSON output
        output = response.to_dict()
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Text output
        print_colored(f"\n🔍 Query: {query}", Colors.GREEN)
        print_colored(
            f"📋 Found {response.total_results} results "
            f"in {response.search_time_ms:.0f}ms\n",
            Colors.BLUE
        )
        
        if response.results:
            for i, result in enumerate(response.results, 1):
                print_colored(f"[{i}] Score: {result.score:.3f} | Source: {result.source}", Colors.CYAN)
                print(f"    {result.content[:300]}...")
                print()
        else:
            print_colored("No results found.", Colors.YELLOW)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Environmental Semantic Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "climate change effects"
  %(prog)s "renewable energy" --top-k 10
  %(prog)s "ESG reporting" --format json
  %(prog)s --interactive
        """,
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Minimum similarity threshold (default: 0.3)",
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logging(level=log_level)
    
    # Validate arguments
    if not args.interactive and not args.query:
        print_colored("Error: Please provide a query or use --interactive mode", Colors.RED)
        sys.exit(1)
    
    # Initialize search engine
    try:
        print_colored("🔄 Loading search engine...", Colors.YELLOW)
        engine = get_search_engine()
        
        stats = engine.get_stats()
        doc_count = stats.get("document_count", 0)
        
        if doc_count == 0:
            print_colored(
                "⚠️  No documents indexed. Run index_documents.py first.",
                Colors.YELLOW
            )
            if not args.interactive:
                sys.exit(1)
        else:
            print_colored(f"✓ Loaded {doc_count} documents", Colors.GREEN)
            
    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        sys.exit(1)
    
    # Run appropriate mode
    if args.interactive:
        run_interactive(engine, top_k=args.top_k)
    else:
        run_single_query(
            engine,
            query=args.query,
            top_k=args.top_k,
            output_format=args.format,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
