#!/usr/bin/env python3
"""
Command Line Interface for Climate Q&A RAG System.

Usage:
    python src/cli.py "What causes climate change?"
    python src/cli.py --interactive
    python src/cli.py --help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.chain import RAGChain, ConversationalRAGChain, AdvancedRAGChain
from src.vector_store import load_existing_index
from src.embeddings import get_embedding_model
from src.utils import setup_logging, detect_language


# =============================================================================
# CLI Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.END):
    """Print colored text."""
    print(f"{color}{text}{Colors.END}")


def print_header():
    """Print CLI header."""
    header = """
╔═══════════════════════════════════════════════════════════════╗
║              🌍 Climate Science Q&A System                     ║
║         Ask questions about climate and environment           ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(header, Colors.CYAN)


def print_help():
    """Print interactive mode help."""
    help_text = """
Commands:
  /help     - Show this help message
  /clear    - Clear conversation history
  /sources  - Toggle source display
  /lang     - Toggle language (en/vi/auto)
  /export   - Export chat history
  /quit     - Exit the program
  
Just type your question and press Enter to get an answer.
    """
    print_colored(help_text, Colors.YELLOW)


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive(chain: ConversationalRAGChain, show_sources: bool = True):
    """Run interactive chat mode."""
    print_header()
    print_help()
    
    language = "auto"
    
    while True:
        try:
            # Get user input
            print_colored("\n📝 You: ", Colors.GREEN)
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/help":
                    print_help()
                    continue
                
                elif command == "/clear":
                    chain.clear_history()
                    print_colored("✓ Conversation history cleared", Colors.YELLOW)
                    continue
                
                elif command == "/sources":
                    show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    print_colored(f"✓ Source display {status}", Colors.YELLOW)
                    continue
                
                elif command == "/lang":
                    if language == "auto":
                        language = "en"
                    elif language == "en":
                        language = "vi"
                    else:
                        language = "auto"
                    print_colored(f"✓ Language set to: {language}", Colors.YELLOW)
                    continue
                
                elif command == "/export":
                    export_chat_history(chain)
                    continue
                
                elif command in ["/quit", "/exit", "/q"]:
                    print_colored("\n👋 Goodbye!", Colors.CYAN)
                    break
                
                else:
                    print_colored(f"Unknown command: {command}", Colors.RED)
                    continue
            
            # Query the system
            print_colored("\n🤖 Assistant: ", Colors.BLUE)
            
            try:
                result = chain.invoke(user_input, return_sources=show_sources)
                
                # Print answer
                print(result["answer"])
                
                # Print sources if enabled
                if show_sources and result.get("sources"):
                    print_colored("\n📚 Sources:", Colors.YELLOW)
                    for i, source in enumerate(result["sources"], 1):
                        metadata = source.get("metadata", {})
                        source_name = metadata.get("source", "Unknown")
                        print_colored(f"  [{i}] {source_name}", Colors.CYAN)
                
            except Exception as e:
                print_colored(f"Error: {e}", Colors.RED)
                logger.exception("Query failed")
        
        except KeyboardInterrupt:
            print_colored("\n\n👋 Goodbye!", Colors.CYAN)
            break
        
        except EOFError:
            break


def export_chat_history(chain: ConversationalRAGChain):
    """Export chat history to file."""
    history = chain.get_history()
    
    if not history:
        print_colored("No chat history to export", Colors.YELLOW)
        return
    
    # Create export file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Climate Q&A Chat History\n")
        f.write(f"Exported: {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            f.write(f"{role}:\n{content}\n\n")
    
    print_colored(f"✓ Chat history exported to: {filename}", Colors.GREEN)


# =============================================================================
# Single Query Mode
# =============================================================================

def run_single_query(
    chain: RAGChain,
    question: str,
    show_sources: bool = True,
    output_format: str = "text",
):
    """Run a single query."""
    try:
        result = chain.invoke(question, return_sources=show_sources)
        
        if output_format == "json":
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Text format
            print_colored("\n🤖 Answer:", Colors.BLUE)
            print(result["answer"])
            
            if show_sources and result.get("sources"):
                print_colored("\n📚 Sources:", Colors.YELLOW)
                for i, source in enumerate(result["sources"], 1):
                    metadata = source.get("metadata", {})
                    source_name = metadata.get("source", "Unknown")
                    page = metadata.get("page", "N/A")
                    print_colored(f"  [{i}] {source_name} (Page: {page})", Colors.CYAN)
    
    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        sys.exit(1)


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Climate Science Q&A System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What causes climate change?"
  %(prog)s --interactive
  %(prog)s "Biến đổi khí hậu là gì?" --lang vi
  %(prog)s "What is ESG?" --format json --no-sources
        """,
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask (optional if --interactive)",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode",
    )
    
    parser.add_argument(
        "--lang", "-l",
        choices=["en", "vi", "auto"],
        default="auto",
        help="Response language (default: auto)",
    )
    
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents",
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
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranking",
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logging(level=log_level)
    
    # Validate arguments
    if not args.interactive and not args.question:
        print_colored("Error: Please provide a question or use --interactive mode", Colors.RED)
        sys.exit(1)
    
    # Setup LangSmith
    settings.setup_langsmith()
    
    # Initialize chain
    print_colored("🔄 Loading knowledge base...", Colors.YELLOW)
    
    try:
        embeddings = get_embedding_model()
        manager = load_existing_index(embeddings=embeddings)
        
        if args.interactive:
            chain = ConversationalRAGChain(
                vector_store=manager.vector_store,
                language=args.lang if args.lang != "auto" else "en",
            )
            run_interactive(chain, show_sources=not args.no_sources)
        else:
            chain = RAGChain(
                vector_store=manager.vector_store,
                use_reranker=not args.no_reranker,
                language=args.lang if args.lang != "auto" else detect_language(args.question),
            )
            run_single_query(
                chain,
                args.question,
                show_sources=not args.no_sources,
                output_format=args.format,
            )
    
    except FileNotFoundError:
        print_colored(
            "Error: No documents indexed. Please run index_documents.py first.",
            Colors.RED
        )
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error initializing system: {e}", Colors.RED)
        if args.verbose:
            logger.exception("Initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
