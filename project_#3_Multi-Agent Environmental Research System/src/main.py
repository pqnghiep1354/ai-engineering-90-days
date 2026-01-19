#!/usr/bin/env python3
"""
Main entry point for Multi-Agent Environmental Research System.

Usage:
    python -m src.main "Research topic"
    python -m src.main --interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import settings
from .orchestrator import ResearchOrchestrator


console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )


def print_header():
    """Print application header."""
    header = """
╔═══════════════════════════════════════════════════════════════╗
║     🤖 Multi-Agent Environmental Research System               ║
║         AI-Powered Research Automation                         ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(header, style="bold green")


def print_result(result):
    """Print research result."""
    if result.success:
        console.print("\n✅ Research Complete!", style="bold green")
        console.print(f"📊 Sources: {len(result.sources)}")
        console.print(f"📝 Findings: {len(result.findings)}")
        console.print(f"⏱️  Time: {result.execution_time_seconds:.1f}s")
        
        console.print("\n" + "=" * 60 + "\n")
        
        # Print report as markdown
        md = Markdown(result.report)
        console.print(md)
    else:
        console.print(f"\n❌ Research Failed: {result.error}", style="bold red")


async def run_research(topic: str, workflow: str, output: str = None, verbose: bool = False):
    """Run research with progress display."""
    orchestrator = ResearchOrchestrator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Researching: {topic[:50]}...", total=None)
        
        result = await orchestrator.research(
            topic=topic,
            workflow=workflow,
        )
        
        progress.update(task, description="Complete!")
    
    # Save report if output specified
    if output and result.success:
        orchestrator.save_report(result, filename=output)
        console.print(f"📁 Report saved to: {output}", style="bold blue")
    
    return result


async def interactive_mode():
    """Run interactive research mode."""
    print_header()
    
    console.print("""
Available commands:
  /research <topic>  - Start new research
  /workflow <type>   - Set workflow (quick, deep)
  /history           - Show research history
  /help              - Show this help
  /quit              - Exit

Just type a topic to start quick research.
    """, style="yellow")
    
    orchestrator = ResearchOrchestrator()
    current_workflow = "quick"
    
    while True:
        try:
            user_input = console.input("\n[bold green]🔍 Research>[/] ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ["/quit", "/exit", "/q"]:
                    console.print("\n👋 Goodbye!", style="bold cyan")
                    break
                    
                elif command == "/help":
                    console.print("""
Commands:
  /research <topic> - Start research on a topic
  /workflow <type>  - Set workflow type (quick, deep)
  /history          - Show research history
  /quit             - Exit the program
                    """, style="yellow")
                    
                elif command == "/workflow":
                    if args in ["quick", "deep"]:
                        current_workflow = args
                        console.print(f"✓ Workflow set to: {current_workflow}", style="green")
                    else:
                        console.print("Available workflows: quick, deep", style="yellow")
                        
                elif command == "/history":
                    history = orchestrator.get_history()
                    if history:
                        for i, h in enumerate(history, 1):
                            status = "✓" if h["success"] else "✗"
                            console.print(f"{i}. {status} {h['topic'][:50]} ({h['sources']} sources)")
                    else:
                        console.print("No research history yet.", style="yellow")
                        
                elif command == "/research":
                    if args:
                        result = await orchestrator.research(args, workflow=current_workflow)
                        print_result(result)
                    else:
                        console.print("Please provide a research topic.", style="yellow")
                else:
                    console.print(f"Unknown command: {command}", style="red")
            
            else:
                # Treat as research topic
                result = await orchestrator.research(user_input, workflow=current_workflow)
                print_result(result)
                
        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="bold cyan")
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"Error: {e}", style="red")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Environmental Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Climate change impacts in Southeast Asia"
  %(prog)s "ESG trends 2024" --workflow deep
  %(prog)s --interactive
        """,
    )
    
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Research topic",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    
    parser.add_argument(
        "--workflow", "-w",
        choices=["quick", "deep"],
        default="quick",
        help="Workflow type (default: quick)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for report",
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
    setup_logging(args.verbose)
    
    # Check API keys
    if not settings.has_openai_key:
        console.print("⚠️  Warning: OPENAI_API_KEY not configured.", style="yellow")
        console.print("Please set it in .env file.", style="yellow")
    
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.topic:
        print_header()
        result = asyncio.run(run_research(
            topic=args.topic,
            workflow=args.workflow,
            output=args.output,
            verbose=args.verbose,
        ))
        print_result(result)
    else:
        console.print("Please provide a topic or use --interactive mode.", style="yellow")
        console.print("Use --help for more information.", style="yellow")


if __name__ == "__main__":
    main()
