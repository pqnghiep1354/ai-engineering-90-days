#!/usr/bin/env python3
"""
Example: Climate Change Research

Demonstrates using the multi-agent system for climate research.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import ResearchOrchestrator


async def main():
    """Run climate research example."""
    
    # Initialize orchestrator
    orchestrator = ResearchOrchestrator()
    
    # Example 1: Quick research on a specific topic
    print("=" * 60)
    print("Example 1: Quick Research")
    print("=" * 60)
    
    result1 = await orchestrator.quick_research(
        "What are the main impacts of climate change on coastal cities?"
    )
    
    if result1.success:
        print(f"\n✅ Research complete!")
        print(f"📊 Sources: {len(result1.sources)}")
        print(f"📝 Findings: {len(result1.findings)}")
        print(f"\n--- Report Preview ---\n")
        print(result1.report[:1000] + "...")
    else:
        print(f"❌ Failed: {result1.error}")
    
    # Example 2: Deep research with fact-checking
    print("\n" + "=" * 60)
    print("Example 2: Deep Research with Fact-Checking")
    print("=" * 60)
    
    result2 = await orchestrator.deep_research(
        "Compare renewable energy adoption rates between EU and US",
        enable_fact_checking=True,
    )
    
    if result2.success:
        print(f"\n✅ Research complete!")
        print(f"⏱️  Time: {result2.execution_time_seconds:.1f}s")
        
        # Save report
        filepath = orchestrator.save_report(result2)
        print(f"📁 Report saved to: {filepath}")
    else:
        print(f"❌ Failed: {result2.error}")
    
    # Show history
    print("\n" + "=" * 60)
    print("Research History")
    print("=" * 60)
    
    for h in orchestrator.get_history():
        status = "✓" if h["success"] else "✗"
        print(f"{status} {h['topic'][:50]} - {h['sources']} sources")


if __name__ == "__main__":
    asyncio.run(main())
