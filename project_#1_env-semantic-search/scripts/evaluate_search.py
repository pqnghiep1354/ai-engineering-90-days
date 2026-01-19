#!/usr/bin/env python3
"""
Search quality evaluation script for Environmental Semantic Search Tool.

Evaluates search quality using predefined test queries and expected results.

Usage:
    python scripts/evaluate_search.py
    python scripts/evaluate_search.py --queries data/sample_queries/test_queries.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.search_engine import SemanticSearchEngine, get_search_engine
from src.utils import setup_logging, Timer


# =============================================================================
# Test Queries
# =============================================================================

DEFAULT_TEST_QUERIES = [
    {
        "query": "What causes climate change?",
        "expected_keywords": ["greenhouse", "gas", "emission", "carbon", "CO2"],
        "category": "climate_basics",
    },
    {
        "query": "How does solar energy work?",
        "expected_keywords": ["solar", "photovoltaic", "sun", "electricity", "panel"],
        "category": "renewable_energy",
    },
    {
        "query": "What is ESG reporting?",
        "expected_keywords": ["environmental", "social", "governance", "sustainability"],
        "category": "esg",
    },
    {
        "query": "Air pollution health effects",
        "expected_keywords": ["respiratory", "health", "PM2.5", "particulate"],
        "category": "air_quality",
    },
    {
        "query": "Biến đổi khí hậu Việt Nam",
        "expected_keywords": ["Việt Nam", "biến đổi", "khí hậu", "nước biển"],
        "category": "vietnam_climate",
    },
    {
        "query": "renewable energy benefits",
        "expected_keywords": ["renewable", "clean", "sustainable", "emission"],
        "category": "renewable_energy",
    },
    {
        "query": "carbon footprint calculation",
        "expected_keywords": ["carbon", "scope", "emission", "footprint"],
        "category": "esg",
    },
    {
        "query": "greenhouse gas types",
        "expected_keywords": ["CO2", "methane", "nitrous", "greenhouse"],
        "category": "climate_basics",
    },
]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_keyword_recall(
    results: List[Dict[str, Any]],
    expected_keywords: List[str],
) -> float:
    """
    Calculate keyword recall - how many expected keywords appear in results.
    
    Args:
        results: Search results
        expected_keywords: Keywords that should appear
        
    Returns:
        Recall score (0-1)
    """
    if not expected_keywords or not results:
        return 0.0
    
    # Combine all result content
    all_content = " ".join(r.get("content", "").lower() for r in results)
    
    # Count found keywords
    found = sum(1 for kw in expected_keywords if kw.lower() in all_content)
    
    return found / len(expected_keywords)


def calculate_mrr(
    results: List[Dict[str, Any]],
    expected_keywords: List[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        results: Search results
        expected_keywords: Keywords that should appear
        
    Returns:
        MRR score (0-1)
    """
    if not results or not expected_keywords:
        return 0.0
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "").lower()
        # Check if any expected keyword is in this result
        if any(kw.lower() in content for kw in expected_keywords):
            return 1.0 / i
    
    return 0.0


def calculate_average_score(results: List[Dict[str, Any]]) -> float:
    """Calculate average similarity score of results."""
    if not results:
        return 0.0
    scores = [r.get("score", 0) for r in results]
    return sum(scores) / len(scores)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_query(
    engine: SemanticSearchEngine,
    query: str,
    expected_keywords: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate a single query.
    
    Args:
        engine: Search engine instance
        query: Query to evaluate
        expected_keywords: Expected keywords in results
        top_k: Number of results
        
    Returns:
        Evaluation metrics dictionary
    """
    with Timer(f"Query: {query[:30]}..."):
        response = engine.search(query, top_k=top_k)
    
    results = [r.to_dict() for r in response.results]
    
    return {
        "query": query,
        "num_results": len(results),
        "search_time_ms": response.search_time_ms,
        "keyword_recall": calculate_keyword_recall(results, expected_keywords),
        "mrr": calculate_mrr(results, expected_keywords),
        "avg_score": calculate_average_score(results),
        "top_score": results[0]["score"] if results else 0.0,
    }


def run_evaluation(
    engine: SemanticSearchEngine,
    test_queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run full evaluation on all test queries.
    
    Args:
        engine: Search engine instance
        test_queries: List of test query dictionaries
        top_k: Number of results per query
        
    Returns:
        Evaluation report dictionary
    """
    results = []
    category_metrics = {}
    
    for test in test_queries:
        query = test["query"]
        expected = test.get("expected_keywords", [])
        category = test.get("category", "general")
        
        logger.info(f"Evaluating: {query}")
        
        metrics = evaluate_query(engine, query, expected, top_k)
        metrics["category"] = category
        results.append(metrics)
        
        # Aggregate by category
        if category not in category_metrics:
            category_metrics[category] = []
        category_metrics[category].append(metrics)
    
    # Calculate overall metrics
    overall = {
        "total_queries": len(results),
        "avg_keyword_recall": sum(r["keyword_recall"] for r in results) / len(results),
        "avg_mrr": sum(r["mrr"] for r in results) / len(results),
        "avg_search_time_ms": sum(r["search_time_ms"] for r in results) / len(results),
        "avg_top_score": sum(r["top_score"] for r in results) / len(results),
    }
    
    # Calculate per-category metrics
    per_category = {}
    for category, metrics_list in category_metrics.items():
        per_category[category] = {
            "num_queries": len(metrics_list),
            "avg_keyword_recall": sum(m["keyword_recall"] for m in metrics_list) / len(metrics_list),
            "avg_mrr": sum(m["mrr"] for m in metrics_list) / len(metrics_list),
        }
    
    return {
        "overall": overall,
        "per_category": per_category,
        "detailed_results": results,
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print evaluation report."""
    overall = report["overall"]
    
    print("\n" + "=" * 60)
    print("🔍 SEARCH EVALUATION REPORT")
    print("=" * 60)
    
    print("\n📊 OVERALL METRICS")
    print("-" * 40)
    print(f"Total Queries:        {overall['total_queries']}")
    print(f"Avg Keyword Recall:   {overall['avg_keyword_recall']:.2%}")
    print(f"Avg MRR:              {overall['avg_mrr']:.3f}")
    print(f"Avg Search Time:      {overall['avg_search_time_ms']:.1f}ms")
    print(f"Avg Top Score:        {overall['avg_top_score']:.3f}")
    
    print("\n📁 PER-CATEGORY METRICS")
    print("-" * 40)
    for category, metrics in report["per_category"].items():
        print(f"\n{category}:")
        print(f"  Queries:          {metrics['num_queries']}")
        print(f"  Keyword Recall:   {metrics['avg_keyword_recall']:.2%}")
        print(f"  MRR:              {metrics['avg_mrr']:.3f}")
    
    print("\n📋 DETAILED RESULTS")
    print("-" * 40)
    for result in report["detailed_results"]:
        status = "✅" if result["keyword_recall"] >= 0.5 else "⚠️"
        print(f"\n{status} {result['query'][:50]}...")
        print(f"   Results: {result['num_results']}, "
              f"Recall: {result['keyword_recall']:.2%}, "
              f"MRR: {result['mrr']:.3f}, "
              f"Top Score: {result['top_score']:.3f}")
    
    print("\n" + "=" * 60)
    
    # Quality assessment
    if overall["avg_keyword_recall"] >= 0.7 and overall["avg_mrr"] >= 0.5:
        print("✅ Search quality: GOOD")
    elif overall["avg_keyword_recall"] >= 0.5 and overall["avg_mrr"] >= 0.3:
        print("⚠️ Search quality: ACCEPTABLE")
    else:
        print("❌ Search quality: NEEDS IMPROVEMENT")
    
    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate search quality"
    )
    
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to JSON file with test queries",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
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
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Load test queries
    if args.queries:
        queries_path = Path(args.queries)
        if not queries_path.exists():
            logger.error(f"Queries file not found: {queries_path}")
            sys.exit(1)
        
        with open(queries_path) as f:
            test_queries = json.load(f)
    else:
        test_queries = DEFAULT_TEST_QUERIES
    
    logger.info(f"Loaded {len(test_queries)} test queries")
    
    # Initialize search engine
    try:
        logger.info("Loading search engine...")
        engine = get_search_engine()
        
        stats = engine.get_stats()
        if stats.get("document_count", 0) == 0:
            logger.error("No documents indexed. Run index_documents.py first.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        sys.exit(1)
    
    # Run evaluation
    logger.info("Running evaluation...")
    report = run_evaluation(engine, test_queries, top_k=args.top_k)
    
    # Print report
    print_report(report)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
