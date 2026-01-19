#!/usr/bin/env python3
"""
RAG Evaluation Script for Climate Q&A System.

Evaluates the quality of RAG responses using various metrics.

Usage:
    python scripts/evaluate.py --test-file data/test_questions.json
    python scripts/evaluate.py --quick-test
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.chain import AdvancedRAGChain
from src.vector_store import load_existing_index
from src.embeddings import get_embedding_model
from src.utils import setup_logging, detect_language


# =============================================================================
# Evaluation Metrics
# =============================================================================

class RAGEvaluator:
    """Evaluator for RAG system responses."""
    
    def __init__(self, chain: AdvancedRAGChain):
        """Initialize evaluator."""
        self.chain = chain
        self.results = []
    
    def evaluate_question(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        expected_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.
        
        Args:
            question: Question to evaluate
            expected_answer: Expected answer (for comparison)
            expected_sources: Expected source documents
            
        Returns:
            Evaluation results
        """
        start_time = time.time()
        
        try:
            result = self.chain.invoke(
                question,
                return_sources=True,
                return_context=True,
            )
            
            latency = time.time() - start_time
            
            evaluation = {
                "question": question,
                "answer": result["answer"],
                "num_sources": result.get("num_sources", 0),
                "latency_seconds": round(latency, 3),
                "language": result.get("language", "unknown"),
                "success": True,
            }
            
            # Calculate metrics if expected answer provided
            if expected_answer:
                evaluation["expected_answer"] = expected_answer
                evaluation["answer_similarity"] = self._calculate_similarity(
                    result["answer"], expected_answer
                )
            
            # Check source relevance
            if result.get("sources"):
                evaluation["source_relevance"] = self._evaluate_sources(
                    question, result["sources"]
                )
            
            # Answer quality metrics
            evaluation["answer_length"] = len(result["answer"])
            evaluation["has_citations"] = "[" in result["answer"] or "Source" in result["answer"]
            
        except Exception as e:
            evaluation = {
                "question": question,
                "error": str(e),
                "success": False,
                "latency_seconds": time.time() - start_time,
            }
        
        self.results.append(evaluation)
        return evaluation
    
    def _calculate_similarity(self, answer: str, expected: str) -> float:
        """Calculate simple word overlap similarity."""
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(answer_words & expected_words)
        return overlap / len(expected_words)
    
    def _evaluate_sources(
        self,
        question: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate relevance of retrieved sources."""
        question_words = set(question.lower().split())
        
        relevance_scores = []
        for source in sources:
            content = source.get("content", "").lower()
            content_words = set(content.split())
            
            overlap = len(question_words & content_words)
            score = overlap / max(len(question_words), 1)
            relevance_scores.append(score)
        
        return {
            "scores": relevance_scores,
            "avg_relevance": sum(relevance_scores) / max(len(relevance_scores), 1),
            "num_sources": len(sources),
        }
    
    def run_evaluation(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple questions.
        
        Args:
            questions: List of question dicts with 'question' and optional 'expected_answer'
            show_progress: Whether to show progress
            
        Returns:
            Aggregated evaluation results
        """
        self.results = []
        
        for i, q in enumerate(questions):
            if show_progress:
                logger.info(f"Evaluating [{i+1}/{len(questions)}]: {q['question'][:50]}...")
            
            self.evaluate_question(
                question=q["question"],
                expected_answer=q.get("expected_answer"),
                expected_sources=q.get("expected_sources"),
            )
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]
        
        latencies = [r["latency_seconds"] for r in successful]
        
        summary = {
            "total_questions": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results),
            "avg_latency_seconds": sum(latencies) / max(len(latencies), 1),
            "min_latency_seconds": min(latencies) if latencies else 0,
            "max_latency_seconds": max(latencies) if latencies else 0,
        }
        
        # Source metrics
        source_relevances = [
            r.get("source_relevance", {}).get("avg_relevance", 0)
            for r in successful
            if r.get("source_relevance")
        ]
        if source_relevances:
            summary["avg_source_relevance"] = sum(source_relevances) / len(source_relevances)
        
        # Answer metrics
        answer_similarities = [
            r.get("answer_similarity", 0)
            for r in successful
            if r.get("answer_similarity")
        ]
        if answer_similarities:
            summary["avg_answer_similarity"] = sum(answer_similarities) / len(answer_similarities)
        
        summary["detailed_results"] = self.results
        
        return summary


# =============================================================================
# Test Questions
# =============================================================================

DEFAULT_TEST_QUESTIONS = [
    {
        "question": "What are the main causes of climate change?",
        "expected_keywords": ["greenhouse", "emissions", "carbon", "fossil fuels"],
    },
    {
        "question": "How does deforestation contribute to global warming?",
        "expected_keywords": ["carbon", "trees", "CO2", "sink"],
    },
    {
        "question": "What is the Paris Agreement?",
        "expected_keywords": ["2015", "temperature", "1.5", "2 degrees", "nations"],
    },
    {
        "question": "What are ESG criteria?",
        "expected_keywords": ["environmental", "social", "governance"],
    },
    {
        "question": "Biến đổi khí hậu ảnh hưởng đến Việt Nam như thế nào?",
        "expected_keywords": ["nước biển", "bão", "lũ lụt", "Đồng bằng sông Cửu Long"],
    },
    {
        "question": "How can individuals reduce their carbon footprint?",
        "expected_keywords": ["energy", "transportation", "reduce", "recycle"],
    },
    {
        "question": "What is carbon neutrality?",
        "expected_keywords": ["emissions", "offset", "balance", "net zero"],
    },
    {
        "question": "Việt Nam cam kết gì về khí nhà kính?",
        "expected_keywords": ["giảm", "2030", "phát thải", "cam kết"],
    },
]


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Climate Q&A RAG System"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="JSON file with test questions",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with default questions",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Setup LangSmith
    settings.setup_langsmith()
    
    # Load test questions
    if args.test_file:
        with open(args.test_file) as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {args.test_file}")
    else:
        questions = DEFAULT_TEST_QUESTIONS
        logger.info(f"Using {len(questions)} default test questions")
    
    # Initialize chain
    logger.info("Initializing RAG chain...")
    try:
        embeddings = get_embedding_model()
        manager = load_existing_index(embeddings=embeddings)
        
        chain = AdvancedRAGChain(
            vector_store=manager.vector_store,
            enable_memory=False,
            language="auto",
        )
    except Exception as e:
        logger.error(f"Failed to initialize chain: {e}")
        sys.exit(1)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    evaluator = RAGEvaluator(chain)
    summary = evaluator.run_evaluation(questions)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Avg Latency: {summary['avg_latency_seconds']:.2f}s")
    
    if summary.get('avg_source_relevance'):
        print(f"Avg Source Relevance: {summary['avg_source_relevance']:.2f}")
    
    print("=" * 60)
    
    # Save results
    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Return exit code based on success rate
    if summary['success_rate'] < 0.8:
        logger.warning("Success rate below 80%")
        sys.exit(1)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
