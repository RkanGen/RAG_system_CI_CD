import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

from src.core.config import settings
from src.pipeline.retrieval import get_retrieval_pipeline
from src.pipeline.generation import get_generation_pipeline
from src.api.schemas import EvaluationQuery, EvaluationResult

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Calculate retrieval evaluation metrics."""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if not retrieved_docs or k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        relevant_set = set(relevant_docs)
        
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_k = retrieved_docs[:k]
        
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Hit Rate (whether any relevant doc was retrieved)."""
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        return 1.0 if retrieved_set.intersection(relevant_set) else 0.0


class GenerationMetrics:
    """Calculate generation evaluation metrics."""
    
    @staticmethod
    def calculate_faithfulness(answer: str, context: str) -> float:
        """
        Calculate faithfulness score (answer grounded in context).
        Placeholder - in production, use NLI model.
        """
        # Simple heuristic: check if answer tokens appear in context
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        overlap = answer_tokens.intersection(context_tokens)
        return len(overlap) / len(answer_tokens)
    
    @staticmethod
    def calculate_relevance(answer: str, query: str) -> float:
        """
        Calculate answer relevance to query.
        Placeholder - in production, use semantic similarity.
        """
        # Simple heuristic: check token overlap
        answer_tokens = set(answer.lower().split())
        query_tokens = set(query.lower().split())
        
        if not query_tokens:
            return 0.0
        
        overlap = answer_tokens.intersection(query_tokens)
        return len(overlap) / len(query_tokens)
    
    @staticmethod
    def calculate_context_precision(
        retrieved_docs: List[str],
        relevant_docs: List[str],
    ) -> float:
        """Calculate context precision (retrieved docs are relevant)."""
        if not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_set)
        
        return relevant_retrieved / len(retrieved_docs)
    
    @staticmethod
    def calculate_context_recall(
        retrieved_docs: List[str],
        relevant_docs: List[str],
    ) -> float:
        """Calculate context recall (all relevant docs retrieved)."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_set)
        
        return relevant_retrieved / len(relevant_docs)


class GoldenDatasetLoader:
    """Load and manage golden evaluation dataset."""
    
    def __init__(self, dataset_path: str = settings.golden_dataset_path):
        self.dataset_path = Path(dataset_path)
    
    def load_dataset(self) -> List[EvaluationQuery]:
        """Load golden dataset from JSONL file."""
        if not self.dataset_path.exists():
            logger.warning(f"Golden dataset not found at {self.dataset_path}")
            return []
        
        queries = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    query = EvaluationQuery(**data)
                    queries.append(query)
        
        logger.info(f"Loaded {len(queries)} queries from golden dataset")
        return queries
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """Save evaluation results to file."""
        if output_path is None:
            output_path = f"evaluation_results_{datetime.utcnow().isoformat()}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {output_path}")


class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self):
        self.retrieval_pipeline = get_retrieval_pipeline()
        self.generation_pipeline = get_generation_pipeline()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.dataset_loader = GoldenDatasetLoader()
    
    async def evaluate_retrieval(
        self,
        query: str,
        expected_doc_ids: List[str],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """Evaluate retrieval for a single query."""
        # Retrieve documents
        retrieval_result = await self.retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
        )
        
        # Extract document IDs
        retrieved_doc_ids = [chunk.document_id for chunk in retrieval_result.chunks]
        
        # Calculate metrics
        metrics = {
            "precision@1": self.retrieval_metrics.precision_at_k(retrieved_doc_ids, expected_doc_ids, 1),
            "precision@3": self.retrieval_metrics.precision_at_k(retrieved_doc_ids, expected_doc_ids, 3),
            "precision@5": self.retrieval_metrics.precision_at_k(retrieved_doc_ids, expected_doc_ids, 5),
            "recall@5": self.retrieval_metrics.recall_at_k(retrieved_doc_ids, expected_doc_ids, 5),
            "mrr": self.retrieval_metrics.mean_reciprocal_rank(retrieved_doc_ids, expected_doc_ids),
            "ndcg@5": self.retrieval_metrics.ndcg_at_k(retrieved_doc_ids, expected_doc_ids, 5),
            "hit_rate": self.retrieval_metrics.hit_rate(retrieved_doc_ids, expected_doc_ids),
        }
        
        return metrics
    
    async def evaluate_generation(
        self,
        query: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate generation for a single query."""
        # Retrieve and generate
        retrieval_result = await self.retrieval_pipeline.retrieve(query)
        generation_result = await self.generation_pipeline.generate(
            query=query,
            retrieval_result=retrieval_result,
            stream=False,
        )
        
        # Build context
        context = "\n".join([chunk.content for chunk in retrieval_result.chunks])
        
        # Calculate metrics
        metrics = {
            "faithfulness": self.generation_metrics.calculate_faithfulness(
                generation_result.answer,
                context,
            ),
            "relevance": self.generation_metrics.calculate_relevance(
                generation_result.answer,
                query,
            ),
        }
        
        # If expected answer provided, calculate similarity
        if expected_answer:
            # Placeholder - in production, use semantic similarity
            metrics["answer_similarity"] = 0.0
        
        return metrics
    
    async def run_full_evaluation(
        self,
        queries: Optional[List[EvaluationQuery]] = None,
    ) -> EvaluationResult:
        """Run full evaluation on golden dataset."""
        if queries is None:
            queries = self.dataset_loader.load_dataset()
        
        if not queries:
            logger.warning("No queries to evaluate")
            return EvaluationResult(
                timestamp=datetime.utcnow(),
                metrics={},
                query_results=[],
                total_queries=0,
                passed_queries=0,
                failed_queries=0,
            )
        
        logger.info(f"Running evaluation on {len(queries)} queries...")
        
        all_metrics = {
            "precision@1": [],
            "precision@3": [],
            "precision@5": [],
            "recall@5": [],
            "mrr": [],
            "ndcg@5": [],
            "hit_rate": [],
            "faithfulness": [],
            "relevance": [],
        }
        
        query_results = []
        passed = 0
        failed = 0
        
        for eval_query in queries:
            try:
                # Evaluate retrieval
                retrieval_metrics = await self.evaluate_retrieval(
                    query=eval_query.query,
                    expected_doc_ids=eval_query.expected_doc_ids,
                )
                
                # Evaluate generation
                generation_metrics = await self.evaluate_generation(
                    query=eval_query.query,
                    expected_answer=eval_query.expected_answer,
                )
                
                # Combine metrics
                combined_metrics = {**retrieval_metrics, **generation_metrics}
                
                # Store results
                query_results.append({
                    "query_id": eval_query.query_id,
                    "query": eval_query.query,
                    "metrics": combined_metrics,
                })
                
                # Aggregate metrics
                for key, value in combined_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
                
                # Check if passed (simple threshold)
                if combined_metrics.get("precision@5", 0) >= 0.6:
                    passed += 1
                else:
                    failed += 1
            
            except Exception as e:
                logger.error(f"Error evaluating query {eval_query.query_id}: {e}")
                failed += 1
        
        # Calculate aggregate metrics
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            if values:
                aggregated_metrics[f"avg_{key}"] = np.mean(values)
                aggregated_metrics[f"median_{key}"] = np.median(values)
                aggregated_metrics[f"std_{key}"] = np.std(values)
        
        result = EvaluationResult(
            timestamp=datetime.utcnow(),
            metrics=aggregated_metrics,
            query_results=query_results,
            total_queries=len(queries),
            passed_queries=passed,
            failed_queries=failed,
        )
        
        # Save results
        self.dataset_loader.save_results(result.dict())
        
        logger.info(f"Evaluation completed: {passed}/{len(queries)} passed")
        
        return result


# Global evaluator instance
_evaluator: Optional[Evaluator] = None


def get_evaluator() -> Evaluator:
    """Get or create evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator