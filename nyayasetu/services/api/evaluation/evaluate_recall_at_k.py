"""Evaluation script for calculating Recall@K and MRR for the HybridRetriever.

This script evaluates how accurately the retriever finds relevant legal sections
for a benchmark set of legal queries.

Metrics:
- Recall@K: Proportion of queries where the target section is within top-K.
- MRR (Mean Reciprocal Rank): Mean of reciprocal ranks of the first target match.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from dataclasses import asdict, dataclass
from typing import Sequence

from app.rag.retriever import HybridRetriever, RetrievalFilter, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalQuery:
    """A test case in the retrieval evaluation benchmark."""

    query_id: str
    query_text: str
    target_act: str
    target_section: str
    jurisdiction: str = "IN"
    as_of_date: str | None = None


@dataclass
class EvalMetrics:
    """Evaluation metrics summary."""

    total_queries: int
    recall_at_k: dict[int, float]
    mrr: float

    def to_dict(self) -> dict:
        return asdict(self)


class RecallEvaluator:
    """Evaluates retrieval performance against ground truth benchmark data."""

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values = k_values or [1, 3, 5, 10, 20]

    async def evaluate_query(
        self,
        retriever: HybridRetriever,
        item: EvalQuery,
        max_k: int = 20,
    ) -> tuple[dict[int, bool], float]:
        """Evaluate a single query and return (hits_at_k, reciprocal_rank)."""
        as_of = (
            datetime.date.fromisoformat(item.as_of_date)
            if item.as_of_date
            else None
        )
        filter_ = RetrievalFilter(jurisdiction=item.jurisdiction, as_of_date=as_of)

        results: list[SearchResult] = await retriever.search(
            query=item.query_text,
            filter_=filter_,
            top_k=max_k,
        )

        hits_at_k: dict[int, bool] = {k: False for k in self.k_values}
        reciprocal_rank = 0.0

        for rank, res in enumerate(results, start=1):
            act_match = res.act_name.lower().strip() == item.target_act.lower().strip()
            sec_match = (
                res.section_number.lower().strip() == item.target_section.lower().strip()
            )

            if act_match and sec_match:
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank

                for k in self.k_values:
                    if rank <= k:
                        hits_at_k[k] = True

        return hits_at_k, reciprocal_rank

    async def evaluate_dataset(
        self,
        retriever: HybridRetriever,
        dataset: Sequence[EvalQuery],
    ) -> EvalMetrics:
        """Evaluate a dataset of benchmark queries and calculate aggregate metrics."""
        if not dataset:
            return EvalMetrics(
                total_queries=0,
                recall_at_k={k: 0.0 for k in self.k_values},
                mrr=0.0,
            )

        max_k = max(self.k_values)
        total_queries = len(dataset)
        hits_count: dict[int, int] = {k: 0 for k in self.k_values}
        mrr_sum = 0.0

        for item in dataset:
            hits_at_k, rr = await self.evaluate_query(retriever, item, max_k=max_k)
            mrr_sum += rr
            for k, is_hit in hits_at_k.items():
                if is_hit:
                    hits_count[k] += 1

        recall_at_k = {k: hits_count[k] / total_queries for k in self.k_values}
        mrr = mrr_sum / total_queries

        return EvalMetrics(
            total_queries=total_queries,
            recall_at_k=recall_at_k,
            mrr=mrr,
        )

    @staticmethod
    def print_report(metrics: EvalMetrics) -> str:
        """Format metrics into a clear markdown report."""
        lines = [
            "### Hybrid Retrieval Benchmark Results",
            f"**Total Benchmark Queries:** {metrics.total_queries}",
            f"**Mean Reciprocal Rank (MRR):** {metrics.mrr:.4f}",
            "",
            "| Metric | Recall Score | Percentage |",
            "|--------|--------------|------------|",
        ]
        for k, score in metrics.recall_at_k.items():
            lines.append(f"| Recall@{k:<2} | {score:.4f}       | {score * 100:.1f}%      |")

        report = "\n".join(lines)
        print(report)
        return report


# ── Sample Benchmark Queries for Evaluation ──────────────────────────────────
SAMPLE_GOLDEN_SET = [
    EvalQuery(
        query_id="q1",
        query_text="What are the essential elements of a valid legal contract?",
        target_act="Indian Contract Act",
        target_section="10",
        jurisdiction="IN",
    ),
    EvalQuery(
        query_id="q2",
        query_text="When is an agreement declared void for uncertainty?",
        target_act="Indian Contract Act",
        target_section="29",
        jurisdiction="IN",
    ),
    EvalQuery(
        query_id="q3",
        query_text="What is the period of limitation for filing a suit on a contract?",
        target_act="Limitation Act",
        target_section="5",
        jurisdiction="IN",
    ),
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Recall@K Evaluator script initialized. To run against a live database, inject a configured HybridRetriever.")
