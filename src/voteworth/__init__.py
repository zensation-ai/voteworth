"""Independence-weighted aggregation for LLM voter and judge panels.

Shows how often two voters that both break from the plurality pick the same answer, reweights
the vote for voters that do, and says where that cannot work.
"""

from voteworth.aggregate import (
    agreement_confidence,
    aggregate,
    bloc_share,
    cluster_answers,
    cluster_answers_by_fn,
    effective_votes,
    learn_independence_weights,
    naive_weights,
    pairwise_coerror,
    structural_weights,
)

__version__ = "0.1.4"

__all__ = [
    "agreement_confidence",
    "aggregate",
    "bloc_share",
    "cluster_answers",
    "cluster_answers_by_fn",
    "effective_votes",
    "learn_independence_weights",
    "naive_weights",
    "pairwise_coerror",
    "structural_weights",
]
