"""Independence-weighted aggregation for LLM voter and judge panels.

Tells you whether your panel can produce an independent majority — and by how much its
agreement-confidence is overstated when it cannot.
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

__version__ = "0.1.0"

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
