"""Contracts the docstrings state and no test bound until 0.1.4.

A check of the release artefact against every statement it makes (2026-11-21) found three
docstring passages that described the code differently from what it does: the module said the
learned weight is inert on any same-model panel, `cluster_answers` said it joins the first
cluster by its centroid, and `effective_votes` said the weights go uniform in the regime where
one bloc holds the plurality. The docstrings now say what the code does; these tests hold them
to it.
"""
from __future__ import annotations

import math
import random

import numpy as np

from test_aggregate import _bloc_panel
from voteworth import (
    aggregate,
    cluster_answers,
    learn_independence_weights,
    naive_weights,
    structural_weights,
)


# ── the structural limit: provenance is inert on one model, the learned weight from four up ──


def test_learned_weights_are_uniform_for_three_voters_of_one_model():
    """With three voters, two that deviate together with the same value ARE the plurality, so
    no pair can share an error in the sense the weight counts: the learned weight cannot move."""
    rng = random.Random(20261121)
    for _ in range(3000):
        history = [[rng.choice([-1, 0, 1, 2, 3]) for _ in range(3)] for _ in range(rng.randint(1, 12))]
        w = learn_independence_weights(history, 3)
        assert max(w) - min(w) < 1e-12, (history, w)


def test_learned_weights_can_differ_from_four_voters_of_one_model():
    """From four voters up the learned weight is not bound by provenance: it separates voters of
    one model that co-deviate, and it can change the winner, while provenance stays uniform."""
    assert learn_independence_weights([[0, 0, 1, 1]], 4) == [1.0, 1.0, 0.5, 0.5]

    history = [[0, 0, 0, 1, 1]] * 6 + [[0, 0, 0, 2, 2]] * 4 + [[1, 1, 1, 0, 0]] * 3
    learned = learn_independence_weights(history, 5)
    assert learned == [1.0, 1.0, 1.0, 0.5, 0.5]
    votes = [5, 5, 6, 0, 0]
    assert aggregate(votes, naive_weights(5)) == 0  # tie 2:2, broken to the smallest id
    assert aggregate(votes, learned) == 5
    provenance = structural_weights(["one-model"] * 5)
    assert aggregate(votes, provenance) == aggregate(votes, naive_weights(5))


# ── cluster_answers: the most similar first member, not the first cluster, not a centroid ──


class _AngleEmbedder:
    """Unit vectors at fixed angles, so that every cosine in these tests is known in advance."""

    def __init__(self, degrees):
        self.degrees = degrees

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[math.cos(math.radians(self.degrees[t])), math.sin(math.radians(self.degrees[t]))]
                         for t in texts])


def test_cluster_answers_joins_the_most_similar_cluster_not_the_first():
    """A at 0 and B at 60 degrees are two clusters (cosine 0.5). x at 32 degrees matches both
    above 0.78: A at 0.848, B at 0.883. It joins B, the most similar, not A, the first."""
    labels, _ = cluster_answers(["A", "B", "x"], _AngleEmbedder({"A": 0, "B": 60, "x": 32}), thresh=0.78)
    assert labels == [0, 1, 1]


def test_cluster_answers_compares_with_the_first_member():
    """a2 at 36 degrees joins A at 0 (cosine 0.809). z at 40 degrees: cosine 0.766 to A, the
    first member, so it starts a new cluster. Against a centroid (18 degrees, cosine 0.927) or
    against any member (a2, cosine 0.998) it would have joined."""
    labels, reps = cluster_answers(["A", "a2", "z"], _AngleEmbedder({"A": 0, "a2": 36, "z": 40}), thresh=0.78)
    assert labels == [0, 0, 1]
    assert reps == {0: "A", 1: "z"}


# ── effective_votes: the signature of the blind spot is a bloc at full weight ──


def test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not():
    """Where one bloc holds the plurality, its learned weight returns to 1.0 while the
    independent voters are down-weighted: the weights are not uniform. Below the plurality
    the bloc is the one down-weighted."""
    for n_bloc, holds_plurality in ((3, False), (5, False), (6, True), (8, True)):
        bloc_w, indep_w = [], []
        for seed in range(5):
            w = learn_independence_weights(_bloc_panel(n_bloc, 10 - n_bloc, seed=seed), 10)
            bloc_w.append(sum(w[:n_bloc]) / n_bloc)
            indep_w.append(sum(w[n_bloc:]) / (10 - n_bloc))
        mean_bloc, mean_indep = sum(bloc_w) / 5, sum(indep_w) / 5
        if holds_plurality:
            assert mean_bloc == 1.0, (n_bloc, mean_bloc)
            assert mean_indep < 0.9, (n_bloc, mean_indep)
        else:
            assert mean_bloc < mean_indep, (n_bloc, mean_bloc, mean_indep)
