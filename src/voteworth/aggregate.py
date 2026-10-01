"""Independence-weighted aggregation for LLM voter and judge panels.

The problem this solves
----------------------
When several LLM writers, judges or samples vote on an answer, the usual move is a plain
majority: one voter, one vote. That silently assumes the voters are independent. They are
not. Voters drawn from the same model — or the same model sampled several times — share
their *error modes*: when they are wrong, they tend to be wrong in the same way. A
correlated bloc then wins the majority while being wrong, and the agreement among its
members is read as confidence.

This module measures that correlation and corrects for it.

`learn_independence_weights` looks at a vote history and, for each pair of voters, asks:
of the items where both deviated from the panel's plurality, how often did they deviate
with the *same* value? A voter that shares its errors with others is redundant and gets
down-weighted; a voter that errs in its own way keeps full weight.

The structural limit, stated up front
-------------------------------------
If every voter in a panel is the same model, this repair does nothing. All voters receive
the same weight, normalisation makes it uniform, and the weighted result is identical to
the naive one. That is not a tuning problem — it is what the method can and cannot do, and
`examples/quickstart.py` prints it as `INERT`. A panel that cannot disagree independently
cannot be repaired by reweighting; it has to be made diverse.

Everything here is deterministic, pure logic, and has no I/O and no LLM calls.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

# Answers that carry no information must never form a winning bloc → sentinel cluster -1.
_NULL_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "i dont know",
    "n/a",
    "na",
    "none",
    "no answer",
}


def _is_null(a: str) -> bool:
    return (a or "").strip().lower().strip(".") in _NULL_ANSWERS


def _plurality(votes) -> int:
    """Most common value; ties broken to the smallest value (deterministic)."""
    counts: dict[int, int] = defaultdict(int)
    for v in votes:
        counts[v] += 1
    return max(sorted(counts), key=lambda v: counts[v])


def _co_deviation_counts(history, n_voters: int):
    """Shared internal of the weight learner and the diagnostic.

    For each history item: find the voters present, take the plurality, and look only at the
    voters that DEVIATE from it. For every deviating pair, count that they co-deviated
    (`co_both`) and whether they did so with the same value (`co_same`) — a shared error mode.

    Returns (co_same, co_both), both n×n integer matrices.
    """
    n = n_voters
    co_same = [[0] * n for _ in range(n)]
    co_both = [[0] * n for _ in range(n)]
    for votes in history:
        present = [i for i in range(n) if votes[i] != -1]
        if len(present) < 2:
            continue
        plur = _plurality([votes[i] for i in present])
        deviating = [i for i in present if votes[i] != plur]
        for a in range(len(deviating)):
            for b in range(a + 1, len(deviating)):
                i, j = deviating[a], deviating[b]
                co_both[i][j] += 1
                co_both[j][i] += 1
                if votes[i] == votes[j]:
                    co_same[i][j] += 1
                    co_same[j][i] += 1
    return co_same, co_both


def learn_independence_weights(history, n_voters: int):
    """Learn a per-voter independence weight from a held-out vote history.

    `history` is a list of vote vectors, each of length `n_voters`, holding one cluster id per
    voter per item (-1 = abstain/null, ignored for that item).

    weight[i] = 1 / (1 + R[i]) with R[i] = Σ_{j≠i} co_same[i][j] / co_both[i][j].

    A voter that never co-deviates has R = 0 and keeps weight 1.0. A voter locked to one
    partner (always the same wrong answer together) has R = 1 and drops to 0.5.

    An empty history, or one that never produces two simultaneous deviations, yields uniform
    weights — nothing was observed, so nothing is claimed.
    """
    co_same, co_both = _co_deviation_counts(history, n_voters)
    weights = []
    for i in range(n_voters):
        r = 0.0
        for j in range(n_voters):
            if j != i and co_both[i][j] > 0:
                r += co_same[i][j] / co_both[i][j]
        weights.append(1.0 / (1.0 + r))
    return weights


def pairwise_coerror(history, n_voters: int):
    """Diagnostic: the shared-error rate for every voter pair.

    Of the items where both voters deviated from the plurality, the fraction where they
    deviated with the *same* value. High values mark a correlated bloc.

    Returns an n×n matrix, `np.nan` on the diagonal and wherever a pair never co-deviated —
    an undefined rate is reported as undefined, never as zero.
    """
    co_same, co_both = _co_deviation_counts(history, n_voters)
    m = np.full((n_voters, n_voters), np.nan)
    for i in range(n_voters):
        for j in range(n_voters):
            if i != j and co_both[i][j] > 0:
                m[i][j] = co_same[i][j] / co_both[i][j]
    return m


def aggregate(votes, weights) -> int:
    """Weighted plurality over cluster ids.

    `votes` is a list of cluster ids (-1 is ignored). Returns the winning id, or -1 if no
    valid vote was cast. Ties break to the smallest id, deterministically.
    """
    acc: dict[int, float] = defaultdict(float)
    for v, w in zip(votes, weights):
        if v != -1:
            acc[v] += w
    if not acc:
        return -1
    return max(sorted(acc), key=lambda v: acc[v])


def naive_weights(n: int):
    """Uniform weights — the plain-majority baseline (one voter, one vote)."""
    return [1.0] * n


def structural_weights(sources):
    """Provenance weight without any history: 1 / (number of voters sharing a source label).

    `sources` is one label per voter — a model id, a vendor, or any grouping you consider a
    shared error source. Three voters labelled "gpt-4o-mini" contribute 1/3 each, so the bloc
    counts once against a single independent voter.

    Use this when you have no vote history. It encodes what you already know about where the
    voters came from, rather than what they have been observed to do.
    """
    counts = Counter(sources)
    return [1.0 / counts[s] for s in sources]


def effective_votes(history, n_voters: int, mode: str = "raw") -> float:
    """How many independent voters your panel is actually worth (Kish's effective sample size).

    n_eff = m / (1 + (m-1) * r̄), where r̄ is the mean pairwise agreement rate. A panel of ten
    voters that all say the same thing is worth one; a panel of ten that never covary is worth
    ten. Comparable to the figure reported for LLM judge panels in the literature.

    `mode` selects what "agreeing" means:
      "raw"   — the two voters cast the same vote (no labels needed; the default).
      "error" — the two voters are both right or both wrong. Needs `truth`, so pass a history
                of booleans instead; see the demo.

    **Read this together with the weights.** The two numbers answer different questions and
    disagree in exactly the case this package is weakest at: when one bloc holds the plurality,
    n_eff keeps falling while the learned weights go flat, because a bloc that *is* the
    plurality never deviates from it and so never registers as correlated. A low n_eff beside
    uniform weights is the signature of that blind spot — see `bloc_share`.

    Caveat, stated because it is easy to over-read: Kish's formula is defined for an
    intraclass correlation. Mean pairwise agreement is a stand-in for it, and a coarse one for
    categorical labels. **The ordering carries; the second decimal does not.**

    Both modes run the same computation; "error" only means the history holds booleans
    (right/wrong) instead of votes. Any other value of `mode` raises, so that a typo cannot
    pass silently.
    """
    if mode not in ("raw", "error"):
        raise ValueError(f'mode must be "raw" or "error", got {mode!r}')
    if n_voters < 2:
        return float(n_voters)
    rates = []
    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            both = agree = 0
            for votes in history:
                a, b = votes[i], votes[j]
                if a == -1 or b == -1:
                    continue
                both += 1
                agree += a == b
            if both:
                rates.append(agree / both)
    if not rates:
        return float(n_voters)
    r_bar = sum(rates) / len(rates)
    return n_voters / (1.0 + (n_voters - 1) * r_bar)


def bloc_share(sources) -> float:
    """The share of the panel held by its largest provenance group.

    Above 0.5 one group can carry the plurality on its own, and the shared-error statistic this
    package relies on stops seeing it: the bloc never deviates from a plurality it constitutes,
    so it accrues no penalty, while the independent minority co-deviates and is down-weighted.
    Measured on a synthetic 10-voter panel with a 6-voter bloc, the weighted vote equals the
    naive one and lands 20 points below the best single voter.

    Check this before trusting the weights. There is no correction for it in this package.
    """
    if not sources:
        return 0.0
    counts = Counter(sources)
    return max(counts.values()) / len(sources)


def agreement_confidence(votes, weights=None) -> float:
    """The share of the (weighted) vote mass that went to the winner.

    This is the quantity most panels report as "confidence". It is returned here so that the
    gap between it and measured accuracy can be computed — see `examples/quickstart.py`.
    Reporting it without that gap is exactly the failure this package is about.
    """
    valid = [(v, w) for v, w in zip(votes, weights or naive_weights(len(votes))) if v != -1]
    if not valid:
        return 0.0
    total = sum(w for _, w in valid)
    if total <= 0:
        return 0.0
    winner = aggregate([v for v, _ in valid], [w for _, w in valid])
    return sum(w for v, w in valid if v == winner) / total


def cluster_answers(answers, embedder, thresh: float = 0.78):
    """Free-text answers → equivalence-class ids (the discrete votes), by embedding cosine.

    Greedy single-link: each answer joins the first cluster whose centroid it matches at
    cosine ≥ `thresh`, else starts a new one. Null answers get id -1. Deterministic in input
    order.

    `embedder` is any object with `.encode(list[str], normalize_embeddings=True) -> ndarray`.

    Caution, learned the hard way: for short *factual* answers, embedding cosine conflates
    distinct facts — two different dates sit above any paraphrase threshold. Use
    `cluster_answers_by_fn` there.

    Returns (labels, reps): the cluster id per answer, and one representative string per
    cluster (its first member).
    """
    labels = [-1] * len(answers)
    reps: dict[int, str] = {}
    live = [i for i, a in enumerate(answers) if not _is_null(a)]
    if not live:
        return labels, reps
    vecs = np.asarray(embedder.encode([answers[i] for i in live], normalize_embeddings=True))
    centroids: list[np.ndarray] = []
    cids: list[int] = []
    nxt = 0
    for n, i in enumerate(live):
        v = vecs[n]
        best_c, best_s = -1, -1.0
        for ci, cen in enumerate(centroids):
            s = float(v @ cen)
            if s > best_s:
                best_s, best_c = s, ci
        if best_c >= 0 and best_s >= thresh:
            labels[i] = cids[best_c]
        else:
            labels[i] = nxt
            reps[nxt] = answers[i]
            centroids.append(v)
            cids.append(nxt)
            nxt += 1
    return labels, reps


def cluster_answers_by_fn(answers, equal_fn):
    """Free-text answers → equivalence-class ids, by an equality callback.

    `equal_fn(a, b) -> bool` decides same-meaning; in production this is usually an LLM judge.
    Identical normalised strings short-circuit to equal *without* calling it. Null answers get
    id -1. Deterministic in input order.

    This is the accurate path for short factual answers, where embedding cosine merges
    distinct facts.

    Returns (labels, reps).
    """

    def norm(a: str) -> str:
        return (a or "").strip().lower().strip(".")

    labels = [-1] * len(answers)
    reps: dict[int, str] = {}
    rep_idx: list[int] = []
    cids: list[int] = []
    nxt = 0
    for i, a in enumerate(answers):
        if _is_null(a):
            continue
        placed = False
        for ci, ri in enumerate(rep_idx):
            if norm(a) == norm(answers[ri]) or equal_fn(a, answers[ri]):
                labels[i] = cids[ci]
                placed = True
                break
        if not placed:
            labels[i] = nxt
            reps[nxt] = a
            rep_idx.append(i)
            cids.append(nxt)
            nxt += 1
    return labels, reps
