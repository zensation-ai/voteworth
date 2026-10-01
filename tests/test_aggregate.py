"""Unit tests for independence-weighted aggregation.

Deterministic, synthetic data with known ground truth, so the maths is verified rather than
sampled. Two tests carry the package's claims:

* `test_weighted_beats_naive_on_detectable_bloc` — the failure this package exists for: a
  correlated bloc wins a plain majority while being wrong, and the learned weighting recovers
  the truth.
* `test_same_model_panel_is_inert` — the package's stated *limit*: on a panel that is one
  model throughout, the repair provably does nothing. A claim and its boundary should both be
  tests, not prose.

Whether these tests can actually fail is itself checked, in `test_mutation.py`.
"""
from __future__ import annotations

import random

import numpy as np

from voteworth import (
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
from voteworth.aggregate import _plurality


class FakeEmbedder:
    """Deterministic encoder: maps each answer to a one-hot unit vector by the first matching
    keyword, so paraphrases of one fact share a direction (cosine 1) and different facts are
    orthogonal (cosine 0). Anything unmatched gets its own direction."""

    def __init__(self, keywords):
        self.keywords = keywords
        self.dim = len(keywords) + 1

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), self.dim))
        for r, t in enumerate(texts):
            tl = t.lower()
            idx = next((k for k, kw in enumerate(self.keywords) if kw in tl), self.dim - 1)
            out[r, idx] = 1.0
        return out


# ── plurality + naive aggregation ──────────────────────────────────────────────


def test_plurality_and_ties():
    assert _plurality([0, 0, 1]) == 0
    assert _plurality([2, 2, 5, 5, 1]) in (2, 5)
    assert _plurality([5, 2]) == 2  # tie → smallest


def test_aggregate_naive_basic():
    assert aggregate([0, 0, 1], naive_weights(3)) == 0
    assert aggregate([0, 1, 2], naive_weights(3)) == 0  # all-singleton tie → smallest id
    assert aggregate([-1, -1], naive_weights(2)) == -1  # all-null → no winner
    assert aggregate([], []) == -1


def test_null_votes_ignored():
    assert aggregate([-1, 3, -1], naive_weights(3)) == 3


# ── structural (no-history) provenance weights ─────────────────────────────────


def test_structural_weights_downweight_shared_source():
    w = structural_weights(["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini", "haiku"])
    assert w == [1 / 3, 1 / 3, 1 / 3, 1.0]
    # the 3-voter bloc now carries 1.0 in total — equal to the lone independent voter
    assert abs(sum(w[:3]) - w[3]) < 1e-9


# ── learned independence weights ───────────────────────────────────────────────


def _bloc_history():
    """Voters 0 and 1 are a correlated bloc: on four items they deviate from the (correct)
    plurality with the SAME wrong value. Voters 2, 3, 4 always vote the plurality. One clean
    unanimous item at the end."""
    return [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_independence_weights_downweight_correlated_bloc():
    w = learn_independence_weights(_bloc_history(), 5)
    # bloc voters co-err with each other at rate 1.0 → R = 1 → weight 0.5
    assert abs(w[0] - 0.5) < 1e-9
    assert abs(w[1] - 0.5) < 1e-9
    # independent voters never co-deviate → R = 0 → weight 1.0
    assert w[2] == 1.0 and w[3] == 1.0 and w[4] == 1.0
    assert w[0] < w[2] and w[1] < w[3]


def test_weighted_beats_naive_on_detectable_bloc():
    """The load-bearing test. A correlated bloc wins the naive vote and is wrong; the learned
    weighting demotes it and the truth survives."""
    w = learn_independence_weights(_bloc_history(), 5)
    # bloc (0,1) agrees on wrong answer 9; voter 2 holds the truth 0; voters 3,4 are each
    # independently wrong. Naive plurality = 9, the two-vote bloc. Truth = cluster 0.
    test = [9, 9, 0, 7, 8]
    assert aggregate(test, naive_weights(5)) == 9  # naive is dragged wrong by the bloc
    assert aggregate(test, w) != 9  # weighting refuses the correlated bloc
    assert aggregate(test, w) == 0  # and lands on the truth


def test_no_history_no_penalty():
    assert learn_independence_weights([], 3) == [1.0, 1.0, 1.0]
    assert learn_independence_weights([[0]], 1) == [1.0]


def test_pairwise_coerror_signal():
    m = pairwise_coerror(_bloc_history(), 5)
    assert abs(m[0][1] - 1.0) < 1e-9  # bloc: always the same wrong value
    assert np.isnan(m[2][3])  # independents never co-deviate → undefined, not zero
    assert np.isnan(m[0][0])  # diagonal undefined


# ── the stated limit, as a test ────────────────────────────────────────────────


def test_same_model_panel_is_inert():
    """The package's boundary claim: a panel that is one model throughout cannot be repaired.

    Every voter shares the one source label, so structural weights are uniform after
    normalisation and the weighted winner equals the naive winner — for *every* vote pattern.
    If this test ever fails, the README's INERT claim is wrong and must be corrected.
    """
    sources = ["gpt-4o-mini"] * 5
    w = structural_weights(sources)
    assert len(set(w)) == 1  # uniform: nothing to redistribute

    for votes in ([9, 9, 0, 7, 8], [1, 1, 1, 2, 2], [0, 1, 2, 3, 4], [-1, 5, 5, -1, 3]):
        # the winner is an id — exact equality is the right assertion
        assert aggregate(votes, w) == aggregate(votes, naive_weights(5))
        # the confidence is a float ratio. Scaling every weight by 1/5 and dividing it out
        # again is algebraically the identity but not bit-identical (3 * 0.2 == 0.6 is False
        # in IEEE 754), so the claim is "equal to within rounding", and the tolerance is
        # tight enough that any real redistribution would break it.
        assert abs(agreement_confidence(votes, w) - agreement_confidence(votes)) < 1e-12


def test_learned_weights_also_inert_on_symmetric_panel():
    """A history in which every voter co-errs with every other at the same rate yields uniform
    learned weights too — the symmetry the method cannot break."""
    history = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    w = learn_independence_weights(history, 4)
    assert max(w) - min(w) < 1e-9


# ── the blind spot, as a test that DOCUMENTS the failure ───────────────────────


def _bloc_panel(n_bloc, n_indep, items=400, p_bloc_right=0.55, p_indep_right=0.70, seed=20260920):
    """A panel with one maximally correlated bloc: when the bloc is wrong, every member gives
    the SAME wrong answer. Independent voters each err their own way. Truth is always 0."""
    rng = random.Random(seed)
    history = []
    for _ in range(items):
        row = []
        bloc_right = rng.random() < p_bloc_right
        bloc_wrong = rng.randint(1, 9)
        row.extend([0 if bloc_right else bloc_wrong] * n_bloc)
        row.extend(0 if rng.random() < p_indep_right else rng.randint(10, 99) for _ in range(n_indep))
        history.append(row)
    return history


def test_dominant_bloc_is_invisible_to_the_weights():
    """⚠️ This test asserts that the package FAILS. It is not a bug to be fixed here — it is
    the documented boundary, and the test exists so that a future change cannot quietly alter
    it without someone noticing.

    The shared-error rate is measured among voters that *deviate from the plurality*. A bloc
    large enough to BE the plurality never deviates, so it never registers as correlated. The
    independent minority co-deviates instead and is penalised — the exact inversion of what
    the method is for.

    Asserted across twenty seeds, not one: a threshold read off a single draw is a number
    about that draw. Over seeds 0-19 the gap to the best single voter runs 0.132 to 0.218
    (median 0.172), so the assertion below is set under the observed minimum.
    """
    n_bloc, n_indep = 6, 4
    n = n_bloc + n_indep
    truth_gaps = []

    for seed in range(20):
        history = _bloc_panel(n_bloc, n_indep, seed=seed)
        w = learn_independence_weights(history, n)

        bloc_w = sum(w[:n_bloc]) / n_bloc
        indep_w = sum(w[n_bloc:]) / n_indep
        # The failure, asserted: the correlated bloc keeps FULL weight...
        assert bloc_w == 1.0, f"seed {seed}: bloc weight {bloc_w}"
        # ...while the independent voters are the ones penalised.
        assert indep_w < bloc_w, f"seed {seed}"

        truth = [0] * len(history)
        acc_naive = sum(aggregate(r, naive_weights(n)) == t for r, t in zip(history, truth)) / len(history)
        acc_w = sum(aggregate(r, w) == t for r, t in zip(history, truth)) / len(history)
        best = max(sum(r[i] == t for r, t in zip(history, truth)) / len(history) for i in range(n))

        assert abs(acc_naive - acc_w) < 1e-12, f"seed {seed}: weighting is not inert"
        truth_gaps.append(best - acc_w)

    assert min(truth_gaps) > 0.11, f"gaps: min={min(truth_gaps):.3f}"


def test_bloc_share_flags_the_regime():
    """The diagnostic that tells you whether you are in the regime above."""
    assert bloc_share(["a"] * 6 + ["b", "c", "d", "e"]) == 0.6
    assert bloc_share(["a", "a", "a", "b", "b", "b", "c", "c", "d", "d"]) == 0.3
    assert bloc_share([]) == 0.0
    assert bloc_share(["x"]) == 1.0


def test_effective_votes_falls_as_the_bloc_grows():
    """n_eff is the second number: it keeps dropping through the regime where the weights go
    flat. That divergence is how the blind spot becomes visible from outside."""
    n = 10
    seen = []
    for n_bloc in (3, 5, 6, 8):
        nes, bws = [], []
        for seed in range(5):
            history = _bloc_panel(n_bloc, n - n_bloc, seed=seed)
            nes.append(effective_votes(history, n))
            w = learn_independence_weights(history, n)
            bws.append(sum(w[:n_bloc]) / n_bloc)
        seen.append((n_bloc, sum(nes) / len(nes), sum(bws) / len(bws)))
    # n_eff is monotonically decreasing in bloc size
    assert all(seen[i][1] > seen[i + 1][1] for i in range(len(seen) - 1)), seen
    # but the bloc's weight goes UP once it owns the plurality — the two disagree
    assert seen[0][2] < seen[-1][2], seen
    assert seen[-1][2] == 1.0


def test_effective_votes_bounds():
    n = 4
    assert abs(effective_votes([[0, 0, 0, 0]] * 20, n) - 1.0) < 1e-9  # unanimous → 1
    disjoint = [[0, 1, 2, 3]] * 20
    assert abs(effective_votes(disjoint, n) - n) < 1e-9  # never agree → n
    assert effective_votes([], n) == n  # nothing observed, nothing claimed
    assert effective_votes([[0]], 1) == 1.0


def test_effective_votes_unknown_mode_raises():
    # Until 2026-09-30 `mode` was never read: a typo such as mode="erorr" returned the "raw"
    # figure without a word. An unknown mode must fail loudly.
    import pytest

    history = [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]]
    with pytest.raises(ValueError):
        effective_votes(history, 4, mode="bogus")


def test_effective_votes_error_mode_on_booleans():
    # "error" is the same computation on a history of right/wrong booleans.
    right_wrong = [[True, True, False, False], [True, False, True, False], [False, False, False, True]]
    assert effective_votes(right_wrong, 4, mode="error") == effective_votes(right_wrong, 4, mode="raw")


def test_effective_votes_default_mode_unchanged():
    history = [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]]
    assert abs(effective_votes(history, 4) - 1.8181818181818181) < 1e-12
    assert effective_votes(history, 4) == effective_votes(history, 4, mode="raw")


# ── agreement confidence ───────────────────────────────────────────────────────


def test_agreement_confidence_basic():
    assert agreement_confidence([0, 0, 0]) == 1.0
    assert abs(agreement_confidence([0, 0, 1]) - 2 / 3) < 1e-9
    assert agreement_confidence([-1, -1]) == 0.0
    assert agreement_confidence([]) == 0.0


def test_agreement_confidence_ignores_abstentions():
    # two real votes, one abstention → unanimity among those who voted
    assert agreement_confidence([0, 0, -1]) == 1.0


def test_agreement_confidence_drops_when_bloc_is_downweighted():
    """Down-weighting a correlated bloc must lower the reported confidence in its answer —
    that is the whole point of reporting it."""
    votes = [9, 9, 0, 7, 8]
    w = learn_independence_weights(_bloc_history(), 5)
    naive_conf = agreement_confidence(votes)  # bloc wins with 2/5
    weighted_conf = agreement_confidence(votes, w)
    assert naive_conf > 0
    assert weighted_conf < 1.0
    # the winner changed, so confidence now describes a different answer — assert it is honest
    assert aggregate(votes, w) != aggregate(votes, naive_weights(5))


# ── answer-equivalence clustering ──────────────────────────────────────────────


def test_cluster_answers_paraphrase_and_null():
    emb = FakeEmbedder(["paris", "london"])
    answers = ["Paris", "paris.", "The capital is Paris", "London", "unknown", ""]
    labels, reps = cluster_answers(answers, emb, thresh=0.78)
    assert labels[0] == labels[1] == labels[2]  # paraphrases → one cluster
    assert labels[3] != labels[0]
    assert labels[4] == -1 and labels[5] == -1  # null → sentinel
    assert reps[labels[0]] == "Paris"  # representative = first member


def test_cluster_answers_empty():
    emb = FakeEmbedder(["x"])
    assert cluster_answers([], emb) == ([], {})
    assert cluster_answers(["unknown", ""], emb) == ([-1, -1], {})


def test_cluster_answers_by_fn_callback():
    """The accurate path for factual answers: cluster by an equality callback (an LLM judge in
    production). Identical strings must short-circuit without calling it."""
    synonyms = [{"running", "jogging"}, {"7 may 2023", "may 7, 2023"}]

    def equal_fn(a, b):
        al, bl = a.lower().strip("."), b.lower().strip(".")
        return any(al in s and bl in s for s in synonyms)

    answers = ["running", "running", "jogging", "May 7, 2023", "7 May 2023", "Berlin", "unknown"]
    labels, _ = cluster_answers_by_fn(answers, equal_fn)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[5] not in (labels[0], labels[3])
    assert labels[6] == -1

    calls = {"n": 0}

    def never_equal(a, b):
        calls["n"] += 1
        return False

    labels2, _ = cluster_answers_by_fn(["Berlin", "Berlin", "Berlin"], never_equal)
    assert labels2 == [0, 0, 0] and calls["n"] == 0


def test_cluster_then_aggregate_endtoend():
    emb = FakeEmbedder(["paris", "london"])
    answers = ["London", "London", "Paris", "Paris", "unknown"]
    labels, reps = cluster_answers(answers, emb)
    win = aggregate(labels, naive_weights(len(answers)))
    assert reps[win] == "London"  # 2v2 tie → smallest id, London came first
    # if the two London voters are a known redundant bloc, weighting flips it
    assert reps[aggregate(labels, [0.3, 0.3, 1.0, 1.0, 1.0])] == "Paris"


def test_boundary_figures_match_the_readme():
    """The four n_eff values the README prints for the boundary must come out of a run.

    They were declared in FOREIGN_FIGURES as "reproduced by tests/test_aggregate.py", and the
    tests above assert only that the series falls — monotonicity, not values. So a wrong digit
    in the README was covered by a declaration rather than by a check, and two of the four
    were wrong: 1.88 for 1.9057 and 1.69 for 1.6845. An acceptance pass found it by
    recomputing them; nothing in the suite could.

    The README is the source, as everywhere else in this suite: the numbers are parsed out of
    it, not restated here, so editing the README to a value the code does not produce is red.
    """
    import re

    from shipped import ROOT, read_flat

    flat = read_flat(ROOT / "README.md")
    m = re.search(
        r"n_eff keeps falling as the bloc grows \("
        r"([\d.]+) → ([\d.]+) → ([\d.]+) → ([\d.]+)"
        r" for blocs of ([\d, ]+?) of (\d+)\)",
        flat,
    )
    assert m, "the boundary sentence is not in the README in the shape this test reads"

    claimed = [float(g) for g in m.groups()[:4]]
    blocs = [int(x) for x in m.group(5).split(",")]
    n = int(m.group(6))
    assert len(blocs) == 4, f"the README names {blocs}, four values are quoted"

    for n_bloc, want in zip(blocs, claimed):
        got = sum(
            effective_votes(_bloc_panel(n_bloc, n - n_bloc, seed=seed), n) for seed in range(5)
        ) / 5
        assert round(got, 2) == want, (
            f"README says n_eff {want} for a {n_bloc}-of-{n} bloc; a run gives {got:.4f} "
            f"(rounds to {round(got, 2)})"
        )


def test_the_boundary_figure_guard_can_fire():
    """Control arm: the check above must reject a wrong digit, not merely pass on a right one."""
    n, n_bloc = 10, 3
    got = sum(
        effective_votes(_bloc_panel(n_bloc, n - n_bloc, seed=seed), n) for seed in range(5)
    ) / 5
    assert round(got, 2) != 1.88, (
        "1.88 was the README's figure and is not what a run produces — if this assertion "
        "fails, the mechanism changed and the guard above would silently accept the old value"
    )


# ── the warning the package exists to print ──────────────────────────────────


def _synthetic_panel_file(tmp_path, n_bloc, n_total=10, items=40):
    """A demo-shaped data file whose largest provenance group is `n_bloc` of `n_total`."""
    import json
    import random

    rng = random.Random(4242)
    panel = [
        {"name": f"v{i}", "model": "bloc-model" if i < n_bloc else f"other-{i}"}
        for i in range(n_total)
    ]
    votes, truth = [], []
    for _ in range(items):
        bloc_answer = 0 if rng.random() < 0.55 else rng.randint(1, 9)
        row = [bloc_answer] * n_bloc
        row += [0 if rng.random() < 0.7 else rng.randint(10, 99) for _ in range(n_total - n_bloc)]
        votes.append(row)
        truth.append(0)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"votesByItem": votes, "correctGroupByItem": truth, "panel": panel}))
    return path


def _demo_output(data_path):
    import io
    from contextlib import redirect_stdout

    from voteworth import demo

    original = demo.DATA
    demo.DATA = data_path
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            demo.main()
        return buf.getvalue()
    finally:
        demo.DATA = original


_WARNING = "ONE GROUP HOLDS THE MAJORITY"


def test_demo_warns_when_one_group_holds_the_majority(tmp_path):
    """The blind spot is what this package is for, and the warning had no test until now.

    The bundled panel's largest group is 30%, so the branch is unreachable from the shipped
    demo — it was written, documented in the README, and never executed by the suite. An
    acceptance pass on 2026-09-21 found it by grepping the tests for the message.
    """
    out = _demo_output(_synthetic_panel_file(tmp_path, n_bloc=6))
    assert _WARNING in out, (
        "a six-of-ten provenance bloc did not trigger the majority warning:\n" + out
    )
    assert "cannot see it" in out, "the warning fired without its explanation"


def test_demo_stays_quiet_below_the_threshold(tmp_path):
    """The discriminating half. A warning that always prints carries no information.

    Five of ten is not a majority: `share > 0.5` is strict, and the boundary is where an
    off-by-one would live.
    """
    out = _demo_output(_synthetic_panel_file(tmp_path, n_bloc=5))
    assert _WARNING not in out, (
        "the warning fired at exactly 50%, where the documented threshold is strictly above"
    )


def test_bundled_demo_does_not_warn():
    """And on the shipped panel it must stay silent — that is why the branch went untested."""
    from voteworth import demo

    assert _WARNING not in _demo_output(demo.DATA)
