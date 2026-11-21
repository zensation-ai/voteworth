"""Contracts the docstrings state and no test bound until 0.1.4.

A check of the release artefact against every statement it makes (2026-11-21) found three
docstring passages that described the code differently from what it does: the module said the
learned weight is inert on any same-model panel, `cluster_answers` said it joins the first
cluster by its centroid, and `effective_votes` said the weights go uniform in the regime where
one bloc holds the plurality. The docstrings now say what the code does; these tests hold them
to it.

A review of the corrected branch the same day found that n_eff from raw agreement moves with
accuracy as well as with dependence, and that a bloc of exactly half the panel can keep full
weight. Its counterexamples, the baseline the demo prints beside n_eff, and the half-panel bloc
are bound here too.
"""
from __future__ import annotations

import inspect
import io
import json
import math
import random
import re
import statistics
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import voteworth.demo
from test_aggregate import _bloc_panel
from voteworth import (
    aggregate,
    bloc_share,
    cluster_answers,
    effective_votes,
    learn_independence_weights,
    naive_weights,
    pairwise_coerror,
    structural_weights,
)
from voteworth.demo import _independent_baseline

_README = Path(__file__).resolve().parent.parent / "README.md"


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
    # Five of ten does not hold the plurality here only because _bloc_panel puts the truth on
    # cluster id 0: every 5:5 tie goes to the truth, so the bloc deviates in it and is the one
    # down-weighted. With the bloc's answer first in input order it wins those ties and keeps
    # full weight (test_a_half_bloc_in_front_keeps_full_weight).
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


def _ids_in_input_order(row):
    """Cluster ids as cluster_answers assigns them: the first answer in input order gets id 0."""
    ids = {}
    return [-1 if v == -1 else ids.setdefault(v, len(ids)) for v in row]


def test_a_half_bloc_in_front_keeps_full_weight():
    """Five of ten, with the bloc's answer first in input order: it gets cluster id 0 and wins
    every 5:5 tie, so it never deviates, keeps weight 1.0, and the independents are penalised.
    With the truth on id 0 instead, as _bloc_panel builds it, the bloc loses its ties and is the
    one down-weighted. This is why the demo warns from half the panel up
    (test_demo_warns_at_exactly_half)."""
    assert bloc_share(["bloc"] * 5 + [f"other-{i}" for i in range(5)]) == 0.5
    for seed in range(5):
        history = _bloc_panel(5, 5, seed=seed)
        front = learn_independence_weights([_ids_in_input_order(row) for row in history], 10)
        assert front[:5] == [1.0] * 5, (seed, front)
        assert sum(front[5:]) / 5 < 0.5, (seed, front)
        back = learn_independence_weights(history, 10)
        assert sum(back[:5]) / 5 < sum(back[5:]) / 5, (seed, back)


# ── n_eff carries accuracy: three counterexamples, and the baseline it needs ──
#
# Raw agreement rises with accuracy as well as with dependence. The review's three
# counterexamples are tests here, with the figures the README and the docstring of
# effective_votes state read out of them. The baseline the demo prints beside n_eff is checked
# by a second route: a simulation of independent voters, scored by effective_votes itself.


def _flat(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def _independent_panel(n, p, items, seed):
    """n independent voters, each right with probability p. Truth is 0, and a wrong answer is
    the voter's own, so no two wrong answers ever coincide."""
    rng = random.Random(seed)
    return [
        [0 if rng.random() < p else 1000 * (i + 1) + rng.randint(0, 999) for i in range(n)]
        for _ in range(items)
    ]


def _mean_pair_agreement(history, n):
    return statistics.mean(
        sum(row[i] == row[j] for row in history) / len(history) for i in range(n) for j in range(i + 1, n)
    )


# The condition is part of the claim: the 80% panel scores 1.48 with every weight at full only if
# no two voters ever give the same wrong answer (review of 2026-11-21). With two answer options
# every wrong answer is the same one, and every weight drops to 0.10.
_README_COUNTEREXAMPLE = re.compile(
    r"[Tt]en independent voters that are each right (\d+)% of the time and never give the same wrong "
    r"answer also score (\d\.\d\d), with every learned weight at (\d\.\d)"
)
_README_TWO_OPTIONS = re.compile(
    r"With two answer options, where every wrong answer is the same one, they score (\d\.\d\d), and "
    r"every weight drops to (\d\.\d\d)"
)
_DOC_COUNTEREXAMPLE = re.compile(
    r"Ten independent voters that are each right (\d+)% of the time and never give the same wrong "
    r"answer agree on about (\d+)% of pairs and score about (\d\.\d\d)"
)
_DOC_TWO_OPTIONS = re.compile(r"with two answer options they agree on (\d+)% and score about (\d\.\d\d)")
_DOC_ERROR_MODE = re.compile(
    r"two independent voters right (\d+)% of the time are both right or both wrong (\d+)% of the "
    r"time, as are two right (\d+)% of the time"
)


def test_raw_n_eff_rises_with_accuracy_alone():
    """Ten independent voters that never share a wrong answer, right 80% of the time, score as
    low as the bundled panel, with every learned weight at full. The figures come out of the
    README and the docstring; the simulation and the closed form 10/(1+9p^2) must both give them."""
    readme = _README_COUNTEREXAMPLE.search(_flat(_README.read_text()))
    doc = _DOC_COUNTEREXAMPLE.search(_flat(inspect.getdoc(effective_votes)))
    assert readme and doc, "the README or effective_votes no longer states the counterexample in the shape read here"
    assert readme.group(1) == doc.group(1) and readme.group(2) == doc.group(3), (readme.groups(), doc.groups())
    p = int(readme.group(1)) / 100
    history = _independent_panel(10, p, items=2000, seed=20261121)
    n_eff = effective_votes(history, 10)
    assert abs(n_eff - float(readme.group(2))) < 0.03, n_eff
    assert round(10 / (1 + 9 * p * p), 2) == float(readme.group(2))
    assert abs(100 * _mean_pair_agreement(history, 10) - int(doc.group(2))) < 1.0
    weights = learn_independence_weights(history, 10)
    assert weights == [float(readme.group(3))] * 10, weights


def test_error_mode_rises_with_accuracy_too():
    """Mode "error" carries accuracy as well, but only above 50%: two independent voters right p of
    the time are both right or both wrong p^2 + (1-p)^2 of the time, which is the same at p and at
    1 - p. One bundled voter is right 49.1% of the time. The docstring's figures are read out of it."""
    doc = _DOC_ERROR_MODE.search(_flat(inspect.getdoc(effective_votes)))
    assert doc, "effective_votes no longer states the error-mode figure in the shape read here"
    for share in (doc.group(1), doc.group(3)):
        p = int(share) / 100
        assert round(100 * (p * p + (1 - p) ** 2)) == int(doc.group(2)), share
        right_or_wrong = [[v == 0 for v in row] for row in _independent_panel(10, p, items=2000, seed=20261121)]
        assert abs(100 * _mean_pair_agreement(right_or_wrong, 10) - int(doc.group(2))) < 1.0, share


def _two_option_panel(n, p, items, seed):
    """n independent voters, each right with probability p, with one wrong answer for everybody."""
    rng = random.Random(seed)
    return [[0 if rng.random() < p else 1 for _ in range(n)] for _ in range(items)]


def test_two_answer_options_put_every_weight_at_one_tenth():
    """With two answer options every wrong answer is the same one: ten independent voters right 80%
    of the time agree on p^2 + (1-p)^2 of pairs, score about 1.40, and every learned weight is
    1/(1 + 9) because every pair that deviates together deviates with the same answer."""
    readme = _README_TWO_OPTIONS.search(_flat(_README.read_text()))
    doc = _DOC_TWO_OPTIONS.search(_flat(inspect.getdoc(effective_votes)))
    accuracy = _README_COUNTEREXAMPLE.search(_flat(_README.read_text()))
    assert readme and doc and accuracy, "the README or effective_votes no longer states the two-option case"
    p = int(accuracy.group(1)) / 100
    assert readme.group(1) == doc.group(2), (readme.groups(), doc.groups())
    assert round(10 / (1 + 9 * (p * p + (1 - p) ** 2)), 2) == float(readme.group(1))
    history = _two_option_panel(10, p, items=2000, seed=20261121)
    assert abs(effective_votes(history, 10) - float(readme.group(1))) < 0.03
    assert abs(100 * _mean_pair_agreement(history, 10) - int(doc.group(1))) < 1.0
    assert all(abs(w - float(readme.group(2))) < 1e-12 for w in learn_independence_weights(history, 10))


def _unequal_items_panel(items, seed, easy=0.95, hard=0.30, n=10):
    """Independent voters on items of two kinds, half easy and half hard; no two voters ever give the
    same wrong answer."""
    rng = random.Random(seed)
    return [
        [0 if rng.random() < (easy if k % 2 == 0 else hard) else 1000 * (i + 1) + rng.randint(0, 999) for i in range(n)]
        for k in range(items)
    ]


def test_unequal_items_lower_n_eff_with_every_weight_at_full():
    """Half easy items (95% right) and half hard ones (30%), mean accuracy 62.5%: n_eff comes out
    near its closed form, 1.83, well under the 2.21 of equally hard items at the same mean, and every
    learned weight stays at 1.0, because no two voters give the same wrong answer."""
    n_eff = effective_votes(_unequal_items_panel(2000, seed=20261121), 10)
    unequal = 10 / (1 + 9 * (0.95 ** 2 + 0.30 ** 2) / 2)
    equal = 10 / (1 + 9 * 0.625 ** 2)
    assert abs(n_eff - unequal) < 0.03, (n_eff, unequal)
    assert equal - n_eff > 0.3, (equal, n_eff)
    assert learn_independence_weights(_unequal_items_panel(2000, seed=20261121), 10) == [1.0] * 10


def test_unequal_items_and_voters_right_together_both_raise_the_all_right_count():
    """The README's reading of 20 against 0.34: independent voters on equally hard items are one
    model, and two different departures from it both raise the count of items every voter gets
    right. Unequal items do it with independent voters; voters that are right together do it on
    equally hard items. The count cannot say which."""
    def excess(history):
        acc = [sum(row[i] == 0 for row in history) / len(history) for i in range(10)]
        expected = len(history) * math.prod(acc)
        return sum(all(v == 0 for v in row) for row in history) / max(expected, 1e-9)

    rng = random.Random(20261121)
    lockstep = []
    for _ in range(2000):
        right = rng.random() < 0.625
        lockstep.append([0 if right else 1000 * (i + 1) + rng.randint(0, 999) for i in range(5)]
                        + [0 if rng.random() < 0.625 else 1000 * (i + 6) + rng.randint(0, 999) for i in range(5)])
    assert excess(_independent_panel(10, 0.625, items=2000, seed=20261121)) < 1.5
    assert excess(_unequal_items_panel(2000, seed=20261121)) > 3
    assert excess(lockstep) > 3


def test_n_eff_order_does_not_follow_dependence():
    """A panel with a rigid bloc scores above one without: three voters that always share their
    wrong answer, right 55% of the time, beside seven independents, against ten independents right
    70% of the time. The bloc is less accurate, the panel agrees less, and n_eff comes out higher."""
    for seed in range(5):
        with_bloc = effective_votes(_bloc_panel(3, 7, seed=seed), 10)
        without = effective_votes(_bloc_panel(0, 10, seed=seed), 10)
        assert with_bloc > without, (seed, with_bloc, without)


def _coupled_and_independent(items, seed, p=0.5):
    """Five voters that always cast the same vote, beside five independents; all right with
    probability p, and no independent wrong answer coincides with another."""
    rng = random.Random(seed)
    rows = []
    for _ in range(items):
        bloc = 0 if rng.random() < p else 1
        rows.append([bloc] * 5 + [0 if rng.random() < p else 1000 * (i + 1) + rng.randint(0, 999) for i in range(5)])
    return rows


def test_mixed_panel_scores_above_an_independent_accurate_one():
    """Five rigidly coupled and five independent voters, right half the time, score above ten
    independent voters right 80% of the time: the panel with the most dependence comes out
    ahead, because it is the least accurate."""
    mixed = effective_votes(_coupled_and_independent(2000, seed=20261121), 10)
    accurate = effective_votes(_independent_panel(10, 0.8, items=2000, seed=20261121), 10)
    assert mixed > accurate, (mixed, accurate)


def _bundled():
    """The demo's 57 answerable items: votes, gold answers, panel size."""
    data = json.loads(voteworth.demo.DATA.read_text())
    votes, truth = data["votesByItem"], data["correctGroupByItem"]
    keep = [t for t in range(len(votes)) if truth[t] >= 0]
    return [votes[t] for t in keep], [truth[t] for t in keep], len(data["panel"])


def _simulated_baseline(rows, gold, n, distinct, runs=200):
    """Mean n_eff of independent voters with the panel's accuracies, scored by effective_votes.

    Each voter keeps its abstentions and is right with its own accuracy. A wrong answer is drawn
    from the wrong answers cast on that item, so wrong answers coincide as often as they do
    there, or, with `distinct`, is the voter's own.
    """
    acc = []
    for i in range(n):
        cast = [(row[i], g) for row, g in zip(rows, gold) if row[i] != -1]
        acc.append(sum(v == g for v, g in cast) / len(cast))
    scores = []
    for seed in range(runs):
        rng = random.Random(seed)
        simulated = []
        for k, (row, g) in enumerate(zip(rows, gold)):
            wrong = [v for v in row if v != -1 and v != g]
            simulated.append([
                -1 if row[i] == -1
                else g if rng.random() < acc[i]
                else 10_000 * (i + 1) + k if distinct or not wrong
                else rng.choice(wrong)
                for i in range(n)
            ])
        scores.append(effective_votes(simulated, n))
    return statistics.mean(scores)


def _printed_baselines():
    buf = io.StringIO()
    with redirect_stdout(buf):
        voteworth.demo.main()
    per_item = re.search(r"wrong answers coinciding as on each item:\s+(\d\.\d\d)", buf.getvalue())
    distinct = re.search(r"no two wrong answers alike:\s+(\d\.\d\d)", buf.getvalue())
    assert per_item and distinct, "the demo no longer prints the baselines in the shape read here"
    return float(per_item.group(1)), float(distinct.group(1))


def test_the_printed_baseline_is_what_independent_voters_score():
    """The demo's two baselines are computed in closed form. A simulation of independent voters
    with the panel's accuracies, scored by effective_votes, must land on both printed figures."""
    per_item, distinct = _printed_baselines()
    rows, gold, n = _bundled()
    assert abs(_simulated_baseline(rows, gold, n, distinct=False) - per_item) < 0.02
    assert abs(_simulated_baseline(rows, gold, n, distinct=True) - distinct) < 0.02


def test_the_baseline_check_tells_the_two_variants_apart():
    """Control arm: the figure for wrong answers that never coincide must not pass for the one
    where they coincide as on each item, and the other way round."""
    per_item, distinct = _printed_baselines()
    rows, gold, n = _bundled()
    assert abs(_simulated_baseline(rows, gold, n, distinct=False) - distinct) >= 0.02
    assert abs(_simulated_baseline(rows, gold, n, distinct=True) - per_item) >= 0.02


def test_the_baseline_sees_duplicated_voters():
    """Positive control for reading n_eff beside the baseline: copying five voters over the other
    five adds dependence without changing what any voter knows, and n_eff falls further below
    the baseline."""
    rows, gold, n = _bundled()
    copied = [row[:5] + row[:5] for row in rows]
    before = effective_votes(rows, n) / _independent_baseline(rows, gold, n)
    after = effective_votes(copied, n) / _independent_baseline(copied, gold, n)
    assert after < before - 0.05, (before, after)


def test_the_baseline_does_not_depend_on_the_cluster_ids():
    """Control on renaming: cluster ids are names given per item. Renaming them within each item,
    votes and gold answer alike, must leave both baselines and n_eff where they were."""
    rows, gold, n = _bundled()
    reference = (
        _independent_baseline(rows, gold, n),
        _independent_baseline(rows, gold, n, distinct_wrong=True),
        effective_votes(rows, n),
    )
    for seed in range(20):
        rng = random.Random(seed)
        renamed_rows, renamed_gold = [], []
        for row, g in zip(rows, gold):
            names = sorted({v for v in row if v != -1} | {g})
            new = names[:]
            rng.shuffle(new)
            rename = dict(zip(names, new))
            renamed_rows.append([-1 if v == -1 else rename[v] for v in row])
            renamed_gold.append(rename[g])
        renamed = (
            _independent_baseline(renamed_rows, renamed_gold, n),
            _independent_baseline(renamed_rows, renamed_gold, n, distinct_wrong=True),
            effective_votes(renamed_rows, n),
        )
        assert all(abs(a - b) < 1e-12 for a, b in zip(reference, renamed)), (seed, reference, renamed)


def test_same_model_pairs_pick_the_same_answer_more_often_when_both_deviate():
    """On the bundled panel, same-model pairs that both break from the plurality pick the same
    answer more often than cross-model pairs. Computed from the votes, the way the demo's first
    section is. Deviating is not being wrong: the plurality is wrong on 23 of the 57 items."""
    rows, _, n = _bundled()
    models = [voter["model"] for voter in json.loads(voteworth.demo.DATA.read_text())["panel"]]
    rates = pairwise_coerror(rows, n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rates[i][j] == rates[i][j]]
    same = [rates[i][j] for i, j in pairs if models[i] == models[j]]
    cross = [rates[i][j] for i, j in pairs if models[i] != models[j]]
    assert same and cross, "the bundled panel no longer has both kinds of pair"
    assert statistics.mean(same) > statistics.mean(cross), (statistics.mean(same), statistics.mean(cross))


def test_the_demo_counts_the_items_all_voters_got_right():
    """The demo's line on items every voter got right, recomputed from the data: the count, and
    what independent voters with these accuracies would get on equally hard items."""
    rows, gold, n = _bundled()
    count = sum(all(v == g for v in row) for row, g in zip(rows, gold))
    accuracies = [sum(row[i] == g for row, g in zip(rows, gold)) / len(rows) for i in range(n)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        voteworth.demo.main()
    line = re.search(r"items all (\d+) voters got right:\s+(\d+) of (\d+)", buf.getvalue())
    expected = re.search(r"on equally hard items: (\d\.\d\d)", buf.getvalue())
    assert line and expected, "the demo no longer prints the all-right line in the shape read here"
    assert (int(line.group(1)), int(line.group(2)), int(line.group(3))) == (n, count, len(rows))
    assert f"{len(rows) * math.prod(accuracies):.2f}" == expected.group(1)


# ── the same-model contrast against votes shuffled within each item ──
#
# The README says that same-model pairs agree beyond what the items produce, and that this needs
# no labels. Shuffling the votes within each item keeps every item's answers and drops who gave
# them, so under the shuffle no pair kind can be ahead of the other except by chance. The figures
# and the number of shuffles are read out of the README.

_SHUFFLE_SENTENCE = re.compile(
    r"fewer than one in a thousand such shuffles put same-model pairs as far ahead, in that rate "
    r"\(\+(\d\.\d{3})\) or in raw agreement \(\+(\d\.\d{3})\)"
)
# Two thousand shuffles, so that "fewer than one in a thousand" means at most one of them. The
# third review counted none in 2,000 and one in 5,000 with its own seed; this seed gives one.
_SHUFFLES = 2000


def _pair_kinds(n):
    models = [voter["model"] for voter in json.loads(voteworth.demo.DATA.read_text())["panel"]]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [p for p in pairs if models[p[0]] == models[p[1]]], [p for p in pairs if models[p[0]] != models[p[1]]]


def _contrasts(rows, n, same, cross):
    rates = pairwise_coerror(rows, n)
    codev = lambda ps: statistics.mean(rates[i][j] for i, j in ps if rates[i][j] == rates[i][j])
    raw = lambda ps: statistics.mean(sum(r[i] == r[j] for r in rows) / len(rows) for i, j in ps)
    return codev(same) - codev(cross), raw(same) - raw(cross), codev(cross)


def test_same_model_pairs_agree_beyond_what_shuffling_within_items_gives():
    match = _SHUFFLE_SENTENCE.search(_flat(_README.read_text()))
    assert match, "the README no longer states the shuffle in the shape read here"
    rows, _, n = _bundled()
    same, cross = _pair_kinds(n)
    codev_gap, raw_gap, cross_rate = _contrasts(rows, n, same, cross)
    assert (f"{codev_gap:.3f}", f"{raw_gap:.3f}") == (match.group(1), match.group(2))
    rng = random.Random(20261121)
    reached = [0, 0]
    null_codev, null_cross = [], []
    for _ in range(_SHUFFLES):
        shuffled = [rng.sample(row, len(row)) for row in rows]
        c, r, x = _contrasts(shuffled, n, same, cross)
        reached[0] += c >= codev_gap
        reached[1] += r >= raw_gap
        null_codev.append(c)
        null_cross.append(x)
    assert all(count / _SHUFFLES < 0.001 for count in reached), reached
    assert abs(statistics.mean(null_codev)) < 0.01, statistics.mean(null_codev)
    # "Diversity is not shown to be enough": cross-model pairs pick the same answer no more often
    # than the same votes shuffled within each item would.
    assert cross_rate < statistics.mean(null_cross), (cross_rate, statistics.mean(null_cross))
    assert "no more often than the same votes shuffled within each item would" in _flat(_README.read_text())
