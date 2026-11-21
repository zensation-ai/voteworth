"""Do the corrected sentences stay corrected?

The figure checks in this suite ask whether a number is producible. A sentence that states no
number passes them whatever it says, and most of the statements found wrong before the 0.1.3
upload and in the review of 0.1.4 (2026-11-21) were of that kind: the learned weight called
inert on any one-model panel, a clustering rule described by its centroid, a signature read as
uniform weights, a threshold stated as above half, n_eff read as a count of independent voters.
Each was corrected in the text and bound to a test of the behaviour it describes.

This file holds the text to the correction. Each contract names a place, the sentence that place
must carry, the 0.1.3 sentence it replaced, and the tests that bind the behaviour. A place passes
when the corrected sentence is there and the 0.1.3 sentence is not. The control arm plants the
0.1.3 sentence in a copy of the text, contract by contract, and requires the check to fail.

Sentences are compared as text, with whitespace and bold markers collapsed. A figure in a
corrected sentence is checked where its value comes from (test_readme_claims.py,
test_docstring_contracts.py), so the contracts quote the parts of the sentences without figures.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import NamedTuple

import pytest

import voteworth

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"

# Not `voteworth.aggregate`: the package's function `aggregate` shadows the submodule there
# (see test_docstring_claims.py).
_AGGREGATE_MODULE = importlib.import_module("voteworth.aggregate")


class Contract(NamedTuple):
    where: str  # "README", "voteworth.aggregate" (the module docstring), or "name()" for a function
    required: str  # the corrected sentence, or the part of it that carries the correction
    old: str  # the 0.1.3 sentence it replaced, as 0.1.3 shipped it
    binds: tuple  # the tests that assert the behaviour the sentence describes


TEXT_CONTRACTS = [
    # ── the module docstring: the structural limit ──
    Contract(
        "voteworth.aggregate",
        "If every voter in a panel is the same model, the provenance weight (`structural_weights`) "
        "does nothing.",
        "If every voter in a panel is the same model, this repair does nothing.",
        ("test_same_model_panel_is_inert",),
    ),
    Contract(
        "voteworth.aggregate",
        "The learned weight is not bound by this: from four voters of one model up, it can "
        "down-weight voters that co-deviate.",
        "it is what the method can and cannot do",
        ("test_learned_weights_can_differ_from_four_voters_of_one_model",
         "test_learned_weights_are_uniform_for_three_voters_of_one_model"),
    ),
    Contract(
        "voteworth.aggregate",
        "A panel that is one model throughout cannot be repaired by provenance weighting; it has "
        "to be made diverse.",
        "A panel that cannot disagree independently cannot be repaired by reweighting; it has to "
        "be made diverse.",
        ("test_same_model_panel_is_inert",),
    ),
    # ── cluster_answers: how an answer finds its cluster ──
    Contract(
        "cluster_answers()",
        "each answer is compared with the first member of every cluster so far and joins the most "
        "similar one if the cosine is ≥ `thresh`",
        "Greedy single-link: each answer joins the first cluster whose centroid it matches at "
        "cosine ≥ `thresh`, else starts a new one.",
        ("test_cluster_answers_joins_the_most_similar_cluster_not_the_first",
         "test_cluster_answers_compares_with_the_first_member"),
    ),
    # ── effective_votes: what n_eff measures, and the signature of the blind spot ──
    Contract(
        "effective_votes()",
        "Agreement rises with accuracy as well as with dependence.",
        "How many independent voters your panel is actually worth (Kish's effective sample size).",
        ("test_raw_n_eff_rises_with_accuracy_alone",),
    ),
    Contract(
        "effective_votes()",
        "between panels of different accuracy the order of n_eff does not carry.",
        "The ordering carries; the second decimal does not.",
        ("test_n_eff_order_does_not_follow_dependence",
         "test_mixed_panel_scores_above_an_independent_accurate_one"),
    ),
    Contract(
        "effective_votes()",
        "A low n_eff beside a largest group at full weight is the signature of that blind spot",
        "A low n_eff beside uniform weights is the signature of that blind spot",
        ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",),
    ),
    Contract(
        "effective_votes()",
        "provided n_eff is also low against independent voters at the same accuracy: accurate "
        "independent voters score low too, with every weight at full.",
        "is the signature of that blind spot — see `bloc_share`.",
        ("test_raw_n_eff_rises_with_accuracy_alone",),
    ),
    Contract(
        "effective_votes()",
        "a coarse one for categorical labels, and it carries the voters' accuracy with it.",
        "and a coarse one for categorical labels.",
        ("test_raw_n_eff_rises_with_accuracy_alone",),
    ),
    Contract(
        "effective_votes()",
        '"error" rises with accuracy too',
        "(right/wrong) instead of votes. Any other value of `mode` raises",
        ("test_error_mode_rises_with_accuracy_too",),
    ),
    # ── bloc_share: from half the panel up ──
    Contract(
        "bloc_share()",
        "From 0.5 up one group can carry the plurality on its own; at exactly half it wins every "
        "tie its answer wins on cluster id, as the first answer in input order does.",
        "Above 0.5 one group can carry the plurality on its own",
        ("test_a_half_bloc_in_front_keeps_full_weight", "test_demo_warns_at_exactly_half"),
    ),
    # ── README, the first finding: what the 1.48 is worth ──
    Contract(
        "README",
        "Raw agreement rises with accuracy as well as with dependence",
        "counts how many independent voters you actually have.",
        ("test_raw_n_eff_rises_with_accuracy_alone",
         "test_the_printed_baseline_is_what_independent_voters_score"),
    ),
    Contract(
        "README",
        "Without labels n_eff cannot tell a redundant panel from an accurate one, and the order of "
        "panels with different accuracy does not carry.",
        "The ordering carries; the second decimal does not",
        ("test_n_eff_order_does_not_follow_dependence",
         "test_mixed_panel_scores_above_an_independent_accurate_one"),
    ),
    Contract(
        "README",
        "so adding them buys less than the count suggests.",
        "Adding voters from the same pool buys far less than the count suggests.",
        ("test_same_model_pairs_share_their_wrong_answers_more_often",),
    ),
    # ── README, the boundary: the signature and the warning ──
    Contract(
        "README",
        "while the bloc's weight returns to full once it never deviates from the plurality.",
        "while the weights go flat.",
        ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",
         "test_effective_votes_falls_as_the_bloc_grows"),
    ),
    Contract(
        "README",
        "A low n_eff beside a largest group at full weight is the signature, provided n_eff is "
        "also low against independent voters at the same accuracy: accurate independent voters "
        "score low too, with every weight at full.",
        "A low n_eff beside uniform weights is the signature.",
        ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",
         "test_raw_n_eff_rises_with_accuracy_alone"),
    ),
    Contract(
        "README",
        "The demo prints all three and warns from 50% up.",
        "The demo prints both and warns above 50%.",
        ("test_demo_warns_at_exactly_half", "test_demo_stays_quiet_below_the_threshold"),
    ),
]

# The places the review named: the module docstring, cluster_answers, the signature in
# effective_votes and in the README, bloc_share, and the README's first finding.
_PLACES = {"voteworth.aggregate", "cluster_answers()", "effective_votes()", "bloc_share()", "README"}


def _flat(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def _text_of(where: str) -> str:
    """The text a user reads at that place: the README file, or a docstring as help() shows it."""
    if where == "README":
        return (_ROOT / "README.md").read_text()
    if where == "voteworth.aggregate":
        return inspect.getdoc(_AGGREGATE_MODULE) or ""
    return inspect.getdoc(getattr(voteworth, where.removesuffix("()"))) or ""


def _contract_problems(text: str, contract: Contract) -> list[str]:
    """What is wrong with `text` under one contract: the correction missing, the 0.1.3 text back."""
    flat = _flat(text)
    problems = []
    if _flat(contract.required) not in flat:
        problems.append(f"{contract.where}: the corrected sentence is missing: {contract.required!r}")
    if _flat(contract.old) in flat:
        problems.append(f"{contract.where}: the 0.1.3 sentence is back: {contract.old!r}")
    return problems


def _label(contract: Contract) -> str:
    return f"{contract.where} {contract.required[:48]}"


@pytest.mark.parametrize("contract", TEXT_CONTRACTS, ids=_label)
def test_the_corrected_sentence_stands(contract):
    problems = _contract_problems(_text_of(contract.where), contract)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("contract", TEXT_CONTRACTS, ids=_label)
def test_the_contract_fires_on_the_0_1_3_sentence(contract):
    """Control arm: with the 0.1.3 sentence planted in place of the corrected one, the check must
    report both, the correction missing and the old sentence back."""
    text = _flat(_text_of(contract.where))
    required = _flat(contract.required)
    assert required in text, "nothing to plant into: the corrected sentence is not there"
    planted = text.replace(required, _flat(contract.old))
    problems = _contract_problems(planted, contract)
    assert len(problems) == 2, problems


def test_every_contract_names_tests_that_exist():
    """A contract bound to a renamed or deleted test reads as coverage and is none."""
    names = set()
    for path in _TESTS.glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    assert names, "no tests discovered — this check is broken"
    missing = {_label(c): [t for t in c.binds if t not in names] for c in TEXT_CONTRACTS}
    missing = {label: tests for label, tests in missing.items() if tests}
    assert not missing, f"contracts bound to tests that do not exist: {missing}"
    assert all(c.binds for c in TEXT_CONTRACTS), "a contract binds no test"


def test_the_contracts_cover_every_place_the_review_named():
    """A list that lost a place would pass on the others."""
    assert {c.where for c in TEXT_CONTRACTS} == _PLACES
