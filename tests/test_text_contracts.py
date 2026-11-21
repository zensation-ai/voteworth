"""Do the corrected sentences stay corrected?

The figure checks in this suite ask whether a number is producible. A sentence that states no
number passes them whatever it says, and most of the statements found wrong before the 0.1.3
upload and in the reviews of 0.1.4 (2026-11-21) were of that kind: the learned weight called
inert on any one-model panel, a clustering rule described by its centroid, a signature read as
uniform weights, a threshold stated as above half, n_eff read as a count of independent voters,
and its gap to a baseline read as dependence. Each was corrected in the text and bound to a test
of the behaviour it describes.

This file holds the text to the correction. Each contract names a place, the sentence that place
must carry, the sentence it replaced, the release or draft that carried the replaced sentence,
and the tests that bind the behaviour. A place passes when the corrected sentence is there and
the replaced one is not. A contract without a corrected sentence only forbids the replaced one.

Two control arms. The first plants the replaced sentence in a copy of today's text and requires
the check to fail; it shows the check can fire, not that the replaced sentence was ever there.
The second reads the released text out of the history (`git show v0.1.3:…`) and requires every
contract whose sentence 0.1.3 carried to fail on it for both reasons, so a replaced sentence
that never stood there is caught. It reads the tag, not a commit of the fix branch: a squash
merge leaves the branch's commits unreachable, and an arm pointed at them would skip without a
word (review of 2026-11-21). Sentences from the unreleased drafts of 0.1.4 are guarded by the
first arm only. The second arm needs the history: it runs in a clone and in CI, and skips in an
sdist or a shallow checkout.

Sentences are compared as text, with whitespace and bold markers collapsed. A figure in a
corrected sentence is checked where its value comes from (test_readme_claims.py,
test_docstring_contracts.py), so the contracts quote the parts of the sentences without figures.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

import voteworth

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"

# Not `voteworth.aggregate`: the package's function `aggregate` shadows the submodule there
# (see test_docstring_claims.py).
_AGGREGATE_MODULE = importlib.import_module("voteworth.aggregate")

# Where a replaced sentence stood: the release 0.1.3, read from its tag, or one of the reviewed
# drafts of 0.1.4 that were never released (first arm only).
_RELEASE_TAG = "v0.1.3"
_REPLACED = {"0.1.3", "draft a14522e", "draft b3a3703"}


class Contract(NamedTuple):
    where: str  # a file, "voteworth" or "voteworth.aggregate" (module docstrings), or "name()"
    required: Optional[str]  # the corrected sentence, or the part of it that carries the correction
    old: str  # the sentence it replaced, as it stood
    replaced: str  # where it stood: "0.1.3", or the draft of 0.1.4 that carried it
    binds: tuple  # the tests that assert the behaviour the sentence describes


TEXT_CONTRACTS = [
    # ── README: the headline and the first finding ──
    Contract("README.md", "effective votes by raw agreement", "Worth 1.48 independent votes.", "0.1.3",
             ("test_n_eff_figures_in_the_prose_are_the_printed_ones",)),
    Contract("README.md", "effective votes by raw agreement",
             "independent voters as accurate would be worth", "draft a14522e",
             ("test_n_eff_figures_in_the_prose_are_the_printed_ones",)),
    Contract("README.md",
             "raw agreement rises with differences in difficulty between items, with shared errors, "
             "and with accuracy once voters are right more often than not",
             "counts how many independent voters you actually have.", "0.1.3",
             ("test_raw_n_eff_rises_with_accuracy_alone",
              "test_unequal_items_lower_n_eff_with_every_weight_at_full")),
    Contract("README.md", "and never give the same wrong answer also score",
             "are each right 80% of the time also score 1.48", "draft a14522e",
             ("test_raw_n_eff_rises_with_accuracy_alone",
              "test_two_answer_options_put_every_weight_at_one_tenth")),
    Contract("README.md", "n_eff alone cannot tell differences in item difficulty from shared errors",
             "n_eff therefore says something about dependence only next to what independent voters",
             "draft a14522e",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",
              "test_unequal_items_and_voters_right_together_both_raise_the_all_right_count")),
    Contract("README.md", "A lower n_eff does not mark the more dependent panel.",
             "The ordering carries; the second decimal does not", "0.1.3",
             ("test_n_eff_order_does_not_follow_dependence",
              "test_mixed_panel_scores_above_an_independent_accurate_one")),
    Contract("README.md", "pick the same answer more often than voters from different models",
             "Voters from the same model share their wrong answers", "draft a14522e",
             ("test_same_model_pairs_agree_beyond_what_shuffling_within_items_gives",)),
    Contract("README.md", "adding voters from the same model buys less than the count suggests.",
             "Adding voters from the same pool buys far less than the count suggests.", "0.1.3",
             ("test_same_model_pairs_agree_beyond_what_shuffling_within_items_gives",)),
    Contract("README.md", "Agreement was not measuring correctness.",
             "Agreement was measuring redundancy, not correctness.", "0.1.3",
             ("test_readme_prose_numbers_are_produced",)),
    Contract("README.md", "none of which isolates that dependence on its own",
             "This package measures that dependence three ways", "0.1.3",
             ("test_raw_n_eff_rises_with_accuracy_alone",
              "test_two_answer_options_put_every_weight_at_one_tenth")),
    # ── README: the usage example and the weight ──
    Contract("README.md", "# Kish's n_eff by raw agreement, not a count of independent voters",
             "# how many independent voters you really have", "0.1.3",
             ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("README.md", "# how often two voters give the same answer when both deviate",
             "# which pairs share their errors", "0.1.3", ("test_pairwise_coerror_signal",)),
    Contract("README.md", "Σ_{j≠i} sameAnswer[i][j]", "Σ_{j≠i} sharedError[i][j]", "0.1.3",
             ("test_independence_weights_downweight_correlated_bloc",)),
    # ── README: the boundary, the signature, the warning ──
    Contract("README.md", "The co-deviation rate is measured among", "The shared-error rate is measured among",
             "0.1.3", ("test_dominant_bloc_is_invisible_to_the_weights",)),
    Contract("README.md", "while the bloc's weight returns to full once it never deviates from the plurality.",
             "while the weights go flat.", "0.1.3",
             ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",
              "test_effective_votes_falls_as_the_bloc_grows")),
    Contract("README.md",
             "The signature is a group of voters that vote together, kept at full weight while the "
             "others are down-weighted.",
             "A low n_eff beside uniform weights is the signature.", "0.1.3",
             ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",
              "test_unequal_items_lower_n_eff_with_every_weight_at_full",
              "test_two_answer_options_put_every_weight_at_one_tenth")),
    Contract("README.md",
             "The signature is a group of voters that vote together, kept at full weight while the "
             "others are down-weighted.",
             "provided n_eff is also low against independent voters at the same accuracy", "draft a14522e",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",)),
    Contract("README.md", "The demo prints n_eff and the largest provenance share and warns from 50% up",
             "The demo prints both and warns above 50%.", "0.1.3",
             ("test_demo_warns_at_exactly_half", "test_demo_stays_quiet_below_the_threshold")),
    Contract("README.md", "The demo prints n_eff and the largest provenance share and warns from 50% up",
             "The demo prints all three and warns from 50% up.", "draft a14522e",
             ("test_demo_warns_at_exactly_half",)),
    # ── README: what the package does not do ──
    Contract("README.md", "Diversity is not shown to be enough.", "Diversity is necessary, not sufficient.",
             "0.1.3", ("test_same_model_pairs_agree_beyond_what_shuffling_within_items_gives",)),
    Contract("README.md", None, "panel of four families is worth", "0.1.3",
             ("test_readme_output_block_is_a_verbatim_run",)),
    # ── the demo, as its source states it (the warning prints only from half the panel up) ──
    Contract("src/voteworth/demo.py", "both break from the plurality, do they pick the same answer?",
             "both break from the majority, do they pick the same answer?", "0.1.3",
             ("test_readme_output_block_is_a_verbatim_run",)),
    Contract("src/voteworth/demo.py", "n_eff alone cannot tell them apart.",
             "for independent voters as accurate.", "draft a14522e",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",)),
    Contract("src/voteworth/demo.py", None, "panel of four families is worth", "0.1.3",
             ("test_readme_output_block_is_a_verbatim_run",)),
    Contract("src/voteworth/demo.py", "The co-deviation statistic below cannot see it",
             "The shared-error statistic below cannot see it", "0.1.3",
             ("test_demo_warns_when_one_group_holds_the_majority",)),
    Contract("src/voteworth/demo.py", "57 answerable items, 4 models, one judge.",
             "57 answerable items, 4 model families, one judge.", "0.1.3",
             ("test_the_judge_sentence_matches_the_data",)),
    # ── the package and module docstrings ──
    Contract("voteworth", "Shows how often two voters that both break from the plurality pick the same answer",
             "Tells you whether your panel can produce an independent majority", "0.1.3",
             ("test_pairwise_coerror_signal",)),
    Contract("voteworth.aggregate",
             "If every voter in a panel is the same model, the provenance weight (`structural_weights`) "
             "does nothing.",
             "If every voter in a panel is the same model, this repair does nothing.", "0.1.3",
             ("test_same_model_panel_is_inert",)),
    Contract("voteworth.aggregate",
             "The learned weight is not bound by this: from four voters of one model up, it can "
             "down-weight voters that co-deviate.",
             "it is what the method can and cannot do", "0.1.3",
             ("test_learned_weights_can_differ_from_four_voters_of_one_model",
              "test_learned_weights_are_uniform_for_three_voters_of_one_model")),
    Contract("voteworth.aggregate",
             "A panel that is one model throughout cannot be repaired by provenance weighting; it has "
             "to be made diverse.",
             "A panel that cannot disagree independently cannot be repaired by reweighting; it has to "
             "be made diverse.", "0.1.3", ("test_same_model_panel_is_inert",)),
    Contract("voteworth.aggregate", "`python -m voteworth.demo` prints it as `INERT`",
             "`examples/quickstart.py` prints it as `INERT`", "0.1.3",
             ("test_advertised_module_command_runs",)),
    Contract("voteworth.aggregate", "A voter that deviates with the same value as others is treated as redundant",
             "A voter that shares its errors with others is redundant", "0.1.3",
             ("test_independence_weights_downweight_correlated_bloc",)),
    # ── function docstrings ──
    Contract("cluster_answers()",
             "each answer is compared with the first member of every cluster so far and joins the most "
             "similar one if the cosine is ≥ `thresh`",
             "Greedy single-link: each answer joins the first cluster whose centroid it matches at "
             "cosine ≥ `thresh`, else starts a new one.", "0.1.3",
             ("test_cluster_answers_joins_the_most_similar_cluster_not_the_first",
              "test_cluster_answers_compares_with_the_first_member")),
    Contract("learn_independence_weights()", "(always deviating together, with the same answer)",
             "(always the same wrong answer together)", "0.1.3",
             ("test_independence_weights_downweight_correlated_bloc",)),
    Contract("pairwise_coerror()", "Diagnostic: the co-deviation rate for every voter pair.",
             "Diagnostic: the shared-error rate for every voter pair.", "0.1.3",
             ("test_pairwise_coerror_signal",)),
    Contract("effective_votes()", "Kish's effective sample size of a panel, from its mean pairwise agreement.",
             "How many independent voters your panel is actually worth (Kish's effective sample size).",
             "0.1.3", ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("effective_votes()",
             "Agreement rises with differences in difficulty between items, with shared errors, and with "
             "accuracy once voters are right more often than not.",
             "Agreement rises with accuracy as well as with dependence.", "draft a14522e",
             ("test_raw_n_eff_rises_with_accuracy_alone",
              "test_unequal_items_lower_n_eff_with_every_weight_at_full")),
    Contract("effective_votes()", "and never give the same wrong answer agree on about",
             "each right 80% of the time agree on about 64% of pairs", "draft a14522e",
             ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("effective_votes()", "and n_eff alone cannot settle it",
             "Read n_eff next to what independent voters with the same accuracies would score", "draft a14522e",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",)),
    Contract("effective_votes()", "A lower n_eff does not mark the more dependent panel.",
             "The ordering carries; the second decimal does not.", "0.1.3",
             ("test_n_eff_order_does_not_follow_dependence",)),
    Contract("effective_votes()",
             "The signature of that blind spot is a group of voters that vote together, kept at full "
             "weight while the others are down-weighted",
             "A low n_eff beside uniform weights is the signature of that blind spot", "0.1.3",
             ("test_a_bloc_that_holds_the_plurality_keeps_full_weight_and_the_others_do_not",)),
    Contract("effective_votes()",
             "The signature of that blind spot is a group of voters that vote together, kept at full "
             "weight while the others are down-weighted",
             "provided n_eff is also low against independent voters at the same", "draft a14522e",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",)),
    Contract("effective_votes()", "it carries the voters' accuracy and the spread of item difficulty with it.",
             "and a coarse one for categorical labels.", "0.1.3",
             ("test_unequal_items_lower_n_eff_with_every_weight_at_full",)),
    Contract("effective_votes()", '"error" rises with accuracy too once voters are right more often than not',
             "(right/wrong) instead of votes. Any other value of `mode` raises", "0.1.3",
             ("test_error_mode_rises_with_accuracy_too",)),
    Contract("effective_votes()", '"error" rises with accuracy too once voters are right more often than not',
             '"error" rises with accuracy too: two independent voters', "draft a14522e",
             ("test_error_mode_rises_with_accuracy_too",)),
    Contract("bloc_share()",
             "From 0.5 up one group can carry the plurality on its own; at exactly half it wins every "
             "tie its answer wins on cluster id, as the first answer in input order does.",
             "Above 0.5 one group can carry the plurality on its own", "0.1.3",
             ("test_a_half_bloc_in_front_keeps_full_weight", "test_demo_warns_at_exactly_half")),
    Contract("bloc_share()", "The co-deviation statistic this package relies on stops seeing it",
             "the shared-error statistic this package relies on stops seeing it", "0.1.3",
             ("test_dominant_bloc_is_invisible_to_the_weights",)),
    Contract("agreement_confidence()", "see `python -m voteworth.demo`", "see `examples/quickstart.py`", "0.1.3",
             ("test_advertised_module_command_runs",)),
    # ── the surfaces PyPI and the citation box show ──
    Contract("pyproject.toml", "report Kish's effective number of votes by raw agreement",
             "report the effective number of independent votes", "0.1.3",
             ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("pyproject.toml", "pick the same answer when both break from the plurality",
             "Measure error correlation in an LLM judge panel", "0.1.3", ("test_pairwise_coerror_signal",)),
    Contract("CITATION.cff", "voteworth: independence-weighted aggregation for LLM voter and judge panels",
             "how many independent votes your LLM judge panel is actually worth", "0.1.3",
             ("test_the_citation_title_is_the_module_summary",)),
    Contract("CITATION.cff", "pick the same answer when both break from the plurality",
             "Measures whether an LLM voter or judge panel can produce an independent majority", "0.1.3",
             ("test_pairwise_coerror_signal",)),
    Contract("CITATION.cff", "a bloc that votes in lockstep and holds the plurality never deviates from it",
             "It states its own boundary:", "0.1.3", ("test_dominant_bloc_is_invisible_to_the_weights",)),
    # ── after the fourth review: the post-hoc best voter, conditions, terms of earlier drafts ──
    Contract("README.md", None, "points below the panel's best single member", "0.1.3",
             ("test_the_demo_names_its_best_voter_as_picked_after_the_fact",)),
    Contract("README.md", None, "voting was the wrong move, and the tool says so", "0.1.3",
             ("test_the_demo_names_its_best_voter_as_picked_after_the_fact",)),
    Contract("README.md", "Read the −7.2 pp above as what happened on these 29 items, not as a direction",
             "Read the −7.2 pp above as a direction, not a size.", "0.1.3",
             ("test_readme_prose_numbers_are_produced",)),
    Contract("README.md", "blinds the co-deviation detector", "blinds the shared-error detector", "0.1.3",
             ("test_suite_detects_mutation",)),
    Contract("README.md", "The bundled matrix holds 80 items, 57 of them answerable",
             "The bundled matrix is 57 answerable items", "0.1.3", ("test_the_judge_sentence_matches_the_data",)),
    Contract("README.md", "raw agreement over all 57 items", "| raw agreement over all items |", "0.1.3",
             ("test_gap_span_and_ratio_factor_match_a_live_run",)),
    Contract("README.md", "On equally hard items, ten independent voters",
             "right more often than not. Ten independent voters that are each right 80%",
             "draft b3a3703", ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("src/voteworth/demo.py", None, "Does the panel beat its own best member?", "0.1.3",
             ("test_the_demo_names_its_best_voter_as_picked_after_the_fact",)),
    Contract("src/voteworth/demo.py", None, "→ The plain majority lands", "0.1.3",
             ("test_the_demo_names_its_best_voter_as_picked_after_the_fact",)),
    Contract("src/voteworth/demo.py", '"raw agreement, all 57 items"', '"raw agreement, all items"',
             "0.1.3", ("test_gap_span_and_ratio_factor_match_a_live_run",)),
    Contract("examples/quickstart.py", "on 29 items that is what happened here, not a rule",
             "scores WORSE than the panel's best", "0.1.3",
             ("test_the_demo_names_its_best_voter_as_picked_after_the_fact",)),
    Contract("voteworth.aggregate", "They often are not.",
             "That silently assumes the voters are independent. They are not.", "0.1.3",
             ("test_same_model_pairs_agree_beyond_what_shuffling_within_items_gives",)),
    Contract("voteworth.aggregate", "wins the vote while being wrong", "wins the majority while being wrong",
             "0.1.3", ("test_weighted_beats_naive_on_detectable_bloc",)),
    Contract("voteworth.aggregate",
             "measures how often two voters that both break from the plurality pick the same answer",
             "This module measures that correlation and corrects for it.", "0.1.3",
             ("test_pairwise_coerror_signal",)),
    Contract("effective_votes()", "voters that all say the same thing scores one",
             "voters that all say the same thing is worth one", "0.1.3", ("test_effective_votes_bounds",)),
    Contract("effective_votes()", "On equally hard items, ten independent", "Ten independent voters that are each",
             "draft b3a3703", ("test_raw_n_eff_rises_with_accuracy_alone",)),
    Contract("CITATION.cff", "corrects the vote for voters that do", "corrects the vote for correlated voters",
             "0.1.3", ("test_pairwise_coerror_signal",)),
]

# The places the reviews named: the README, the demo and the quickstart, the module and package
# docstrings, six function docstrings, and the two surfaces PyPI and GitHub's citation box show.
_PLACES = {
    "README.md", "src/voteworth/demo.py", "voteworth", "voteworth.aggregate", "cluster_answers()",
    "learn_independence_weights()", "pairwise_coerror()", "effective_votes()", "bloc_share()",
    "agreement_confidence()", "pyproject.toml", "CITATION.cff", "examples/quickstart.py",
}


def _flat(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def _text_of(where: str) -> str:
    """The text a user reads at that place: a file, or a docstring as help() shows it."""
    if where == "voteworth":
        return inspect.getdoc(voteworth) or ""
    if where == "voteworth.aggregate":
        return inspect.getdoc(_AGGREGATE_MODULE) or ""
    if where.endswith("()"):
        return inspect.getdoc(getattr(voteworth, where[:-2])) or ""
    return (_ROOT / where).read_text()


def _contract_problems(text: str, contract: Contract) -> list[str]:
    """What is wrong with `text` under one contract: the correction missing, the replaced text back."""
    flat = _flat(text)
    problems = []
    if contract.required is not None and _flat(contract.required) not in flat:
        problems.append(f"{contract.where}: the corrected sentence is missing: {contract.required!r}")
    if _flat(contract.old) in flat:
        problems.append(f"{contract.where}: the replaced sentence is back: {contract.old!r}")
    return problems


def _label(contract: Contract) -> str:
    return f"{contract.where} {contract.replaced} {(contract.required or 'forbids ' + contract.old)[:44]}"


@pytest.mark.parametrize("contract", TEXT_CONTRACTS, ids=_label)
def test_the_corrected_sentence_stands(contract):
    problems = _contract_problems(_text_of(contract.where), contract)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("contract", TEXT_CONTRACTS, ids=_label)
def test_the_contract_fires_on_the_planted_sentence(contract):
    """Control arm one: the replaced sentence planted in a copy of today's text must be reported,
    as the corrected sentence missing and the replaced one back, or, for a contract that only
    forbids, as the replaced one back."""
    text = _flat(_text_of(contract.where))
    if contract.required is None:
        planted, expected = text + " " + _flat(contract.old), 1
    else:
        required = _flat(contract.required)
        assert required in text, "nothing to plant into: the corrected sentence is not there"
        planted, expected = text.replace(required, _flat(contract.old)), 2
    problems = _contract_problems(planted, contract)
    assert len(problems) == expected, problems


def _text_at(revision: str, where: str) -> Optional[str]:
    """The text at that place in a revision of the history, or None if this checkout lacks it."""
    if shutil.which("git") is None or not (_ROOT / ".git").exists():
        return None
    files = {"voteworth": "src/voteworth/__init__.py", "voteworth.aggregate": "src/voteworth/aggregate.py"}
    path = files.get(where, "src/voteworth/aggregate.py" if where.endswith("()") else where)
    shown = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=_ROOT, capture_output=True, text=True)
    if shown.returncode != 0:
        return None
    if where in files or where.endswith("()"):
        tree = ast.parse(shown.stdout)
        if where in files:
            return ast.get_docstring(tree, clean=True) or ""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == where[:-2]:
                return ast.get_docstring(node, clean=True) or ""
        return ""
    return shown.stdout


@pytest.mark.parametrize("contract", [c for c in TEXT_CONTRACTS if c.replaced == "0.1.3"], ids=_label)
def test_the_contract_fires_on_the_released_text(contract):
    """Control arm two: the text as released in 0.1.3 must fail the contract for both reasons, or,
    for a contract that only forbids, carry the forbidden sentence."""
    text = _text_at(_RELEASE_TAG, contract.where)
    if text is None:
        pytest.skip(f"{_RELEASE_TAG} is not in this checkout (no .git, no tags, or a shallow clone)")
    problems = _contract_problems(text, contract)
    assert len(problems) == (1 if contract.required is None else 2), problems


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
    assert all(c.replaced in _REPLACED for c in TEXT_CONTRACTS), "a contract names an unknown source"


def test_the_contracts_cover_every_place_the_reviews_named():
    """A list that lost a place would pass on the others."""
    assert {c.where for c in TEXT_CONTRACTS} == _PLACES


def test_the_citation_title_is_the_module_summary():
    """The citation title is the first line of the module docstring, under the package name, so
    the two cannot say different things about what the package is."""
    title = re.search(r'^title: "voteworth: (.*)"$', (_ROOT / "CITATION.cff").read_text(), re.M)
    summary = (inspect.getdoc(_AGGREGATE_MODULE) or "").splitlines()[0].rstrip(".")
    assert title, "CITATION.cff has no title of the form 'voteworth: …'"
    assert title.group(1) == summary[0].lower() + summary[1:], (title.group(1), summary)
