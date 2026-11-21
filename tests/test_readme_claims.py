"""Does the README still describe this package?

A README is a claim about software, and claims rot. This file reads the README as the source
of truth — the numbers in it are parsed out of the file, not restated here — and checks them
against a live run. If someone changes the mechanism, these fail. If someone edits the README
to a number the code does not produce, these fail too.

The pattern: the document is the source. Nothing below hard-codes a result that also appears
in the README, because then a wrong README and a wrong test would agree with each other.

What it does *not* do: judge whether the prose is true. It checks the falsifiable parts —
the advertised commands run, and the printed numbers match the table.

Two limits of the number check, stated so that green does not read as total coverage:

* It matches decimals with a leading digit (0.438, 17.2). Figures written ".47" or "2,377"
  or plain integers like "219 points" pass unchecked. Those forms appear in this README only
  inside third-party citations, where the source, not a live run, is the authority — but a
  future claim about our own measurement written in one of those forms would slip through.
* It checks that a number is *producible*, not that it is used correctly. A figure quoted in
  the wrong sentence still passes.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from test_aggregate import _bloc_panel, boundary_run, boundary_summary

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return _README.read_text()


@pytest.fixture(scope="module")
def demo_output() -> str:
    from voteworth.demo import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    return buf.getvalue()


def _percentages(block: str) -> set[str]:
    """Every percentage in a block of text, normalised (drop leading zeros in '05.1%')."""
    return {f"{float(m):g}" for m in re.findall(r"(\d+\.\d)%", block)}


# Identifiers whose digits are not measurements: DOIs, arXiv ids, URLs, years in citations.
# Stripped before numbers are extracted, so that 10.5281/zenodo.21775275 is not read as a
# claim about 10.53.
_NOT_A_MEASUREMENT = re.compile(
    r"https?://\S+"                       # links
    r"|10\.\d{4,}/\S+"                    # DOIs
    r"|arXiv:\d{4}\.\d{4,}"               # arXiv ids
    r"|\(\d{4}[a-z]?(,\s*\d{4})*\)"       # citation years: (2003), (1982, 1984)
    r"|\b(19|20|21|22|23|24|25|26)\d{2}\b"  # bare years
    r"|\bgpt-[\d.]+\S*"                   # model names: gpt-4.1-mini, gpt-4o-mini
    r"|\bclaude-\S+|\bmistral-\S+"        # ditto
    r"|\bApache-[\d.]+|CC BY [\d.]+"       # licence versions
    r"|\bPython [\d.]+|>=[\d.]+"           # version constraints
    r"|\bv?\d+\.\d+\.\d+\b"                # release numbers of other packages: 0.2.0
)


def _decimals(block: str) -> set[str]:
    """Every decimal quantity in a block, after identifiers are removed.

    Broader than `_percentages` on purpose. The first version of this file matched only
    `\\d+\\.\\d%`, which covered three numbers in the prose and silently ignored every n_eff
    value, every absolute gap and every 'N points' figure — a gate that covered its own
    checklist rather than the subject.
    """
    cleaned = _NOT_A_MEASUREMENT.sub(" ", block)
    # A trailing sentence period is allowed after the number ("…moves it to 14.9."). The first
    # version forbade it, and silently dropped every figure that ended a sentence.
    # Three decimals, not two: the base/definition table prints 0.438 / 0.200 / 0.237, and a
    # two-decimal pattern would have walked straight past the table it exists to check.
    return {f"{float(m):g}" for m in re.findall(r"(?<![\w.])(\d{1,3}\.\d{1,3})(?![\w\d])", cleaned)}


def test_readme_output_block_matches_a_real_run(readme, demo_output):
    """The fenced block under 'What it prints' must be output this package actually produces.

    Every percentage quoted there has to appear in a live run. This is the assertion that
    would have caught a README kept from an older version of the mechanism.
    """
    block = readme.split("## What it prints", 1)[1].split("```", 2)[1]
    quoted = _percentages(block)
    produced = _percentages(demo_output)

    assert quoted, "no percentages found under 'What it prints' — has the README changed shape?"
    missing = sorted(quoted - produced, key=float)
    assert not missing, (
        f"the README advertises {missing} but a live run does not produce them.\n"
        f"Either the mechanism changed or the README is stale.\n\nRun output:\n{demo_output}"
    )


def _output_block(readme: str) -> list[str]:
    """The lines of the fenced block under 'What it prints'."""
    return readme.split("## What it prints", 1)[1].split("```", 2)[1].strip("\n").splitlines()


def _block_problems(block: list[str], demo_output: str) -> list[str]:
    """Where the README's output block departs from a live run; nothing if it is a run of it."""
    out = demo_output.splitlines()
    if not block:
        return ["the block under 'What it prints' is empty"]
    starts = [i for i, line in enumerate(out) if line == block[0]]
    if not starts:
        return [f"the block's first line is not printed by the demo: {block[0]!r}"]
    best = []
    for start in starts:
        problems = [
            f"block line {k + 1}: README {line!r}, run {out[start + k] if start + k < len(out) else None!r}"
            for k, line in enumerate(block)
            if start + k >= len(out) or out[start + k] != line
        ]
        if not problems:
            return []
        if not best or len(problems) < len(best):
            best = problems
    return best


def test_readme_output_block_is_a_verbatim_run(readme, demo_output):
    """Line for line, the block under 'What it prints' is a contiguous run of the demo's output.

    The check above reads only the percentages. Two printed figures in each other's place, or an
    n_eff line kept from an older run, passed it. Until 0.1.4 the line-for-line equality was
    checked by hand before a release; it is a test now.
    """
    problems = _block_problems(_output_block(readme), demo_output)
    assert not problems, "the README's output block is not what the demo prints:\n" + "\n".join(problems)


def test_the_output_block_check_can_fire(readme, demo_output):
    """Control arm: one figure changed in one line of the block is reported, on that line only."""
    block = _output_block(readme)
    k = next(i for i, line in enumerate(block) if "effective votes" in line)
    changed = block[:k] + [re.sub(r"\d\.\d\d", "9.99", block[k], count=1)] + block[k + 1:]
    assert changed[k] != block[k], "the arm changed nothing; has the block changed shape?"
    problems = _block_problems(changed, demo_output)
    assert len(problems) == 1 and problems[0].startswith(f"block line {k + 1}:"), problems


# Figures the README quotes that a live demo run cannot print: other people's measurements, our
# own measurement note, and values from the synthetic boundary experiment. Each is listed here
# with where it comes from. Anything in the prose that is NOT on this list and not derived from
# printed figures (`_derived_figures` below) is a claim about our own run and must be
# reproducible from it.
#
# A README that cites the literature needs this list. Without it the test would either pass
# vacuously (drop the prose check) or force us to delete honest citations — and the second
# failure mode is worse than the first.
#
# Until 0.1.4 the list also carried 13.8, 2.3, 0.243 and 0.43, figures of our own declared away
# by a note: three are arithmetic on printed figures, the fourth was said to come from twenty
# seeds that no test ran (review of 2026-11-21). The first three are recomputed from the
# printed run now; 0.43 is recomputed from the boundary experiment, and its entry names the test.
FOREIGN_FIGURES = {
    # ── cited work, including our own measurement note ───────────────────────────
    # other people's measurements. Read at source on 2026-11-21; the README carries that date.
    # Two entries here were wrong until then: a "band" of +0.05 to +0.11 that one of its two
    # sources did not support, and a range attributed to cross-provider pairs that the source
    # reports as the mean over ALL pairs. A provenance note is a claim too.
    "2.2": "Kohli 2026 (arXiv:2605.29800, preprint): nine judges over seven families, effective "
           "votes from the phi of error indicators against gold, not from raw agreement",
    "0.047": "Kohli 2026, section 5.3: same-family minus cross-family phi of error indicators, MNLI",
    "0.109": "Kohli 2026, section 5.3: the same difference on RewardBench",
    "0.022": "Kim et al. ICML 2025 (arXiv:2506.07962), Table 1: same-company coefficient, Helm",
    "0.066": "Kim et al. 2025, Table 1: same-company coefficient, HuggingFace leaderboard",
    "0.42": "Kim et al. 2025, section 3.2: mean agreement over ALL model pairs when both are "
            "wrong, HuggingFace leaderboard (0.423), quoted as the lower end of 0.42-0.60",
    "0.423": "Kim et al. 2025, section 3.2: the same mean, HuggingFace leaderboard, unrounded",
    "0.6": "Kim et al. 2025, section 3.2: the same mean over all pairs on Helm (also as 0.60)",
    # our own measurement note (10.5281/zenodo.21775275), on all 57 items rather than the
    # held-out half. Our own work, but not this run, so it is cited rather than produced.
    "15.3": "our measurement note (10.5281/zenodo.21775275): overconfidence on all 57 items",
    "14.9": "our measurement note: provenance-weighted overconfidence on all 57 items",
    # ── the synthetic boundary experiment and other values bound by a named test ─────
    # Not printed by the demo. Each entry names the test that recomputes the value, and
    # test_foreign_figures_name_tests_that_exist holds those names to the suite. 'Reproduced by
    # tests/test_aggregate.py' used to mean the file asserted the series FALLS; the values
    # themselves were unchecked, and two of the four were wrong.
    "1.91": "n_eff at a 3-of-10 bloc, mean over five seeds; test_boundary_figures_match_the_readme",
    "1.68": "n_eff at a 5-of-10 bloc, mean over five seeds; test_boundary_figures_match_the_readme",
    "1.55": "n_eff at a 6-of-10 bloc, mean over five seeds; test_boundary_figures_match_the_readme",
    "1.26": "n_eff at an 8-of-10 bloc, mean over five seeds; test_boundary_figures_match_the_readme",
    "0.43": "mean independent-voter weight over twenty seeds of the boundary experiment; "
            "test_boundary_weights_and_gap_match_the_readme",
    "1": "a weight at full: the bloc's 1.00 in the boundary experiment "
         "(test_boundary_weights_and_gap_match_the_readme), a voter that never co-deviates "
         "(test_independence_weights_downweight_correlated_bloc), and every learned weight of "
         "ten independent voters right 80% of the time (test_raw_n_eff_rises_with_accuracy_alone)",
    "1.4": "n_eff of ten independent voters right 80% of the time with two answer options; "
           "test_two_answer_options_put_every_weight_at_one_tenth",
    "0.1": "the learned weight of each of those ten voters; "
           "test_two_answer_options_put_every_weight_at_one_tenth",
    "0.5": "two uses: the weight a voter locked to one partner receives "
           "(test_independence_weights_downweight_correlated_bloc), and the bloc share from "
           "which the demo warns (test_demo_warns_at_exactly_half)",
    "0.333": "1/3, the structural weight of a three-voter bloc in the usage example; "
             "test_structural_weights_downweight_shared_source and "
             "test_readme_code_blocks_run_as_written",
}


# ── figures the README derives from printed ones ─────────────────────────────
#
# The README states figures that the demo does not print but that follow from figures it does:
# the ratio of the same-model to the cross-model share, and the gap of the pair means. Each is
# recomputed here from the printed run, so that a change which moves a printed figure moves the
# derived one with it, and each is checked in the sentence that states it. Until the fourth
# review of 0.1.4 (2026-11-21) the README also derived the plain majority's distance to the best
# single voter; that voter is picked after the fact on the same items, and the comparison is gone.


def _printed(pattern: str, demo_output: str) -> float:
    match = re.search(pattern, demo_output, re.M)
    assert match, f"the demo no longer prints a line matching {pattern!r}:\n{demo_output}"
    return float(match.group(1))


def _derived_figures(demo_output: str) -> dict[str, float]:
    """The derived figures, from the printed run: what a reader of the output would compute."""
    same = _printed(r"^  same model\s+(\d+\.\d)%", demo_output)
    cross = _printed(r"^  different model\s+(\d+\.\d)%", demo_output)
    pair_same = _printed(r"voter PAIRS \((\d\.\d{3}) / ", demo_output)
    pair_cross = _printed(r"voter PAIRS \(\d\.\d{3} / (\d\.\d{3})\)", demo_output)
    return {
        "same-model over cross-model share": round(same / cross, 1),
        "gap of the pair means": round(pair_same - pair_cross, 3),
    }


# The sentence that states each derived figure. The gap of the pair means is stated in the span
# sentence and checked there (test_gap_span_and_ratio_factor_match_a_live_run).
_DERIVED_SENTENCES = {
    "same-model over cross-model share":
        r"pick the same answer (\d+\.\d)× as often as cross-model pairs",
}


def _derived_problems(text: str, demo_output: str) -> list[str]:
    derived = _derived_figures(demo_output)
    flat = " ".join(text.replace("**", "").split())
    problems = []
    for key, pattern in _DERIVED_SENTENCES.items():
        match = re.search(pattern, flat)
        if not match:
            problems.append(f"{key}: no sentence matching {pattern!r}")
        elif float(match.group(1)) != derived[key]:
            problems.append(f"{key}: text {match.group(1)}, recomputed from the run {derived[key]:g}")
    return problems


def test_derived_figures_are_recomputed_from_the_printed_run(readme, demo_output):
    """The same-model ratio is arithmetic on printed figures, so it is recomputed from the printed
    run rather than declared."""
    problems = _derived_problems(readme, demo_output)
    assert not problems, "the README's derived figures do not follow from a live run:\n" + "\n".join(problems)


def test_the_derived_figure_check_can_fire(demo_output):
    """Control arm: a derived figure one off must be reported."""
    derived = _derived_figures(demo_output)
    off = derived["same-model over cross-model share"] + 1
    planted = f"Same-model pairs pick the same answer {off:.1f}× as often as cross-model pairs."
    problems = _derived_problems(planted, demo_output)
    assert len(problems) == 1 and problems[0].startswith("same-model over cross-model share"), problems
    assert len(_derived_problems("no such sentences here", demo_output)) == len(_DERIVED_SENTENCES)


# ── the boundary experiment, as the README reports it ────────────────────────
#
# The boundary paragraph reports one experiment in a run of figures. Until 0.1.4 two of them
# were checked nowhere: the independents' mean weight was declared in FOREIGN_FIGURES with
# "twenty seeds" that no test ran, and the median gap was checked only in the docstring of
# bloc_share (review of 2026-11-21). They are read out of the README here and compared with
# `boundary_summary`, the computation the docstring check reads too.

_BOUNDARY_SENTENCE = re.compile(
    r"a synthetic (\w+)-voter panel with a (\w+)-voter bloc that is wrong together (\d+)% of the "
    r"time, across (\w+) seeds: the bloc keeps full weight (\d\.\d\d) in (\d+)/(\d+) runs, the "
    r"independents drop to (\d\.\d\d), the weighted vote is identical to the naive one in "
    r"(\d+)/(\d+), and both land a median (\d+) points below the best single voter"
)


def _boundary_problems(text: str, summary: dict) -> list[str]:
    """What is wrong with the boundary sentence in `text`, against the experiment's summary."""
    import inspect

    match = _BOUNDARY_SENTENCE.search(" ".join(text.replace("**", "").split()))
    if not match:
        return ["no boundary sentence of the form 'a synthetic N-voter panel with a B-voter bloc "
                "… a median P points below the best single voter'"]
    panel, bloc, wrong, seeds, full, at_full, runs, indep, equal, runs2, points = match.groups()
    run_defaults = inspect.signature(boundary_run).parameters
    n_bloc, n_indep = run_defaults["n_bloc"].default, run_defaults["n_indep"].default
    p_right = inspect.signature(_bloc_panel).parameters["p_bloc_right"].default
    expected = [
        ("panel size", _WORD_NUMBERS.get(panel.lower()), n_bloc + n_indep),
        ("bloc size", _WORD_NUMBERS.get(bloc.lower()), n_bloc),
        ("items on which the bloc is wrong together, %", int(wrong), round(100 * (1 - p_right))),
        ("seeds", _WORD_NUMBERS.get(seeds.lower()), summary["runs"]),
        ("the bloc's weight", float(full), 1.0),
        ("runs with the bloc at full weight", int(at_full), summary["bloc_at_full"]),
        ("runs", int(runs), summary["runs"]),
        ("the independents' mean weight", float(indep), round(summary["indep_w"], 2)),
        ("runs with the weighted vote equal to the naive one", int(equal), summary["naive_equal"]),
        ("runs", int(runs2), summary["runs"]),
        ("median points below the best single voter", int(points), summary["median_points"]),
    ]
    return [f"{what}: text {said}, experiment {got}" for what, said, got in expected if said != got]


def test_boundary_weights_and_gap_match_the_readme(readme):
    """The boundary paragraph's figures are the boundary experiment's, over the seeds it names."""
    problems = _boundary_problems(readme, boundary_summary())
    assert not problems, "the README's boundary paragraph does not match the experiment:\n" + "\n".join(problems)


def test_the_boundary_check_can_fire(readme):
    """Control arm, constructed: the README's own sentence with the independents' weight one
    hundredth off and the median three points off must be reported on exactly those two counts."""
    summary = boundary_summary()
    match = _BOUNDARY_SENTENCE.search(" ".join(readme.replace("**", "").split()))
    assert match, "the README no longer carries the boundary sentence this arm plants into"
    indep, points = match.group(8), match.group(11)
    planted = (match.group(0)
               .replace(f"drop to {indep}", f"drop to {float(indep) + 0.01:.2f}")
               .replace(f"a median {points} points", f"a median {int(points) + 3} points"))
    problems = _boundary_problems(planted, summary)
    assert len(problems) == 2, problems
    assert _boundary_problems("no such sentence here", summary), "a missing sentence passed"


def test_readme_prose_numbers_are_produced(readme, demo_output):
    """Every number in the prose is reproducible from a live run, derived from printed figures,
    or declared in FOREIGN_FIGURES with its source.

    This is the assertion that stops the README drifting from the code. It does not forbid
    citing other people's figures — it forbids citing them *indistinguishably* from ours.
    """
    derived = {f"{value:g}" for value in _derived_figures(demo_output).values()}
    produced = _decimals(demo_output) | _percentages(demo_output) | derived
    # The WHOLE file, not one section. Scoping this to a heading was how the boundary
    # section's figures went unchecked while the test reported green.
    unexplained = [
        q for q in _decimals(readme) if q not in produced and q not in FOREIGN_FIGURES
    ]
    assert not unexplained, (
        f"the README states {unexplained} but a live run does not produce them and they are "
        f"not declared in FOREIGN_FIGURES with a source.\n\nRun output:\n{demo_output}"
    )


def test_foreign_figures_list_is_not_a_blanket(readme, demo_output):
    """Control arm for the list above: it must not have grown into an excuse.

    If a figure is declared foreign but the run *does* produce it, the declaration is wrong and
    hides a real check. And the list must stay small relative to what the run itself covers.
    """
    produced = _decimals(demo_output) | _percentages(demo_output)
    wrongly_declared = sorted(set(FOREIGN_FIGURES) & produced)
    assert not wrongly_declared, (
        f"declared foreign but produced by a live run: {wrongly_declared} — remove them from "
        "FOREIGN_FIGURES so the real check applies"
    )
    for figure, source in FOREIGN_FIGURES.items():
        assert len(source) > 20, f"{figure} has no usable provenance note: {source!r}"

    # An entry that never fires is dead weight, and dead weight is where a blanket exemption
    # starts. Every declared figure must actually occur in the README.
    in_readme = _decimals(_README.read_text())
    unused = sorted(set(FOREIGN_FIGURES) - in_readme, key=float)
    assert not unused, f"declared but not present in the README: {unused} — delete them"

    # And a derived figure must not be declared as well: the declaration would hide the check.
    derived = {f"{value:g}" for value in _derived_figures(demo_output).values()}
    assert not set(FOREIGN_FIGURES) & derived, (
        f"declared foreign but derived from the run: {sorted(set(FOREIGN_FIGURES) & derived)}"
    )


def _all_test_names() -> set[str]:
    names: set[str] = set()
    for path in (_ROOT / "tests").glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    return names


def test_foreign_figures_name_tests_that_exist():
    """An entry that says which test recomputes its value must name a test that exists.

    A provenance note pointing at a renamed or deleted test reads as coverage and is none.
    """
    names = _all_test_names()
    assert names, "no tests discovered — this check is broken"
    named = {
        figure: re.findall(r"\btest_\w+", source) for figure, source in FOREIGN_FIGURES.items()
    }
    missing = {f: [t for t in tests if t not in names] for f, tests in named.items()}
    missing = {f: ts for f, ts in missing.items() if ts}
    assert not missing, f"FOREIGN_FIGURES names tests that do not exist: {missing}"
    assert any(named.values()), "no entry names a test any more; has the list changed shape?"


def test_advertised_module_command_runs():
    """`python -m voteworth.demo` is promised in the README's install block."""
    result = subprocess.run(
        [sys.executable, "-m", "voteworth.demo"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert result.returncode == 0, f"the advertised command failed:\n{result.stderr}"
    assert "INERT" in result.stdout


def test_advertised_example_script_runs():
    """`python examples/quickstart.py` is the other advertised entry point."""
    result = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert result.returncode == 0, f"the advertised command failed:\n{result.stderr}"
    assert "INERT" in result.stdout


def test_both_entry_points_agree():
    """Two advertised commands, one implementation — they must print the same thing."""
    a = subprocess.run(
        [sys.executable, "-m", "voteworth.demo"],
        capture_output=True, text=True, cwd=_ROOT,
    ).stdout
    b = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / "quickstart.py")],
        capture_output=True, text=True, cwd=_ROOT,
    ).stdout
    assert a == b, "the two entry points have drifted apart"


def _version_problems(pyproject: str, init: str, citation: str) -> list[str]:
    """The three places a release names its version, read as text."""
    found = {
        "pyproject.toml": re.search(r'^version = "([^"]+)"$', pyproject, re.M),
        "__version__": re.search(r'^__version__ = "([^"]+)"$', init, re.M),
        "CITATION.cff": re.search(r"^version: (\S+)$", citation, re.M),
    }
    missing = [k for k, m in found.items() if not m]
    if missing:
        return [f"no version found in {missing}"]
    versions = {k: m.group(1) for k, m in found.items()}
    return [] if len(set(versions.values())) == 1 else [f"the versions differ: {versions}"]


def test_versions_agree():
    """v0.1.1 shipped with __version__ still at 0.1.0. publish.yml stops such a release; this
    finds it before one is made."""
    problems = _version_problems(
        (_ROOT / "pyproject.toml").read_text(),
        (_ROOT / "src" / "voteworth" / "__init__.py").read_text(),
        (_ROOT / "CITATION.cff").read_text(),
    )
    assert not problems, problems


def test_the_version_check_can_fire():
    """Control arm, constructed: the 0.1.1 mismatch must be reported."""
    assert _version_problems('version = "0.1.1"\n', '__version__ = "0.1.0"\n', "version: 0.1.1\n")
    assert not _version_problems('version = "0.1.4"\n', '__version__ = "0.1.4"\n', "version: 0.1.4\n")


def test_cited_dois_are_consistent(readme):
    """Every DOI in the README must also be declared in pyproject, so the two cannot drift."""
    in_readme = set(re.findall(r"10\.5281/zenodo\.\d+", readme))
    in_pyproject = set(re.findall(r"10\.5281/zenodo\.\d+", (_ROOT / "pyproject.toml").read_text()))
    assert in_readme, "no DOIs found in the README"
    assert in_readme == in_pyproject, (
        f"README and pyproject cite different records.\n"
        f"only in README: {sorted(in_readme - in_pyproject)}\n"
        f"only in pyproject: {sorted(in_pyproject - in_readme)}"
    )


# Figures corrected elsewhere, which must never reappear on a public surface.
#
# Assembled from fragments rather than written out: a guard that spells its own needles in
# plain text finds ITSELF and reports an offender that is only the guard. That happened here
# on the first run. Excluding this file would have hidden the self-hit — and with it any real
# hit in a test file — so the needles are built instead, and the scanner below stays able to
# read every shipped file including this one.
_WITHDRAWN = [
    # Figures that were corrected after publication elsewhere and must not reappear here.
    "1" + "/106", "12" + "/12", "91" + ".3%", "91" + ",3",
    # Figures from a *different* work line that this package has no business carrying at all.
    # It cites three records and reports its own run; any benchmark headline appearing here
    # would have arrived by copy-paste, which is exactly how a withdrawn number survives a
    # rewrite: the sentence around it changes, the number looks precise, nobody re-checks it
    # at source. Listed as out-of-scope, not as disputed — this package makes no claim either
    # way about their correctness.
    #
    # Note on the fragments: each must break so that NO listed pattern appears whole anywhere
    # in this file, not even inside a longer fragment. A first attempt split "6" + ".2e-31",
    # which left "e-31" literal and made the guard report itself. And a pattern as short as
    # "e-31" is too generic to list at all — it matches ordinary hyphenated text.
    "6.2e-" + "31", "6,2e-" + "31",
    "LongMemEval-" + "500",
]

def _scan_for_withdrawn(root: Path) -> list[str]:
    """Every occurrence of a withdrawn figure in the shipped text files under `root`.

    The walk and the suffix list both come from `shipped`, not from a private copy here. The
    copy that used to live at this spot skipped `.venv` and `.git` by name; pointed at a tree
    whose virtual environment was called anything else, it scanned site-packages and reported
    figures out of numpy's own test data. Measured 2026-09-21 on a built sdist.
    """
    from shipped import _TEXT_NAMES, _TEXT_SUFFIXES, walk_files

    offenders = []
    for path, _ in walk_files(root):
        if path.suffix not in _TEXT_SUFFIXES and path.name not in _TEXT_NAMES:
            continue
        # Flattened: a figure split by a line wrap ("91\n.3%") must still be caught. This is
        # a PROHIBITION check, and a prohibition that misses reports clean — the one failure
        # direction that looks exactly like success.
        text = " ".join(path.read_text(errors="ignore").split())
        for figure in _WITHDRAWN:
            # Whitespace-tolerant, because flattening a line wrap inside a figure leaves a
            # space where none belongs, and a literal substring search then misses it.
            # Allowing optional spaces between characters costs nothing in false alarms:
            # prose does not space out digits.
            #
            # NOTE — do not write an example of such a figure in this comment. The tolerant
            # pattern would match it, and the guard would report itself. That happened on the
            # first attempt here, and it is the third time in this repository that a check
            # meant to find something found its own description of it. Whoever documents an
            # absence ends up quoting the thing that is supposed to be absent.
            pattern = r"\s*".join(re.escape(c) for c in figure)
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(root)}: {figure!r}")
    return offenders


def test_no_withdrawn_figures_anywhere():
    """The guard itself: no shipped file may carry a withdrawn figure."""
    offenders = _scan_for_withdrawn(_ROOT)
    assert not offenders, "withdrawn figures found on a shipped surface:\n" + "\n".join(offenders)


def test_the_guard_can_fire(tmp_path):
    """Control arm. The guard is pointed at a directory that DOES contain each figure, and
    must report all of them.

    Without this, a guard whose needles silently stopped matching — a renamed suffix, a
    changed encoding, a typo in a fragment — would report clean forever and be
    indistinguishable from a clean repository. Asserting on a real detection is the only way
    to know the clean result means anything.
    """
    planted = tmp_path / "decoy.md"
    planted.write_text("\n".join(f"a line carrying {fig} in it" for fig in _WITHDRAWN))
    (tmp_path / "ignored.rst").write_text(_WITHDRAWN[0])  # wrong suffix → must NOT be flagged

    found = _scan_for_withdrawn(tmp_path)
    assert len(found) == len(_WITHDRAWN), (
        f"the guard found {len(found)} of {len(_WITHDRAWN)} planted figures: {found}"
    )
    assert all("decoy.md" in f for f in found), f"unexpected file flagged: {found}"


def test_the_guard_reads_this_file_too(tmp_path):
    """The scanner must not be blind to test files — that is where the self-hit occurred, and
    a scanner that skipped them would miss a real leak in a fixture."""
    planted = tmp_path / "test_something.py"
    planted.write_text(f"LEAK = {_WITHDRAWN[0]!r}")
    assert _scan_for_withdrawn(tmp_path), "the scanner skips test files"


# ── test counts: the integer gap this file's own docstring named ─────────────
#
# The docstring above has said since it was written that plain integers "pass unchecked", and
# listed the risk: "a future claim about our own measurement written in one of those forms
# would slip through". Two did. The README advertised "35 tests" in one place and "120 tests"
# in another for the same command, while the suite held 123 — three numbers, no two alike, all
# of them green, because `_decimals` matches only digits with a decimal point.
#
# Naming a limit is not closing it. A general integer pattern is the wrong closure — it would
# read years, arXiv ids and "80 items" as measurements, which is the false-alarm trap an
# earlier version of this file already fell into. So this binds the one claim that needs it,
# exactly: a count of tests is checked against a count of tests.


def _claimed_test_counts(text: str) -> list[int]:
    """Every "N tests" the README advertises for running the suite."""
    return [int(m) for m in re.findall(r"(?<![\w.])(\d{1,4})\s+tests\b", text)]


def _collected_test_count() -> int:
    """What `pytest` actually collects here, asked of pytest rather than assumed."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    assert match, (
        "could not read a collected count out of pytest's own output — this guard cannot "
        f"report a number it did not measure:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    return int(match.group(1))


def test_advertised_test_count_is_the_real_one(readme):
    """Every "N tests" in the README must be the number the suite collects."""
    claimed = _claimed_test_counts(readme)
    assert claimed, "the README no longer advertises a test count — has it changed shape?"
    actual = _collected_test_count()
    wrong = sorted({c for c in claimed if c != actual})
    assert not wrong, (
        f"the README advertises {wrong} tests; pytest collects {actual}. "
        "Update the README — the count moves whenever a test is added."
    )


def test_the_test_count_guard_can_fire():
    """Control arm. A guard that only ever sees agreement has not been seen to fire.

    Both halves: the extractor must find a wrong number in text that carries one, and must not
    invent one in text that carries none.
    """
    actual = _collected_test_count()
    planted = f"Run `pytest` — {actual + 7} tests, including the mutation gate."
    found = _claimed_test_counts(planted)
    assert found == [actual + 7], f"the extractor read {found} out of a planted claim"
    assert any(c != actual for c in found), "the planted wrong count was not seen as wrong"
    assert _claimed_test_counts("no counts here, only 80 items and 10 voters") == [], (
        "the extractor invents counts in prose that advertises none"
    )


# ── what the source distribution carries ─────────────────────────────────────


def _manifest_rules() -> tuple[set[str], list[tuple[str, tuple[str, ...]]]]:
    """(explicit includes, recursive includes) read out of MANIFEST.in."""
    includes: set[str] = set()
    recursive: list[tuple[str, tuple[str, ...]]] = []
    for raw in (_ROOT / "MANIFEST.in").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "include":
            includes.update(parts[1:])
        elif parts[0] == "recursive-include" and len(parts) >= 3:
            recursive.append((parts[1], tuple(parts[2:])))
    return includes, recursive


# Files setuptools ships whatever the manifest says: the build inputs and the metadata it
# generates. Named here rather than assumed, so that a file arriving by accident is not
# mistaken for one arriving by rule.
_ALWAYS_SHIPPED = {"pyproject.toml", "README.md", "LICENSE", "NOTICE", "setup.cfg", "PKG-INFO"}


def _would_ship(rel: str) -> bool:
    """Is this path covered by a packaging rule, rather than by luck?"""
    import fnmatch

    if rel in _ALWAYS_SHIPPED or rel.startswith("src/"):
        return True  # src/ is covered by packages.find plus package-data
    includes, recursive = _manifest_rules()
    if rel in includes or any(fnmatch.fnmatch(rel, pat) for pat in includes):
        return True
    for directory, patterns in recursive:
        if rel.startswith(directory.rstrip("/") + "/"):
            name = rel[len(directory.rstrip("/")) + 1:]
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                return True
    return False


# The directories this package owns. A file inside them is part of the package and must be
# carried by a rule; a file beside them in the working directory is somebody's scratch note.
# Scoped this way on purpose: the first version asked the question of every file under the
# root, and went red because a measurement script had left its output there. A gate that fires
# on a stray file teaches people to stop reading it, which is how the gates in this suite lost
# their audience before.
_OWNED_DIRS = ("src/", "tests/", "examples/", ".github/")
_OWNED_ROOT_FILES = ("README.md", "NOTICE", "LICENSE", "CITATION.cff", "pyproject.toml",
                     "MANIFEST.in", ".gitignore")


def _files_the_package_owns() -> list[str]:
    """Discovered, not typed — but restricted to what the package is responsible for.

    The walk comes from `shipped`, not from a private skip list beside it: a second copy of
    "which directories to ignore" is a second place the truth lives, and this one was written
    with the same handful of names that let a differently named virtual environment into the
    scan.
    """
    from shipped import discover_all

    return [
        f for f in discover_all()
        if f.startswith(_OWNED_DIRS) or f in _OWNED_ROOT_FILES
    ]


def test_every_file_in_the_tree_is_covered_by_a_packaging_rule():
    """The README says the checks above run from a clone OR from the source distribution.

    That sentence is only true if the distribution carries the files those checks need. On
    2026-09-21 it did not: setuptools' default file set ships `tests/test*.py` but not
    `tests/shipped.py`, which every prohibition gate imports, so a built sdist collected two
    import errors and ran none of its tests — while this README promised they could all be
    run. Nothing in the suite could see it, because the suite installs the working tree.

    This is the cheap, offline half: every file in the tree must be matched by a rule. The
    authoritative half is the `artefact` job in CI, which builds the real artefact, diffs it
    against `git ls-files` and runs the suite inside it. Two arms, because a static model of
    packaging rules can drift from what the packager actually does.
    """
    uncovered = [f for f in _files_the_package_owns() if not _would_ship(f)]
    assert not uncovered, (
        "these files are in the tree but no packaging rule carries them into the sdist:\n  "
        + "\n  ".join(uncovered)
        + "\nAdd them to MANIFEST.in, or they will be missing for anyone who installs from "
        "source rather than from a clone."
    )


def test_the_packaging_rule_check_can_fire():
    """Control arm: a file that no rule mentions must be reported as uncovered."""
    assert not _would_ship("tests/helper.txt"), (
        "an unmatched path was reported as covered — the matcher says yes to everything"
    )
    assert _would_ship("tests/shipped.py"), (
        "the file whose absence caused the finding is still not covered by a rule"
    )
    assert _would_ship("examples/quickstart.py")

    # And the scope: a scratch file beside the package is not the package's problem, while a
    # new file inside a directory it owns is.
    owned = _files_the_package_owns()
    assert "tests/shipped.py" in owned, "the scope no longer covers the directory that broke"
    assert not [f for f in owned if f.startswith("notes") or f.endswith("demo.txt")], (
        "the scope reaches outside the package's own directories"
    )


# ── spans, factors, roles and code: what a figure-by-figure check cannot see ────
#
# Everything above asks whether each figure is producible. A check before the 0.1.3 upload
# (2026-11-21) found statements that passed it and were still wrong: a span that left out one
# of the variants it claimed to cover, a factor read off three of four rows, a headline that
# gave the ten voters the role of the one judge, and a code example that raised as written.
# Each guard below is red on the 0.1.3 text, and its control arm plants that text and must fire.
# Where the 0.1.3 text lacks the shape a guard reads, the arm plants the real 0.1.3 sentence,
# which must be reported as missing, and a variant in today's shape carrying the 0.1.3 figures,
# named in the arm as constructed. The span sentence is that case: in 0.1.3 it gave no range
# for the ratio at all.

_WORD_NUMBERS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
}

_SPAN_SENTENCE = re.compile(
    r"The absolute gap stays between \+(\d\.\d{3}) and \+(\d\.\d{3}) through all of it; "
    r"the ratio swings between \+(\d+)% and \+(\d+)%, a factor of (\w+)\."
)


def _printed_variants(demo_output: str) -> list[tuple[float, int]]:
    """(gap, ratio in percent) for every variant the demo prints.

    The three pooled rows are read as printed. The pair mean is printed only as its two rates,
    so its gap and ratio are formed from those printed rates: the arithmetic a reader of the
    output would do, and the reason "through all of it" has to include it.
    """
    rows = re.findall(
        r"^  \S.*?\s(\d\.\d{3})\s+(\d\.\d{3})\s+\+(\d\.\d{3})\s+\+(\d+)%\s*$", demo_output, re.M
    )
    pair = re.search(r"voter PAIRS \((\d\.\d{3}) / (\d\.\d{3})\)", demo_output)
    assert len(rows) == 3 and pair, (
        f"the demo no longer prints the three rows and the pair mean this check reads:\n{demo_output}"
    )
    variants = [(float(gap), int(ratio)) for _, _, gap, ratio in rows]
    same, cross = float(pair.group(1)), float(pair.group(2))
    variants.append((round(same - cross, 3), round(100 * (same - cross) / cross)))
    return variants


def _span_problems(text: str, demo_output: str) -> list[str]:
    """What is wrong with the span sentence in `text`, measured against a live run."""
    match = _SPAN_SENTENCE.search(" ".join(text.split()))
    if not match:
        return ["no sentence of the form 'The absolute gap stays between +X and +Y through all "
                "of it; the ratio swings between +A% and +B%, a factor of N.'"]
    variants = _printed_variants(demo_output)
    gaps = [gap for gap, _ in variants]
    ratios = [ratio for _, ratio in variants]
    factor = round(max(ratios) / min(ratios))
    problems = []
    if float(match.group(1)) != min(gaps):
        problems.append(f"lowest gap: text +{match.group(1)}, run +{min(gaps):.3f}")
    if float(match.group(2)) != max(gaps):
        problems.append(f"highest gap: text +{match.group(2)}, run +{max(gaps):.3f}")
    if int(match.group(3)) != min(ratios):
        problems.append(f"lowest ratio: text +{match.group(3)}%, run +{min(ratios)}%")
    if int(match.group(4)) != max(ratios):
        problems.append(f"highest ratio: text +{match.group(4)}%, run +{max(ratios)}%")
    if _WORD_NUMBERS.get(match.group(5).lower()) != factor:
        problems.append(f"factor: text {match.group(5)!r}, run {factor}")
    return problems


def test_gap_span_and_ratio_factor_match_a_live_run(readme, demo_output):
    """'Through all of it' means all four variants the demo prints, the pair mean included."""
    problems = _span_problems(readme, demo_output)
    assert not problems, "the README's span sentence does not match a live run:\n" + "\n".join(problems)


def test_the_span_check_can_fire(demo_output):
    """Control arm, two plantings.

    The real 0.1.3 sentence, word for word, gives no range for the ratio, so the check must
    report that it found no sentence of the form. The second planting is constructed, not
    quoted: today's form carrying the 0.1.3 figures, read off three of the four variants
    (highest gap, highest ratio, and the factor between the ratios), which must be reported on
    exactly those three counts.
    """
    real_013 = ("The absolute gap stays between +0.097 and +0.237 through all of it; the ratio "
                "swings by a factor of seven.")
    problems = _span_problems(real_013, demo_output)
    assert len(problems) == 1 and problems[0].startswith("no sentence of the form"), problems
    constructed = ("The absolute gap stays between +0.097 and +0.237 through all of it; the ratio "
                   "swings between +16% and +119%, a factor of seven.")
    problems = _span_problems(constructed, demo_output)
    assert len(problems) == 3, f"expected highest gap, highest ratio and factor; got {problems}"
    assert _span_problems("no such sentence here", demo_output), "a missing sentence passed"


# The words that name what the bundled ten do. They answered the questions; one separate model
# judged the answers. "judges" names the one, not the ten, and the 0.1.3 headline used it.
_ANSWERER_ROLES = {"voters", "writers"}


def _headline_role(text: str):
    """(count word, role noun) out of a headline like '**Ten LLM voters, 1.48 …'.

    The start of the line only, closed by a comma or, as in 0.1.3, a full stop: what follows
    carries the headline's figure, which test_n_eff_figures_in_the_prose_are_the_printed_ones reads.
    """
    match = re.search(r"^\*\*(\w+) (?:LLM )?(\w+)[.,] ", text, re.M)
    return match.groups() if match else None


def test_headline_names_the_ten_by_their_role_in_the_data(readme):
    import json

    import voteworth

    data = json.loads(
        (Path(voteworth.__file__).parent / "data" / "panel_10_voters.json").read_text()
    )
    found = _headline_role(readme)
    assert found, "the headline no longer has the shape '**Ten … voters, …'"
    count, role = found
    assert _WORD_NUMBERS.get(count.lower()) == len(data["panel"]) == len(data["writers"]), (
        f"the headline counts {count!r}; the bundled data hold {len(data['panel'])} voters"
    )
    assert isinstance(data.get("judge"), str), "the data no longer name one separate judge"
    assert role in _ANSWERER_ROLES, (
        f"the headline calls the ten {role!r}; in the bundled data they are writers, and their "
        f"answers were judged by {data['judge']}"
    )


def test_the_role_check_can_fire():
    """Control arm: the 0.1.3 headline must be read, and its role word refused."""
    assert _headline_role("**Ten judges. Worth 1.48 independent votes.**") == ("Ten", "judges")
    assert "judges" not in _ANSWERER_ROLES
    assert _headline_role("no headline here") is None


# The prose states n_eff and the figures printed beside it: in the headline, in the first
# finding, and in the sentences on the baselines and on the items every voter got right. The
# figure check above sees only that each is producible, so two producible figures in each
# other's place pass it. Each sentence is read here against the lines the demo prints. Until
# the review of 2026-11-21 the headline also stated a baseline, which the review withdrew: the
# comparison cannot separate differences in item difficulty from shared errors.
_HEADLINE = re.compile(r"^\*\*\w+ (?:LLM )?\w+, (\d\.\d\d) effective votes by raw agreement\.\*\*$", re.M)
_LEAD = re.compile(
    r"came to (\d\.\d\d) effective votes by raw agreement, on the (\d+) items with a gold answer "
    r"\((\d\.\d\d) on all (\d+)\)"
)
_BASELINES_SENTENCE = re.compile(
    r"(\d\.\d\d) if no two wrong answers coincide, (\d\.\d\d) if wrong answers coincide as they do "
    r"on each item"
)
_ALL_RIGHT_SENTENCE = re.compile(
    r"[Aa]ll (\w+) voters got (\d+) of the (\d+) right, where such voters with these accuracies would "
    r"get (\d\.\d\d)"
)


def _printed_n_eff_figures(demo_output: str) -> dict:
    """The figures the demo prints around n_eff, as printed (strings, not rounded again)."""
    found = {
        "n_eff": re.search(r"effective votes \(Kish, raw agreement\):\s+(\d\.\d\d) of (\d+)", demo_output),
        "items": re.search(r"on the (\d+) items with a gold answer; on all (\d+) items: (\d\.\d\d)", demo_output),
        "per_item": re.search(r"wrong answers coinciding as on each item:\s+(\d\.\d\d)", demo_output),
        "distinct": re.search(r"no two wrong answers alike:\s+(\d\.\d\d)", demo_output),
        "all_right": re.search(r"items all (\d+) voters got right:\s+(\d+) of (\d+)", demo_output),
        "expected": re.search(r"on equally hard items: (\d\.\d\d)", demo_output),
    }
    missing = [k for k, m in found.items() if not m]
    assert not missing, f"the demo no longer prints {missing} in the shape read here:\n{demo_output}"
    return {k: m.groups() for k, m in found.items()}


def _n_eff_figure_problems(text: str, demo_output: str) -> list[str]:
    printed = _printed_n_eff_figures(demo_output)
    ne = printed["n_eff"][0]
    answerable, total, ne_all = printed["items"]
    flat = " ".join(text.replace("**", "").split())
    problems = []
    headline = _HEADLINE.search(text)
    if not headline:
        problems.append("the headline is not of the form '**Ten LLM voters, N effective votes by raw "
                        "agreement.**'")
    elif headline.group(1) != ne:
        problems.append(f"the headline states {headline.group(1)}; the demo prints n_eff {ne}")
    lead = _LEAD.search(flat)
    if not lead:
        problems.append("no first finding of the form 'came to N effective votes by raw agreement, on "
                        "the K items with a gold answer (M on all T)'")
    elif lead.groups() != (ne, answerable, ne_all, total):
        problems.append(f"the first finding states {lead.groups()}; the demo prints {(ne, answerable, ne_all, total)}")
    baselines = _BASELINES_SENTENCE.search(flat)
    if not baselines:
        problems.append("no sentence stating the two baselines in the shape read here")
    elif baselines.groups() != (printed["distinct"][0], printed["per_item"][0]):
        problems.append(f"the baselines sentence states {baselines.groups()}; the demo prints "
                        f"{(printed['distinct'][0], printed['per_item'][0])}")
    all_right = _ALL_RIGHT_SENTENCE.search(flat)
    if not all_right:
        problems.append("no sentence stating the items every voter got right in the shape read here")
    else:
        voters, got, of, expected = all_right.groups()
        if (_WORD_NUMBERS.get(voters.lower()), got, of, expected) != (
            int(printed["all_right"][0]), printed["all_right"][1], printed["all_right"][2], printed["expected"][0]
        ):
            problems.append(f"the all-right sentence states {all_right.groups()}; the demo prints "
                            f"{printed['all_right']} and {printed['expected']}")
    return problems


def test_n_eff_figures_in_the_prose_are_the_printed_ones(readme, demo_output):
    """The headline, the first finding and the sentences on the baselines and on the items every
    voter got right state the figures the demo prints, each in its own place."""
    problems = _n_eff_figure_problems(readme, demo_output)
    assert not problems, "\n".join(problems)


def test_the_n_eff_figure_check_can_fire(demo_output):
    """Control arm: the 0.1.3 headline and the first 0.1.4 headline (a baseline beside n_eff,
    withdrawn) must be refused, and constructed sentences with figures in each other's place must
    be reported on exactly those counts."""
    printed = _printed_n_eff_figures(demo_output)
    ne, (answerable, total, ne_all) = printed["n_eff"][0], printed["items"]
    per_item, distinct = printed["per_item"][0], printed["distinct"][0]
    body = (f"Ten voters came to {ne} effective votes by raw agreement, on the {answerable} items with a "
            f"gold answer ({ne_all} on all {total}). {distinct} if no two wrong answers coincide, "
            f"{per_item} if wrong answers coincide as they do on each item. All ten voters got "
            f"{printed['all_right'][1]} of the {printed['all_right'][2]} right, where such voters with "
            f"these accuracies would get {printed['expected'][0]}.")
    assert not _n_eff_figure_problems(f"**Ten LLM voters, {ne} effective votes by raw agreement.**\n\n{body}", demo_output)
    for old in ("**Ten judges. Worth 1.48 independent votes.**",
                "**Ten LLM voters. Worth 1.48 effective votes; independent voters as accurate would be worth 2.03.**"):
        problems = _n_eff_figure_problems(f"{old}\n\n{body}", demo_output)
        assert len(problems) == 1 and problems[0].startswith("the headline is not"), problems
    headline = f"**Ten LLM voters, {ne} effective votes by raw agreement.**\n\n"
    swapped = body.replace(f"{distinct} if no", f"{per_item} if no").replace(f"{per_item} if wrong", f"{distinct} if wrong")
    problems = _n_eff_figure_problems(headline + swapped, demo_output)
    assert len(problems) == 1 and problems[0].startswith("the baselines sentence"), problems
    # the first finding and the all-right sentence, each with one figure planted wrong
    wrong_lead = body.replace(f"({ne_all} on all {total})", f"({ne} on all {total})")
    problems = _n_eff_figure_problems(headline + wrong_lead, demo_output)
    assert len(problems) == 1 and problems[0].startswith("the first finding"), problems
    wrong_all_right = body.replace(f"would get {printed['expected'][0]}", "would get 9.99")
    problems = _n_eff_figure_problems(headline + wrong_all_right, demo_output)
    assert len(problems) == 1 and problems[0].startswith("the all-right sentence"), problems


# The judge is part of the data's description, not of the votes: one model judged the ten
# voters' answers, and it is the model of three of them (review of 2026-11-21).
_JUDGE_SENTENCE = re.compile(r"The judge is (\S+), the model of (\w+) of the (\w+) voters\.")


def _judge_problems(text: str, data: dict) -> list[str]:
    match = _JUDGE_SENTENCE.search(" ".join(text.split()))
    if not match:
        return ["no sentence of the form 'The judge is M, the model of K of the N voters.'"]
    judge, k, total = match.groups()
    same = sum(voter["model"] == data["judge"] for voter in data["panel"])
    expected = [("judge", judge, data["judge"]), ("voters with the judge's model", _WORD_NUMBERS.get(k.lower()), same),
                ("voters", _WORD_NUMBERS.get(total.lower()), len(data["panel"]))]
    return [f"{what}: text {said}, data {got}" for what, said, got in expected if said != got]


def _bundled_data() -> dict:
    import json

    import voteworth

    return json.loads((Path(voteworth.__file__).parent / "data" / "panel_10_voters.json").read_text())


def test_the_judge_sentence_matches_the_data(readme):
    problems = _judge_problems(readme, _bundled_data())
    assert not problems, "\n".join(problems)


def test_the_judge_check_can_fire():
    """Control arm, constructed: another judge and another count must both be reported."""
    planted = "The judge is claude-haiku-4-5, the model of two of the ten voters."
    assert len(_judge_problems(planted, _bundled_data())) == 2
    assert _judge_problems("no such sentence", _bundled_data())


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.S)


def _run_block(code: str) -> int:
    """Run one README code block statement by statement; return how many shown results it compared.

    A bare expression followed by a comment line that holds a literal list ('# [0.333, …]') is
    a shown result: it is compared with the value, to the decimals the comment shows.
    """
    import ast

    namespace: dict = {}
    lines = code.splitlines()
    compared = 0
    for node in ast.parse(code).body:
        if isinstance(node, ast.Expr):
            value = eval(compile(ast.Expression(node.value), "<README>", "eval"), namespace)
            after = lines[node.end_lineno] if node.end_lineno < len(lines) else ""
            literal = re.match(r"\s*#\s*(\[[^\]]*\])", after)
            if literal:
                shown = ast.literal_eval(literal.group(1))
                places = max(
                    (len(repr(x).split(".")[1]) for x in shown if isinstance(x, float)), default=0
                )
                got = [round(x, places) for x in value]
                assert got == shown, f"the README shows {shown}, the code returns {got}"
                compared += 1
        else:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<README>", "exec"), namespace)
    return compared


def test_readme_code_blocks_run_as_written(readme):
    """Every Python example in the README runs as printed, and its shown results hold."""
    blocks = _python_blocks(readme)
    assert len(blocks) >= 2, "the README no longer carries the usage examples this test runs"
    compared = sum(_run_block(block) for block in blocks)
    assert compared >= 1, "no shown result in the examples was compared; has the README changed shape?"


def test_the_code_block_runner_can_fire():
    """Control arm: the 0.1.3 example with its '...' placeholder must fail, and so must a wrong shown result."""
    with pytest.raises(TypeError):
        _run_block(
            "from voteworth import effective_votes\n"
            "history = [[0, 0, 1, 0], [1, 1, 0, 2], ...]\n"
            "effective_votes(history, n_voters=4)\n"
        )
    with pytest.raises(AssertionError):
        _run_block(
            "from voteworth import structural_weights\n"
            "structural_weights(['a', 'a', 'b'])\n"
            "# [0.4, 0.4, 1.0]\n"
        )
