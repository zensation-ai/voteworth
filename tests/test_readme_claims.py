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


# Percentages the README quotes from OTHER people's measurements or from the synthetic
# boundary experiment — they cannot appear in a live demo run, and each is listed here with
# where it comes from. Anything in the prose that is NOT on this list is a claim about our own
# run and must be reproducible from it.
#
# A README that cites the literature needs this list. Without it the test would either pass
# vacuously (drop the prose check) or force us to delete honest citations — and the second
# failure mode is worse than the first.
FOREIGN_FIGURES = {
    # other people's measurements
    "2.2": "Kohli 2026 (arXiv:2605.29800): nine judges over seven families, effective votes",
    "0.05": "lower end of published absolute same-family gaps, Kohli 2026",
    "0.11": "upper end of that band, measured on this panel's raw-agreement base",
    "0.42": "Kim et al. ICML 2025 (arXiv:2506.07962): cross-provider agreement when both wrong",
    "0.6": "Kim et al. 2025: upper end of that cross-provider range (quoted as 0.60)",
    "0.5": "the weight a voter locked to one partner receives; asserted in test_aggregate.py",
    # The synthetic boundary experiment. 'Reproduced by tests/test_aggregate.py' used to
    # mean the file asserted the series FALLS; the values themselves were unchecked, and
    # two of the four were wrong. test_boundary_figures_match_the_readme now parses them
    # out of the README and recomputes each one.
    "1.91": "n_eff at a 3-of-10 bloc, synthetic, mean over five seeds",
    "1.68": "n_eff at a 5-of-10 bloc, synthetic, mean over five seeds",
    "1.55": "n_eff at a 6-of-10 bloc, synthetic, mean over five seeds",
    "1.26": "n_eff at an 8-of-10 bloc, synthetic, mean over five seeds",
    "0.43": "mean independent-voter weight in the boundary experiment, twenty seeds",
    "1": "bloc weight 1.00 in the boundary experiment, unchanged across twenty seeds",
    # the cited note, measured on all 57 items rather than the held-out half
    "15.3": "the cited note (10.5281/zenodo.21775275): overconfidence on all 57 items",
    "14.9": "the cited note: provenance-weighted overconfidence on all 57 items",
    "0.333": "1/3, the structural weight of a three-voter bloc in the usage example; "
             "asserted in test_structural_weights_downweight_shared_source",
    # arithmetic on printed figures, not printed themselves
    "13.8": "58.6 minus 44.8, derived from the run rather than printed by it",
    "2.3": "42.8 divided by 18.5, derived from the run",
}


def test_readme_prose_numbers_are_produced(readme, demo_output):
    """Every number in the prose is either reproducible from a live run or declared foreign.

    This is the assertion that stops the README drifting from the code. It does not forbid
    citing other people's figures — it forbids citing them *indistinguishably* from ours.
    """
    produced = _decimals(demo_output) | _percentages(demo_output)
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
