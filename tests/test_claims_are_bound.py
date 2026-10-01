"""Is every sweeping sentence in the README bound to something?

A number gate checks numbers. It does not check **conclusions** — and conclusions are where
the expensive errors live, because they summarise a table without repeating any value from it.
"the only", "no other", "never", "every": none of these has a field to compare against, so
none of them appears on a checklist built around values.

The failure this guards against was measured by a sibling session on a different artefact: a
sentence reading "no system holds the same position under all three" survived **eight** review
rounds and a 5,668-check verifier, while the table three lines above it showed a system that
did. The verifier said so itself in its own header — *a claim that is not listed is not checked.*

The construction here follows the same session's second finding: **the declaration drives the
loop, not a typed list.** Sentences are discovered in the README; each discovered one must
appear in `CLAIMS` either bound to a test that asserts it, or marked unbindable with a reason.
A new sweeping sentence with no entry fails this file — the omission has to be stated, not
merely happen.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_TESTS = _ROOT / "tests"

# Constructions that summarise rather than report: universals, absences, exclusivity.
# Two shapes: universals/exclusivity by keyword, and absence claims of the form
# "no <thing> <verb>". The second was missing from the first draft of this file — and the
# sentence that motivated the whole guard, "no system holds the same position under all
# three", is exactly that shape. A pattern built from the examples one already has in front of
# oneself catches those examples. The exclusions after (?!...) are the ordinary uses that
# would otherwise flood it: "no network", "no longer", "no need for labels".
_SWEEPING = re.compile(
    r"\b(?:the only|only one|no other|none of|nobody|nothing|never|always|every|all of them)\b"
    r"|\bno\s+(?!longer|need|network|API|reason|hint|interval)\w+\s+"
    r"(?:holds|hold|is|are|was|were|does|do|did|has|have|can|could|will|would"
    r"|measures|asks|offers|exists|tests)\b",
    re.I,
)

# Each key is a distinctive fragment of a sweeping sentence in the README. The value is either
# the name of a test that asserts it, or a reason it cannot be bound to one.
#
# "bound" means: if the code stopped behaving this way, the named test would go red. Not
# "a test mentions the topic" — the assertion has to carry the sentence.
CLAIMS: dict[str, str] = {
    # --- bound to a test -------------------------------------------------------
    "cannot be repaired by reweighting": "test:test_same_model_panel_is_inert",
    "does nothing at all": "test:test_same_model_panel_is_inert",
    "every voter carries the same weight": "test:test_same_model_panel_is_inert",
    "An empty history yields uniform weights": "test:test_no_history_no_penalty",
    "never deviates from it, so it never registers": (
        "test:test_dominant_bloc_is_invisible_to_the_weights"
    ),
    "For every pair of voters": "test:test_independence_weights_downweight_correlated_bloc",
    # --- unbindable, with the reason -------------------------------------------
    "A passing suite proves nothing on its own": (
        "not a claim about this code — it is the reason the mutation gate exists, and that "
        "gate is itself tested in test_mutation.py"
    ),
    "is the reference for panels of disjoint model families": (
        "a statement about someone else's paper (PoLL); nothing in this repository can "
        "verify it, and it is attributed with an arXiv id so a reader can check"
    ),
    "it beats every weighting scheme including this one": (
        "a concession about deterministic checks (GroundEval), deliberately stronger than "
        "anything we could prove — a test that upheld it would be pretending to measure "
        "someone else's method"
    ),
    "None of those three asks whether the votes": (
        "an absence claim, already qualified in the sentence itself as read-at-source in "
        "September 2026 with the sweep's own gap disclosed; no test can establish an absence "
        "in the world"
    ),
    "Treat every": (
        "an instruction to the reader, not an assertion about behaviour"
    ),
    "Everything below runs from the source": (
        "test:test_every_file_in_the_tree_is_covered_by_a_packaging_rule"
    ),
    "nothing observed, nothing claimed": "test:test_no_history_no_penalty",
    "A voter that never co-deviates keeps weight 1.0": (
        "test:test_independence_weights_downweight_correlated_bloc"
    ),
    "Asked as a downloader would": (
        "frames the section that separates verifiable from trusted; the individual rows "
        "below it are each bound by the command they name, and 'nothing but the files in "
        "this package' is the condition under which that section was measured"
    ),
    "The three DOIs resolve to what we say they do": (
        "listed under what a reader must take on trust — the sentence exists precisely to "
        "say it is unverifiable from this package; binding it to a test would contradict it"
    ),
    '"no tool does X" here, including this one': (
        "an instruction about how to read every absence claim in this file, including its "
        "own; binding it to a test would be a category error"
    ),
}


def _prose_paragraphs() -> list[tuple[int, str]]:
    """Prose paragraphs, joined across line breaks.

    Reading line by line splits sentences at the wrap and makes a claim spanning two lines
    invisible to both this guard and the staleness check below — the first version of this
    file had exactly that bug. Tables, code and block quotes are excluded: a table row
    reports rather than summarises, and a quote is someone else's voice.
    """
    out: list[tuple[int, str]] = []
    in_code = False
    start, buf = 0, []

    def flush():
        if buf:
            out.append((start, " ".join(buf)))
        buf.clear()

    for i, raw in enumerate(_README.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            flush()
            continue
        if in_code or not line or line.startswith(("|", "#", ">")):
            flush()
            continue
        if not buf:
            start = i
        buf.append(line)
    flush()
    return [(n, t) for n, t in out if _SWEEPING.search(t)]


def _all_test_names() -> set[str]:
    names = set()
    for path in _TESTS.glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    return names


def test_every_sweeping_sentence_is_declared():
    """The guard. A summarising sentence with no entry in CLAIMS is an unchecked conclusion."""
    undeclared = []
    for lineno, line in _prose_paragraphs():
        if not any(fragment in line for fragment in CLAIMS):
            undeclared.append(f"README.md:{lineno}: {line[:100]}")
    assert not undeclared, (
        "these sentences summarise without being declared in CLAIMS:\n  "
        + "\n  ".join(undeclared)
        + "\n\nBind each to a test that would fail if the behaviour changed, or record why "
        "it cannot be bound. An unchecked conclusion is how a wrong sentence survives review."
    )


def test_every_binding_names_a_test_that_exists():
    """Control arm for the declaration: a binding to a deleted or renamed test is worse than
    no binding, because it reads as coverage."""
    names = _all_test_names()
    assert names, "no tests discovered — this control arm is broken"
    for fragment, binding in CLAIMS.items():
        if binding.startswith("test:"):
            target = binding.split(":", 1)[1]
            assert target in names, (
                f"claim {fragment!r} is bound to {target!r}, which does not exist. "
                "Either the test was renamed or the binding was never real."
            )


def test_unbindable_entries_carry_a_real_reason():
    """The other half: 'cannot be tested' must not become the easy exit."""
    for fragment, binding in CLAIMS.items():
        if not binding.startswith("test:"):
            assert len(binding) > 40, (
                f"{fragment!r} is declared unbindable with a reason too thin to audit: "
                f"{binding!r}"
            )


def test_every_declared_claim_still_appears_in_the_readme():
    """Dead entries are how a declaration turns into decoration: it grows, nothing prunes it,
    and eventually it describes a document that no longer exists."""
    # Same normalisation as the guard above: a fragment spanning a line wrap must still be
    # findable, or pruning would delete live entries.
    readme = " ".join(_README.read_text().split())
    stale = [f for f in CLAIMS if f not in readme]
    assert not stale, (
        f"declared but no longer in the README: {stale} — delete them, or the declaration "
        "stops describing the document it guards."
    )


@pytest.mark.parametrize(
    "sentence,should_flag",
    [
        # gate-fixture: begin — positive and negative controls for the sweeping-claim pattern
        # the sibling session's actual failing sentence — the reason this file exists
        ("no system holds the same position under all three", True),
        ("no package measures this", True),
        ("no other work asks whether", True),
        ("this is the only tool that does it", True),
        ("the weights never change", True),
        ("every judge is independent", True),
        ("the panel scored 44.8% on held-out items", False),
        ("weights are learned from a vote history", False),
        ("see the boundary section below", False),
        ("no network, no API key", False),
        ("no longer supported", False),
        ("no need for labels", False),
        # gate-fixture: end
    ],
)
def test_sweeping_pattern_separates_summary_from_report(sentence, should_flag):
    """Both control arms. The pattern must catch the sibling session's actual failing sentence
    and stay quiet on ordinary reporting — otherwise it either floods or sleeps."""
    assert bool(_SWEEPING.search(sentence)) == should_flag, f"misjudged: {sentence!r}"
