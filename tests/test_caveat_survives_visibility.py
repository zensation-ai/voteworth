"""Does the caveat survive onto the surfaces people actually read?

A limitation written only where the careful reader gets to is not a limitation, it is a
footnote. The failure mode has a shape: **the qualification thins out as visibility rises.**
The long-form text carries the full caveat, the summary carries less, and the one line that
appears in a search result — the line most readers will ever see — carries none.

Measured on this package before this test existed: the README devoted a whole section to the
dominant-bloc blind spot, `CITATION.cff` stated it in the abstract, and the `description` in
`pyproject.toml` — the line PyPI and GitHub show in search results — promised "correct the
majority vote for correlated blocs" with no hint that the correction has a regime where it
inverts. Exactly backwards from the order a reader meets them in.

So this file checks the caveat top-down, starting with the most visible surface. It is the
inverse of the usual reading order on purpose.

Credit where due: the failure class was identified by a sibling session auditing a different
artefact, which found the same gradient there — full qualification in body text, less in a
table cell, none in the record's public description.
"""
from __future__ import annotations

import re
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.9 / 3.10 — the package itself needs no TOML parser, only
    import tomli as tomllib  # this gate does, so it is a dev dependency, not a runtime one
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

# Words that concede a limit. Deliberately a class, not a fixed phrase: a guard that looks for
# last week's wording finds last week's wording.
_CAVEAT = re.compile(
    r"\bcannot\b|\bcan not\b|\bdoes not\b|\bwhere it (?:cannot|fails|inverts)\b"
    r"|\bblind\b|\bboundary\b|\blimit(?:s|ation)?\b|\binert\b|\bnot a benchmark\b"
    r"|\bonly when\b|\bexcept\b|\bfails?\b",
    re.I,
)


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())


def test_the_most_visible_line_carries_a_caveat():
    """The `description` is what a search result shows. If any surface must concede the limit,
    it is this one — and it is the one most likely to be written as pure promise."""
    description = _pyproject()["project"]["description"]
    assert _CAVEAT.search(description), (
        "the package description promises capability without conceding any limit:\n"
        f"  {description}\n"
        "This is the line PyPI shows in search results. A caveat that lives only in the README "
        "is a caveat the reader meets last, if at all."
    )


def test_citation_abstract_carries_a_caveat():
    """The abstract is what a citation index shows."""
    text = (_ROOT / "CITATION.cff").read_text()
    match = re.search(r"abstract: >\n(.*?)(?=^\w)", text, re.S | re.M)
    assert match, "CITATION.cff has no abstract block"
    assert _CAVEAT.search(match.group(1)), "the citation abstract concedes no limit"


def test_readme_states_the_boundary_before_the_usage_section():
    """A reader who copies the usage example and stops must already have passed the limit.

    Not merely "the README mentions it somewhere" — it has to come early enough that skimming
    to the code block does not skip it.
    """
    readme = (_ROOT / "README.md").read_text()
    lines = readme.splitlines()

    def first_line_matching(pattern: str) -> int:
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.I):
                return i
        return -1

    first_caveat = next(
        (i for i, line in enumerate(lines) if _CAVEAT.search(line) and line.strip()), -1
    )
    usage = first_line_matching(r"^## Using it")

    assert first_caveat >= 0, "the README concedes no limit at all"
    assert usage >= 0, "the usage section heading moved — update this test deliberately"
    assert first_caveat < usage, (
        f"the first caveat appears at line {first_caveat + 1}, after the usage section at line "
        f"{usage + 1}. A reader who skims to the code never meets it."
    )


def test_the_boundary_has_its_own_section():
    """The dominant-bloc case is the one failure a user cannot discover from a green run —
    the weights look plausible, the vote just does not change. It gets a heading, not a line."""
    readme = (_ROOT / "README.md").read_text()
    headings = [l for l in readme.splitlines() if l.startswith("#")]
    assert any(re.search(r"boundary|blind|limit", h, re.I) for h in headings), (
        "no heading names the boundary; it has been demoted into prose:\n  "
        + "\n  ".join(headings)
    )


@pytest.mark.parametrize(
    "sentence,should_match",
    [
        # real concessions — must fire
        ("corrects the vote, including the panel shape where it cannot", True),
        ("reports when it cannot", True),
        ("this method is blind when one bloc holds the plurality", True),
        ("not a benchmark and ranks no models", True),
        ("it does not make a bad panel good", True),
        ("the boundary: one bloc holding the plurality", True),
        # plain capability claims — must stay silent
        ("corrects the majority vote for correlated voters", False),
        ("the fastest way to aggregate LLM judges", False),
        ("measure error correlation and weight your panel", False),
        ("closed-form, label-free, numpy only", False),
    ],
)
def test_caveat_pattern_separates_concession_from_promise(sentence, should_match):
    """Both control arms in one. The pattern must fire on a real concession and stay silent on
    a plain capability claim — otherwise the checks above would pass on marketing copy, and a
    green result would mean nothing."""
    assert bool(_CAVEAT.search(sentence)) == should_match, (
        f"pattern misjudged {sentence!r}: expected match={should_match}"
    )


# Rules the README imposes on *this package* — as opposed to instructions to the reader.
# Empty today, and that is the only reason the failure below has never occurred here.
#
# A sibling session measured the failure on their own artefact: their README states "a record
# that says X should say where it is", their long-form text honours it, and the landing-page
# description — the surface read first — states the same claim without the pointer. True, and
# still the weaker disclosure on the more-read surface.
#
# Their conclusion, which this guard implements: *a self-imposed rule does not automatically
# extend to the package description.* Whoever writes one has to add it here, or it holds only
# where it happens to be written.
#
# Each entry: the rule's distinctive fragment → the surfaces it must hold on.
SELF_IMPOSED_RULES: dict[str, tuple[str, ...]] = {}

_SURFACES = {
    "description": lambda: _pyproject()["project"]["description"],
    "readme": lambda: (_ROOT / "README.md").read_text(),
    "citation": lambda: (_ROOT / "CITATION.cff").read_text(),
}

# Normative constructions that bind the package rather than instruct the reader.
_NORMATIVE = re.compile(
    r"\b(?:this (?:package|file|record|README)|we|our)\b[^.]{0,80}?"
    r"\b(?:should|must|has to|is required to|always|never)\b",
    re.I,
)


def test_any_self_imposed_rule_is_declared():
    """If the README starts obliging this package to do something, that obligation has to be
    listed — otherwise it silently applies to one surface only.

    This is the preventive half. The sibling session's version of this guard was written after
    the failure; ours is written before, which is the whole difference between the two."""
    readme = (_ROOT / "README.md").read_text()
    undeclared = []
    in_code = False
    for lineno, raw in enumerate(readme.splitlines(), 1):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith(("|", "#")):
            continue
        if _NORMATIVE.search(line) and not any(f in line for f in SELF_IMPOSED_RULES):
            undeclared.append(f"README.md:{lineno}: {line[:95]}")
    assert not undeclared, (
        "the README imposes a rule on this package that is not in SELF_IMPOSED_RULES:\n  "
        + "\n  ".join(undeclared)
        + "\nAdd it with the surfaces it must hold on, or a rule written once will be "
        "honoured once."
    )


def test_declared_rules_hold_on_every_surface_they_name():
    """The other half: a declared rule must actually be present wherever it claims to apply."""
    for fragment, surfaces in SELF_IMPOSED_RULES.items():
        for surface in surfaces:
            assert surface in _SURFACES, f"unknown surface {surface!r} for rule {fragment!r}"
            text = _SURFACES[surface]()
            assert fragment in text, (
                f"rule {fragment!r} claims to hold on {surface!r} but is absent there"
            )


def test_the_normative_pattern_can_fire():
    """Control arm. With SELF_IMPOSED_RULES empty, the guard above can only ever pass — so the
    pattern is proven against sentences of both kinds instead."""
    binds_us = [
        "This package should always name the record it draws from.",
        "We never report a figure without its base.",
        "This README must state the boundary before the usage section.",
    ]
    instructs_reader = [
        "Check bloc_share before you trust the weights.",
        "Match all three choices before comparing your panel.",
        "Read them; they are open access.",
    ]
    for s in binds_us:
        assert _NORMATIVE.search(s), f"missed a self-imposed rule: {s!r}"
    for s in instructs_reader:
        assert not _NORMATIVE.search(s), f"false alarm on reader instruction: {s!r}"
