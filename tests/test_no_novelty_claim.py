# gate-fixture: begin — this file defines the prohibition, so it must quote what it forbids
"""Does anything shipped here claim to be new?

This package's prior-art sweep found no work occupying its mechanism — but that null result
was gathered as *internal assurance*, not as evidence for a published assertion. The moment a
shipped surface says "first", "novel" or "no existing tool", that null starts carrying a public
claim, and a null carries a claim only if its coverage was measured. Ours was not measured to
that standard: ten search modalities named by the sweep's own completeness critic were never
run.

So the rule is simple and enforced here rather than remembered: **this package describes what
it does and cites what exists; it does not claim to be first.** If someone later wants to make
that claim, this test fails, and the honest response is to widen the sweep — not to delete the
test.

The pattern matches CONSTRUCTIONS, not words. "the first argument" is not a novelty claim;
"the first package" is. An earlier draft matched bare words and produced three false alarms on
ordinary prose — and a check that cries wolf teaches people to look away.
"""
# gate-fixture: end
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shipped import (
    EXCLUDED,
    FIXTURE_BEGIN,
    FIXTURE_END,
    discover_all,
    fixture_regions,
    gated_all,
    locate,
    read_flat_outside_fixtures,
)

_ROOT = Path(__file__).resolve().parent.parent

# Nouns a novelty claim would attach to. "first version of this file" is prose; "first
# package" is a claim — the difference is the noun.
_THING = (
    r"(?:package|library|tool|method|approach|technique|algorithm|work|implementation|"
    r"framework|solution|software|weight|estimator|measure|metric|correction|mechanism)"
)

# [\w-] rather than \w: technical prose is full of hyphenated modifiers ("a novel closed-form
# weight"), and a pattern blind to hyphens would miss exactly the phrasing this field uses.
NOVELTY_CLAIM = re.compile(
    rf"""
      \b(?:the\s+)?first\s+(?:[\w-]+\s+){{0,2}}{_THING}s?\b
    | \bfirst\s+to\s+[\w-]+
    | \bnovel\s+(?:[\w-]+\s+){{0,2}}{_THING}s?\b
    | \bno\s+(?:other|existing|known|published|prior|comparable)\s+(?:[\w-]+\s+){{0,2}}{_THING}s?\b
    | \bthe\s+only\s+(?:[\w-]+\s+){{0,2}}{_THING}s?\b
    | \bnothing\s+else\s+[\w-]+
    | \bunprecedented\b | \bbreakthrough\b | \bstate[-\s]of[-\s]the[-\s]art\b
    | \bwe\s+are\s+the\s+first\b | \bnever\s+before\b
    | \bunique(?:ly)?\s+(?:[\w-]+\s+){{0,2}}{_THING}s?\b
    """,
    re.I | re.X,
)


# gate-fixture: begin — positive controls: every line below is a real novelty claim, on purpose
# Sentences that MUST trip the check. Without these the pattern could rot into one that
# matches nothing and still reports a clean surface.
MUST_MATCH = [
    "This is the first package to measure judge independence.",
    "No existing tool does this.",
    "A novel closed-form weight.",
    "We report the only label-free correction.",
    "The first known method for correlated voters.",
    "No comparable library exists.",
    "An unprecedented approach.",
    "We are the first to report this.",
    "A uniquely simple estimator.",
    "state-of-the-art results",
]
# gate-fixture: end

# Ordinary prose that must NOT trip it — every one of these appears, or could appear, in the
# shipped text.
MUST_NOT_MATCH = [
    "The first argument is the history.",
    "numpy is the only dependency.",
    "Read the new README section.",
    "Each answer joins the first cluster whose centroid it matches.",
    "this package only does the first.",
    "check this FIRST — see the boundary below",
    "turn them into ids first",
    "a representative string per cluster (its first member)",
    "starts a new one",
    "the first version of this file matched only percentages",
]


@pytest.mark.parametrize("sentence", MUST_MATCH)
def test_pattern_detects_a_real_novelty_claim(sentence):
    """Positive control. A guard that cannot fire proves nothing when it stays silent."""
    assert NOVELTY_CLAIM.search(sentence), f"missed a real novelty claim: {sentence!r}"


@pytest.mark.parametrize("sentence", MUST_NOT_MATCH)
def test_pattern_ignores_ordinary_prose(sentence):
    """Negative control. A check with false alarms trains people to look away."""
    found = NOVELTY_CLAIM.findall(sentence)
    assert not found, f"false alarm {found} on ordinary prose: {sentence!r}"


def test_no_shipped_file_claims_novelty():
    """The guard itself."""
    offenders = []
    for name in gated_all():
        path = _ROOT / name
        if not path.exists():
            continue
        # Flattened, not line by line: a forbidden phrase that wraps must still be caught.
        # Outside declared regions: this suite's own positive controls are novelty claims by
        # design, and the files that define the rule have to quote it.
        for match in NOVELTY_CLAIM.finditer(read_flat_outside_fixtures(path)):
            offenders.append(f"{name}:{locate(path, match.group(0))}: {match.group(0)!r}")
    assert not offenders, (
        "a shipped surface claims novelty, which the prior-art sweep was not scoped to "
        "support:\n" + "\n".join(offenders)
    )


def test_the_guard_reads_the_files_it_claims_to():
    """Control arm for the file walk: a clean result must mean the files were opened.

    Without this, a renamed file or a wrong root would produce zero offenders and look
    identical to a clean repository.
    """
    expected = gated_all()
    read = [n for n in expected if (_ROOT / n).exists() and (_ROOT / n).read_text().strip()]
    assert len(read) == len(expected), (
        f"only {len(read)} of {len(expected)} shipped files were found — the guard above "
        f"scanned less than it claims: missing {sorted(set(expected) - set(read))}"
    )

    # A count, not a single marker. The previous version asserted that *some* file contained
    # "INERT" — a word that lives in exactly one of them. That proved the walk reached one
    # file and claimed it for all. A sibling session put it precisely: a liveness check built
    # as a single probe, or as the difference of two sums, reports confidently about a
    # population it never touched. So: every file must yield readable content, counted.
    non_empty = sum(1 for n in expected if len((_ROOT / n).read_text().strip()) > 10)
    assert non_empty == len(expected), (
        f"{len(expected) - non_empty} of {len(expected)} shipped files read as empty — the "
        "scan is not seeing what it reports on"
    )


def test_every_shipped_file_is_either_gated_or_excluded_with_a_reason():
    """The list-drift guard. A file that ships must be walked by the prose gates or carry a
    stated reason for not being — silence is not a decision.

    This replaced two hand-typed lists on 2026-09-20, after a measurement found `LICENSE`
    shipped and covered by neither.
    """
    unaccounted = [f for f in discover_all() if f not in gated_all() and f not in EXCLUDED]
    assert not unaccounted, (
        "these files ship but are neither gated nor excluded with a reason:\n  "
        + "\n  ".join(unaccounted)
        + "\nAdd them to the gates, or to EXCLUDED in tests/shipped.py with why."
    )
    everything = set(discover_all())
    for name, reason in EXCLUDED.items():
        assert (__import__("shipped").ROOT / name).exists(), (
            f"{name} is excluded but does not exist — the exclusion is stale"
        )
        # Reachability, not just existence. Until 2026-09-21 this walk used `discover()`,
        # which filters by suffix; both entries here have suffixes it drops, so neither could
        # ever have been "unaccounted for" and neither exclusion was doing any work. An
        # exemption that cannot fire is indistinguishable from one that is not needed.
        assert name in everything, (
            f"{name} is excluded but the walk never reaches it — the exclusion exempts "
            "nothing and the rule it belongs to is not being applied"
        )
        assert len(reason) > 25, f"{name} is excluded without a usable reason: {reason!r}"


def test_declared_regions_in_this_file_all_fire():
    """Control arm for the exemptions: a region that exempts nothing is a blanket.

    Scoped to THIS file, because a region is an exemption from a specific prohibition: the
    regions in the submission gate hold submission attributions, which this pattern has no
    opinion about. Each gate vouches for its own. `test_no_stray_fixture_markers` in
    test_gates_assert.py holds the other end — no file may carry a marker that no gate checks.
    """
    regions = fixture_regions(Path(__file__))
    assert len(regions) >= 2, (
        f"only {len(regions)} declared regions in this file — the marker scan is not seeing "
        "them, and the gate above would be scanning text it believes is exempt"
    )
    for region in regions:
        assert NOVELTY_CLAIM.search(region), (
            "a declared region in this file contains nothing this gate forbids — it exempts "
            f"nothing and should be deleted:\n  {region[:160]!r}"
        )


def test_a_violation_outside_a_region_is_still_caught(tmp_path):
    """The other half: the regions must not blind the scan to the file around them."""
    probe = tmp_path / "probe.py"
    # The probe reuses the declared positive controls instead of restating them. Two reasons,
    # both measured while this test was written: a marker written out as a literal is read as
    # a real marker by the scan of this very file, and a novelty claim written out as a
    # literal is a novelty claim on an unmarked line. Borrowing from MUST_MATCH — already
    # inside a declared region — leaves nothing new on the page.
    claim_inside, claim_outside = MUST_MATCH[0], MUST_MATCH[1]
    probe.write_text(
        f"{FIXTURE_BEGIN}\n{claim_inside}\n{FIXTURE_END}\n{claim_outside}\n"
    )
    inside = fixture_regions(probe)
    assert inside and NOVELTY_CLAIM.search(inside[0]), "the probe's region is not a real one"
    outside = read_flat_outside_fixtures(probe)
    assert NOVELTY_CLAIM.search(outside), (
        "a violation one line outside a declared region was not seen — the region is "
        "swallowing more than it marks"
    )
    assert not NOVELTY_CLAIM.search(
        read_flat_outside_fixtures(probe).replace(claim_outside, "")
    ), "the region's own content leaked into the scanned prose"
