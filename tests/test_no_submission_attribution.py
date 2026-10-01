# gate-fixture: begin — this file defines the prohibition, so it must quote what it forbids
"""Does any shipped surface tie this work to a venue submission?

Research code often travels alongside papers under blind review, where a public surface naming
the venue or the review status would break the authors' anonymity. That is not a state a
repository can detect, and it changes without warning — so this guard holds unconditionally:
**this package never announces a submission of its own**, whether or not one exists.

Stated that way on purpose. An earlier version of this docstring explained the rule by
describing the situation it was written for, which told a reader more about the authors than
the rule required. A guard against disclosure should not itself disclose — the fourth time in
this repository that a check turned out to sit inside its own subject.

The distinction it has to get right: **citing someone else's venue is normal scholarship;
announcing your own is the leak.** "Kim et al. (ICML 2025) measure exactly this" must pass.
"Currently under review at NeurIPS" must fail. A first draft of this check matched bare venue
names and flagged the citation — a guard that fires on correct behaviour gets switched off,
and then it protects nothing.

So the pattern requires a SELF-REFERENCE near the venue or status word: we/our/this
work/this package/currently/accepted/submitted. A venue name sitting in a bracket next to an
author and a year is left alone.
"""
# gate-fixture: end
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shipped import fixture_regions, gated_all, locate, read_flat_outside_fixtures

_ROOT = Path(__file__).resolve().parent.parent

_SELF = r"(?:we|our|this\s+(?:work|package|paper|library|tool|study)|the\s+authors?)"
_VENUE = r"(?:NeurIPS|ICML|ICLR|TMLR|COLM|ACL|EMNLP|AAAI|CVPR|workshop)"

SUBMISSION_ATTRIBUTION = re.compile(
    rf"""
      \b{_SELF}\s+(?:\w+\s+){{0,3}}(?:submitted|submission|under\s+review|in\s+press|forthcoming)\b
    | \b(?:submitted|under\s+review|accepted)\s+(?:at|to)\s+(?:\w+\s+){{0,2}}{_VENUE}\b
    | \bcurrently\s+under\s+review\b
    | \b{_SELF}\s+(?:is|are|was|were)\s+(?:\w+\s+){{0,2}}anonymou?s
    | \banonymou?s\s+for\s+(?:double[-\s]blind\s+)?(?:peer\s+)?review\b
    | \bdouble[-\s]blind\s+submission\b
    | \bcamera[-\s]ready\b
    | \brebuttal\b
    | \b{_SELF}\s+(?:\w+\s+){{0,3}}{_VENUE}\s+(?:submission|paper|entry)\b
    """,
    re.I | re.X,
)


# gate-fixture: begin — positive controls: every line below is a real attribution, on purpose
MUST_MATCH = [
    "Currently under review at NeurIPS 2026.",
    "This work is anonymous for double-blind review.",
    "Submitted to ICML.",
    "Our submission is in press.",
    "We submitted this to a workshop.",
    "anonymous for peer review",
    "the camera-ready version",
    "our rebuttal is posted",
]
# gate-fixture: end

MUST_NOT_MATCH = [
    "Kim et al. (ICML 2025) measure exactly this.",
    "Verga et al., Replacing Judges with Juries (arXiv:2404.18796)",
    "Wang et al. introduced self-consistency.",
    "Balasubramanian et al. (2026) use Ising couplings.",
    "published at Zenodo under CC BY 4.0",
    "a decision the method makes",
    "Kuncheva et al., Limits on the Majority Vote Accuracy (2003)",
    "Ding audits 265k samples",
]


@pytest.mark.parametrize("sentence", MUST_MATCH)
def test_detects_self_attribution(sentence):
    """Positive control — a guard that cannot fire proves nothing by staying silent."""
    assert SUBMISSION_ATTRIBUTION.search(sentence), f"missed a real leak: {sentence!r}"


@pytest.mark.parametrize("sentence", MUST_NOT_MATCH)
def test_leaves_third_party_citations_alone(sentence):
    """Negative control — citing another author's venue is scholarship, not a leak."""
    found = SUBMISSION_ATTRIBUTION.findall(sentence)
    assert not found, f"false alarm {found} on a citation: {sentence!r}"


def test_no_shipped_file_attributes_a_submission():
    offenders = []
    for name in gated_all():
        path = _ROOT / name
        if not path.exists():
            continue
        # Flattened, not line by line: a forbidden phrase that wraps must still be caught.
        # Outside declared regions: this file's positive controls are attributions by design.
        for match in SUBMISSION_ATTRIBUTION.finditer(read_flat_outside_fixtures(path)):
            offenders.append(f"{name}:{locate(path, match.group(0))}: {match.group(0)!r}")
    # gate-fixture: begin — the failure message names the thing it reports
    assert not offenders, (
        "a shipped surface ties this work to a submission:\n" + "\n".join(offenders)
    )
    # gate-fixture: end


def test_the_guard_actually_opened_the_files():
    """Control arm for the walk: zero offenders must mean the files were read, not missed."""
    read = [n for n in gated_all() if (_ROOT / n).exists() and (_ROOT / n).read_text().strip()]
    assert len(read) == len(gated_all()), f"missing: {sorted(set(gated_all()) - set(read))}"


def test_declared_regions_in_this_file_all_fire():
    """Control arm for the exemptions: a region that exempts nothing is a blanket.

    Scoped to this file, for the same reason the novelty gate scopes its own: a region is an
    exemption from one prohibition, and the regions in the other gate hold novelty claims,
    which this pattern has no opinion about.
    """
    regions = fixture_regions(Path(__file__))
    assert len(regions) >= 3, (
        f"only {len(regions)} declared regions in this file — the marker scan is not seeing "
        "them, and the gate above would be scanning text it believes is exempt"
    )
    for region in regions:
        assert SUBMISSION_ATTRIBUTION.search(region), (
            "a declared region in this file contains nothing this gate forbids — it exempts "
            f"nothing and should be deleted:\n  {region[:160]!r}"
        )


def test_a_violation_outside_a_region_is_still_caught(tmp_path):
    """The regions must not blind the scan to the file around them."""
    from shipped import FIXTURE_BEGIN, FIXTURE_END

    inside_text, outside_text = MUST_MATCH[0], MUST_MATCH[1]
    probe = tmp_path / "probe.py"
    probe.write_text(f"{FIXTURE_BEGIN}\n{inside_text}\n{FIXTURE_END}\n{outside_text}\n")

    regions = fixture_regions(probe)
    assert regions and SUBMISSION_ATTRIBUTION.search(regions[0]), "the probe's region is not real"
    outside = read_flat_outside_fixtures(probe)
    assert SUBMISSION_ATTRIBUTION.search(outside), (
        "a violation one line outside a declared region was not seen — the region is "
        "swallowing more than it marks"
    )
    assert not SUBMISSION_ATTRIBUTION.search(outside.replace(outside_text, "")), (
        "the region's own content leaked into the scanned prose"
    )
