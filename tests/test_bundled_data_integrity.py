"""Is the bundled dataset the published one — and can a reader check that themselves?

A package that ships someone's published data, and then reports numbers derived from it, is
asking to be trusted twice: once that the code is right, once that the data was not altered.
The second is avoidable, and until 2026-09-20 this package did not avoid it — the file was
bundled with no checksum anywhere, so a reader had no way to tell an intact copy from an
edited one.

The finding came from a sibling session's "download lens", which asks three questions of a
published record: does it run, what can a downloader verify, and what must they take on trust.
Their answer for their own artefact was that 44 of 44 source files were absent and no hash was
recomputable. Ours was smaller and of the same kind.

Their follow-on point is why this file also checks the NOTICE text: *a paragraph that explains
an absence should name the alternative.* A checksum with nothing to compare it against moves
the trust rather than removing it, so the NOTICE has to say where the independent copy lives.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "src" / "voteworth" / "data" / "panel_10_voters.json"
_NOTICE = _ROOT / "NOTICE"


def test_notice_records_the_datasets_checksum():
    """The hash in NOTICE must be this file's hash. A stale checksum is worse than none: it
    reads as verification while certifying a file that no longer exists."""
    actual = hashlib.sha256(_DATA.read_bytes()).hexdigest()
    declared = re.findall(r"\b([0-9a-f]{64})\b", _NOTICE.read_text())
    assert declared, "NOTICE carries no SHA-256 for the bundled dataset"
    assert actual in declared, (
        f"the dataset's hash is {actual}, but NOTICE declares {declared}. "
        "Either the file changed or the checksum was never recomputed."
    )


def test_notice_names_where_to_compare_the_checksum_against():
    """A checksum alone relocates the trust — the reader still has to believe this package's
    own number. The NOTICE must point at the independent source."""
    text = _NOTICE.read_text()
    assert "Zenodo" in text and "10.5281/zenodo.21773065" in text, (
        "NOTICE gives a checksum but does not say what to compare it with"
    )
    assert re.search(r"\bdiffer\b|\baltered\b", text), (
        "NOTICE does not say what a mismatch means for the reader"
    )


def test_notice_describes_a_comparison_that_can_actually_be_made():
    """Pointing at a source is not enough if the pointer does not survive contact with it.

    An acceptance pass followed the old wording — "compare it against the checksum Zenodo
    publishes for that file on the record page" — and could not: Zenodo shows MD5 there, and
    the file carries a different name on the record than in this package. The instruction was
    true about the existence of a checksum and wrong about every step of reaching it, which is
    the shape a half-quoted reference takes. Both facts must now be named.
    """
    text = _NOTICE.read_text()
    assert "core2_writers_n10_100K.json" in text, (
        "NOTICE does not name the file as it is called on the record — a reader comparing by "
        "filename will not find it"
    )
    assert "MD5" in text, (
        "NOTICE does not warn that the record page shows a different hash algorithm"
    )
    assert "SHA256.txt" in text, (
        "NOTICE does not name the file on the record that actually carries a SHA-256"
    )


def test_the_check_can_fail(tmp_path):
    """Control arm: point the comparison at an altered copy and require a mismatch. Without
    this, a hash check that silently compared a file to itself would pass forever."""
    altered = tmp_path / "altered.json"
    altered.write_bytes(_DATA.read_bytes() + b"\n")
    declared = re.findall(r"\b([0-9a-f]{64})\b", _NOTICE.read_text())
    assert hashlib.sha256(altered.read_bytes()).hexdigest() not in declared, (
        "an altered copy matched the declared checksum — the comparison is not discriminating"
    )


def test_the_package_never_reads_the_records_own_summary_fields():
    """The bundled record carries a `quorum` block, and one of its fields disagrees with the
    others. The package must keep recomputing rather than quoting it.

    Measured 2026-09-21: `quorum.testN` is 40, while `quorum.bestSingleTestPct` is
    58.62068965517241 — exactly 17/29, and 29 is the number of held-out items this data
    actually yields (57 answerable, split by parity). The percentage is right; the count
    beside it is not.

    This file is the published record, byte-identical to Zenodo and pinned by the hash above.
    Editing it to remove the inconsistency would break that identity and misrepresent someone
    else's record, so the inconsistency stays and the guard goes here instead: nothing under
    `src/` may read `quorum`. A number quoted from a block with a wrong field in it is a
    number nobody recomputed.
    """
    import json

    for path in sorted((_ROOT / "src").rglob("*.py")):
        assert "quorum" not in path.read_text(), (
            f"{path.relative_to(_ROOT)} reads the record's own summary block; recompute "
            "instead — one of its fields is internally inconsistent"
        )

    # And the discrepancy is where it was measured, not somewhere else: if a future version of
    # the record fixes `testN`, this assertion fails and the docstring above needs rewriting
    # rather than quietly becoming false.
    quorum = json.loads(_DATA.read_text())["quorum"]
    assert quorum["testN"] == 40, "the record's testN changed — re-measure the note above"
    assert abs(quorum["bestSingleTestPct"] - 17 / 29 * 100) < 1e-9, (
        "the record's bestSingleTestPct is no longer 17/29 — re-measure the note above"
    )


def test_the_records_percentages_agree_with_a_live_run():
    """The half of the record that does hold: its accuracy figures match what we compute.

    This is what bounds the finding above. The count field is wrong; the percentages beside it
    are not, and the demo reproduces them from the votes. Without this, "the record has an
    inconsistent field" would be an unbounded doubt about the whole file.
    """
    import json

    d = json.loads(_DATA.read_text())
    votes, truth = d["votesByItem"], d["correctGroupByItem"]
    n = len(d["panel"])
    answerable = [t for t in range(len(votes)) if truth[t] >= 0]
    test_items = [t for t in answerable if t % 2 == 1]
    best = max(
        sum(votes[t][i] == truth[t] for t in test_items) / len(test_items) for i in range(n)
    )
    assert abs(100 * best - d["quorum"]["bestSingleTestPct"]) < 1e-9, (
        f"the record reports a best single voter of {d['quorum']['bestSingleTestPct']}%, a "
        f"run over the same votes gives {100 * best}%"
    )
