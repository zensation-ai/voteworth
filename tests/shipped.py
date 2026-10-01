"""Which files are shipped — discovered, not typed.

A typed file list is a second place where the truth lives, and the two drift apart silently:
a new file simply never appears in it, and every gate that walks the list reports clean while
covering less. Measured here on 2026-09-20: `LICENSE` was shipped and checked by nothing,
because nobody thought to add it to two hand-written lists.

So the list is derived from the directory, and exclusions carry a reason. A file that is
neither discovered nor excluded-with-a-reason makes the check below fail — an omission has to
be stated, not merely happen.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Names that are NEVER a source directory, whatever else is going on. Nobody ships a package
# module called `__pycache__`, and `site-packages` is by definition somebody else's code.
_NEVER_SOURCE = {
    "__pycache__", ".git", ".pytest_cache", ".tox", ".mypy_cache", ".ruff_cache",
    "site-packages", "node_modules",
}

# Names that USUALLY mean a virtual environment — and sometimes mean a module. `env` is a
# perfectly ordinary package name. These are pruned only when the structure agrees.
_MAYBE_VENV = {".venv", "venv", ".env", "env", ".virtualenv"}

# Build outputs, which live beside the project and never inside it. `src/voteworth/build/`
# would be a module; `./build/` is what the packager just wrote.
_BUILD_OUTPUT = {"build", "dist"}

# Suffixes and bare names that carry prose or configuration a reader will see.
_TEXT_SUFFIXES = {".md", ".py", ".toml", ".cff", ".yml", ".yaml", ".txt", ".in", ".cfg"}
# PKG-INFO and setup.cfg exist only in a built source distribution — setuptools writes them
# from pyproject.toml. They are listed here rather than excluded, so that the prose gates walk
# them in the channel where they exist and simply do not see them in a checkout. An exclusion
# would have to assert their existence, which is false in half the cases.
_TEXT_NAMES = {"NOTICE", "LICENSE", "PKG-INFO"}

# Files deliberately outside the prose gates, each with the reason. An entry here is a
# decision on the record; an absence from both this map and the discovery is a failure.
EXCLUDED: dict[str, str] = {
    ".gitignore": "no prose; a list of path patterns",
    "src/voteworth/data/panel_10_voters.json": (
        "third-party-cited data, published separately under CC BY 4.0 at "
        "10.5281/zenodo.21773065 — its content is the record's, not ours to edit"
    ),
}


def _prune(directory: Path) -> bool:
    """Should the walk stop at this directory?

    A name list alone is a rule about the names someone thought of. Measured 2026-09-21: a
    virtual environment created as `.v` — any name but the six on the old list — put the whole
    of site-packages inside the walk. Two gates then reported numpy's own test files as
    offenders, and one of them took 128 seconds to do it. `.gitignore` already named `venv/`,
    which the list did not, so a stranger following the most ordinary convention would have
    hit exactly this.

    So the rule is structural first: a directory holding `pyvenv.cfg` IS a virtual environment,
    whatever it is called. The names remain as a cheap first test.
    """
    name = directory.name
    if name in _NEVER_SOURCE or name.endswith(".egg-info"):
        return True
    # A virtual environment, whatever it is called: the structure decides, not the name. This
    # is the rule — a directory holding `pyvenv.cfg` IS a venv.
    if (directory / "pyvenv.cfg").exists():
        return True
    # And the names that only SUGGEST one get no vote of their own. An acceptance pass on
    # 25.09.2026 measured the cost of letting them: a file at `src/voteworth/env/real.py` was
    # invisible to every gate, because a name list had overruled the structural test it was
    # supposed to shortcut. That is the same defect as the one this function was written to
    # fix, pointing the other way — first the list was too narrow, then too wide. A shortcut
    # that can overrule its own rule is not a shortcut.
    if name in _MAYBE_VENV:
        return False  # looks like one, is not one — walk it
    # Build output lives beside the project, never inside it.
    return name in _BUILD_OUTPUT and directory.parent.resolve() == ROOT.resolve()


def walk_files(root: Path):
    """Every file under `root`, with skipped directories PRUNED rather than filtered.

    Pruning, not filtering: `rglob` descends into a virtual environment before discarding what
    it found there, which is where the 128 seconds went.
    """
    import os

    for base, dirnames, filenames in os.walk(root):
        base_path = Path(base)
        dirnames[:] = sorted(d for d in dirnames if not _prune(base_path / d))
        for name in sorted(filenames):
            path = base_path / name
            if path.suffix == ".pyc":
                continue
            yield path, str(path.relative_to(root))


def _walk_files():
    return walk_files(ROOT)


def discover(include_tests: bool = False) -> list[str]:
    """Every shipped text file, relative to the package root, sorted."""
    found = []
    for path, rel in _walk_files():
        if not include_tests and rel.startswith("tests/"):
            continue
        if path.suffix in _TEXT_SUFFIXES or path.name in _TEXT_NAMES:
            found.append(rel)
    return sorted(found)


def discover_all() -> list[str]:
    """EVERY shipped file, whatever its suffix — the surface the exclusion rule is about.

    `discover()` filters to prose-bearing suffixes, which is right for the prose gates and
    wrong for the completeness check that uses it: a file with an unlisted suffix was never
    discovered, so it could never be "unaccounted for", so its entry in EXCLUDED could never
    fire. Measured on 2026-09-21: both entries in EXCLUDED were unreachable that way. The
    rule is about what ships; so is this walk.
    """
    return sorted(rel for _, rel in _walk_files())


def gated() -> list[str]:
    """The files the prose gates must walk: discovered, minus stated exclusions."""
    return [f for f in discover() if f not in EXCLUDED]


def gated_all() -> list[str]:
    """Same, but including this suite's own files.

    The prohibition gates used to walk `gated()`, which drops `tests/` — more than three times
    the prose of `src/`, unscanned, while the guarantee read as "nothing shipped claims this".
    A rule that skips the largest surface it applies to is a rule about the rest.
    """
    return [f for f in discover(include_tests=True) if f not in EXCLUDED]


# ── declared teaching regions ────────────────────────────────────────────────
#
# A file that DEFINES a prohibition has to quote what it forbids: its positive control is, by
# construction, a line carrying exactly the phrasing the rule bans. Scanning the suite without
# a way to say so would force the honest choice between deleting the controls and deleting the
# scan. (This comment used to carry a worked example, which the widened scan then flagged —
# documenting an absence by quoting the absent thing is how that check fails.)
#
# So a region can be declared — and `test_declared_regions_all_fire` in each gate requires
# every declared region to actually contain a match. A region that exempts nothing is dead
# weight, and dead weight is where a blanket exemption starts.
# Composed, not written out: a line holding the whole marker text IS a marker to the scan
# below, and this file would then declare an empty region at its own definition. Measured
# 2026-09-21 — the third time in this repository that a check sat inside its own subject.
_MARKER = "gate-fixture"
FIXTURE_BEGIN = f"# {_MARKER}: begin"
FIXTURE_END = f"# {_MARKER}: end"


def fixture_regions(path: Path) -> list[str]:
    """The text of each declared region, in order."""
    regions, current = [], None
    for line in path.read_text(errors="ignore").splitlines():
        if FIXTURE_BEGIN in line:
            assert current is None, f"{path.name}: nested {FIXTURE_BEGIN}"
            current = []
        elif FIXTURE_END in line:
            assert current is not None, f"{path.name}: {FIXTURE_END} without a begin"
            regions.append(" ".join(" ".join(current).split()))
            current = None
        elif current is not None:
            current.append(line)
    assert current is None, f"{path.name}: {FIXTURE_BEGIN} without an end"
    return regions


def read_flat_outside_fixtures(path: Path) -> str:
    """The file's prose with declared regions removed, whitespace collapsed.

    Line numbers are not preserved — `locate` finds the needle independently — but the region
    boundaries are, so a forbidden phrase one line outside a region is still caught.
    """
    kept, inside = [], False
    for line in path.read_text(errors="ignore").splitlines():
        if FIXTURE_BEGIN in line:
            inside = True
        elif FIXTURE_END in line:
            inside = False
        elif not inside:
            kept.append(line)
    return " ".join(" ".join(kept).split())


def read_flat(path: Path) -> str:
    """A file's text with every run of whitespace collapsed to one space.

    **For prohibition checks, read this — not the raw lines.** A positive check whose sentence
    wraps produces a false alarm: loud, annoying, harmless. A *prohibition* check whose
    forbidden phrase wraps produces silence, and silence is indistinguishable from a clean
    file. Measured here on 2026-09-20: three prohibition gates (novelty claims, submission
    attribution, withdrawn figures) each let a wrapped instance through while reporting green.

    The distinction is a sibling session's: it is the direction of the check that decides
    whether a line-wrap is an annoyance or a hole.
    """
    return " ".join(path.read_text(errors="ignore").split())


def locate(path: Path, needle: str) -> int:
    """Best-effort line number for a needle found in the flattened text.

    Returns the line where the match starts, or 0 if it cannot be pinned down. A wrapped
    match is reported at its first line — enough to find it, and honest that it spans more.
    """
    words = needle.split()
    if not words:
        return 0
    first = words[0]
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        if first in line:
            return i
    return 0
