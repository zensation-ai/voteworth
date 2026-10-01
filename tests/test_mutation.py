"""Does the test suite actually test anything?

A passing suite proves nothing on its own. It proves something only if we have seen it fail
for the right reason. So this file breaks the mechanism on purpose, in three ways, and
requires that the suite catches each one — and then requires that the unmutated source still
passes, so a suite that fails for *any* reason cannot be mistaken for a working alarm.

The question behind it: *have I ever seen this instrument produce the opposite value?* If the
answer is no, I do not know that it measures — I only know what it currently says.

Each mutation names the claim it attacks:

* `uniform_weights` kills the repair itself. If nothing fails, the package's core claim is
  untested.
* `tie_break_reversed` flips the documented determinism. If nothing fails, "deterministic" is
  an unchecked word in the docstring.
* `blind_to_shared_errors` removes the correlation signal while leaving the plumbing intact.
  If nothing fails, the suite is testing the plumbing, not the detection.

This runs in CI. It is slower than the rest of the suite by design — it forks a pytest per
mutation.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = _ROOT / "src" / "voteworth" / "aggregate.py"

# Non-code files the copied tests read. Kept beside the copy itself so the two cannot drift.
_SANDBOX_FILES = ("README.md", "NOTICE", "pyproject.toml", "CITATION.cff")

# (name, needle, replacement, the claim this mutation attacks)
MUTATIONS = [
    (
        "uniform_weights",
        "        weights.append(1.0 / (1.0 + r))",
        "        weights.append(1.0)  # MUTANT",
        "independence weighting changes the outcome at all",
    ),
    (
        "tie_break_reversed",
        "    return max(sorted(acc), key=lambda v: acc[v])",
        "    return max(sorted(acc, reverse=True), key=lambda v: acc[v])  # MUTANT",
        "ties break to the smallest id, deterministically",
    ),
    (
        "blind_to_shared_errors",
        "                if votes[i] == votes[j]:",
        "                if False:  # MUTANT",
        "shared error modes are detected, not just co-deviation",
    ),
]


def _run_suite(package_root: Path) -> subprocess.CompletedProcess:
    """Run the unit tests (not this file) against a given copy of the package."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(package_root / "tests" / "test_aggregate.py"),
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=package_root,
        env={"PYTHONPATH": str(package_root / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def sandbox():
    """An isolated copy of the package, so a mutation can never touch the real source."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pkg"
        shutil.copytree(_ROOT / "src", root / "src")
        shutil.copytree(_ROOT / "tests", root / "tests")
        (root / "tests" / "test_mutation.py").unlink()  # do not recurse into ourselves

        # The surfaces the copied tests read. test_aggregate.py parses the boundary figures
        # out of the README, so a sandbox without it fails for a reason that has nothing to
        # do with any mutation — which is the one failure mode a positive control must not
        # have. Asserted rather than assumed: a test that later reads a file missing from
        # this list should say so, not report a mutation that was never applied.
        for name in _SANDBOX_FILES:
            shutil.copy2(_ROOT / name, root / name)
            assert (root / name).exists(), f"sandbox is missing {name}"
        yield root


def test_positive_control_suite_passes_unmutated(sandbox):
    """The control arm. If the untouched copy does not pass, every failure below is
    meaningless — it would prove the sandbox is broken, not that the suite detects mutants."""
    result = _run_suite(sandbox)
    assert result.returncode == 0, (
        "the unmutated copy must pass, otherwise the mutation results below say nothing:\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("name,needle,replacement,claim", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_suite_detects_mutation(sandbox, name, needle, replacement, claim):
    target = sandbox / "src" / "voteworth" / "aggregate.py"
    original = target.read_text()

    assert needle in original, (
        f"mutation {name!r} no longer applies — the line it rewrites has moved or changed. "
        "Update MUTATIONS; a mutation that silently stops applying is a test that silently "
        "stops testing."
    )

    try:
        target.write_text(original.replace(needle, replacement, 1))
        result = _run_suite(sandbox)
        assert result.returncode != 0, (
            f"the suite passed with mutation {name!r} applied, so nothing tests the claim: "
            f"{claim}.\n{result.stdout}"
        )
    finally:
        target.write_text(original)


def test_source_is_unchanged_after_mutations():
    """Belt and braces: the real source file must never have been touched."""
    assert "MUTANT" not in _SOURCE.read_text()
