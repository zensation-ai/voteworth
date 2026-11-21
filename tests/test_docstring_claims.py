"""Do the docstrings still describe this package?

`help(voteworth.bloc_share)` is a public surface like the README, and it ships inside the wheel,
where the README's own checks never run. Until 0.1.4 nothing tested it: the docstring of
`bloc_share` said the weighted vote "lands 20 points below the best single voter", while the
experiment it describes gives a median of 17 over twenty seeds, and 14.5 on the seed the helper
defaults to. A check before the 0.1.3 upload found it (2026-11-21).

This file binds docstring figures the way test_readme_claims.py binds the README's: a decimal
must be produced by a live run or declared with a source, and a "points below the best single
voter" figure must be the one the boundary experiment gives.
"""
from __future__ import annotations

import inspect
import io
import re
import statistics
from contextlib import redirect_stdout

import pytest

import voteworth
import voteworth.aggregate
import voteworth.demo
from test_aggregate import boundary_run
from test_readme_claims import FOREIGN_FIGURES, _decimals, _percentages


@pytest.fixture(scope="module")
def demo_output() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        voteworth.demo.main()
    return buf.getvalue()


def _docstrings() -> dict[str, str]:
    """Every docstring a user can reach with help(): the modules and each public function."""
    docs = {m.__name__: inspect.getdoc(m) or "" for m in (voteworth, voteworth.aggregate, voteworth.demo)}
    for name in voteworth.__all__:
        docs[f"voteworth.{name}"] = inspect.getdoc(getattr(voteworth, name)) or ""
    return docs


def _unexplained(doc: str, produced: set[str]) -> list[str]:
    return sorted(q for q in _decimals(doc) if q not in produced and q not in FOREIGN_FIGURES)


def test_docstring_decimals_are_produced_or_declared(demo_output):
    """Every decimal in a docstring is reproducible from a live run or declared with a source."""
    produced = _decimals(demo_output) | _percentages(demo_output)
    unexplained = {where: u for where, doc in _docstrings().items() if (u := _unexplained(doc, produced))}
    assert not unexplained, (
        "docstrings state figures that a live run does not produce and FOREIGN_FIGURES does "
        f"not declare: {unexplained}"
    )


def test_the_docstring_figure_check_can_fire(demo_output):
    """Control arm: a planted figure that nothing produces must be reported."""
    produced = _decimals(demo_output) | _percentages(demo_output)
    assert _unexplained("Above 0.77 the method stops working.", produced) == ["0.77"]


_POINTS_CLAIM = re.compile(r"(\d+) points below the best single voter")


def _boundary_median_points() -> int:
    """The median gap to the best single voter over the twenty seeds of the boundary test."""
    gaps = [run["best"] - run["acc_w"] for run in (boundary_run(seed) for seed in range(20))]
    return round(100 * statistics.median(gaps))


def _points_problems(doc: str, median_points: int) -> list[str]:
    return [
        f"says {m.group(1)} points; the boundary experiment gives a median of {median_points}"
        for m in _POINTS_CLAIM.finditer(" ".join(doc.split()))
        if int(m.group(1)) != median_points
    ]


def test_docstring_point_figures_match_the_boundary_experiment():
    """A docstring that quotes the boundary gap quotes the median the experiment gives."""
    median_points = _boundary_median_points()
    docs = _docstrings()
    problems = {where: p for where, doc in docs.items() if (p := _points_problems(doc, median_points))}
    assert not problems, f"docstring figures the boundary experiment does not give: {problems}"
    assert any(_POINTS_CLAIM.search(" ".join(doc.split())) for doc in docs.values()), (
        "no docstring quotes the boundary gap any more; has the docstring of bloc_share changed shape?"
    )


def test_the_points_check_can_fire():
    """Control arm: the 0.1.3 docstring of bloc_share must be reported."""
    old = ("Measured on a synthetic 10-voter panel with a 6-voter bloc, the weighted vote equals "
           "the naive one and lands 20 points below the best single voter.")
    assert _points_problems(old, _boundary_median_points())
