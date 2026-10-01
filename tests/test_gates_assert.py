"""Does every gate in this suite actually assert anything?

A test that collects violations into a list and never checks the list is always green. It
reads like a gate, it appears in the pass count, and it guards nothing.

This is not hypothetical here. On 2026-09-20 a regex rewrite of two prohibition gates removed
their `assert` lines along with the loop body being replaced. The suite reported 107 passed
while two of those tests checked nothing at all. It was found only because a fault injection
was repeated afterwards and stayed silent — not by reading the diff, and not by the count.

The detection strategy is a sibling session's, and their framing came with it: **measure the
instrument before believing its null.** Their first sweep returned zero findings and they
discarded it, because a zero from an instrument never seen to fire says nothing. So the
positive control below runs first, on a file that deliberately contains both shapes.

The discriminating trick, also theirs: when counting whether a collected list is ever *read*,
exclude the receiver of `.append`. Without that exclusion every filled list looks used, and
the check goes blind in exactly the case it exists for.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent

# Methods that FILL a collection rather than read it. Their receiver is a Name load, but it is
# not a use of the contents — so it must not count as "someone looks at this list".
#
# The first version listed only append and extend: the two that came to mind. Measured against
# a probe carrying all six, it caught 2 of 6. The gap was found by a sibling session, who hit
# the same wall one method earlier — their list had append alone. Neither of us found our own
# gap; each found the other's by reading a single word in a message.
#
# Measured against the stdlib on 2026-09-21, after an acceptance pass asked where the list
# came from: `appendright` stood here and **exists on no built-in type** — the list had been
# extended by pattern completion, not by reading the API. `extendleft` is real and was
# missing. A completeness list whose positive control plants exactly its own entries can
# never show its own gap, so the control below is generated from the types instead.
_FILLING = {"append", "extend", "extendleft", "insert", "appendleft", "add", "update"}


# Constructors that make a fresh, empty collection. Only list literals and comprehensions were
# tracked before, which meant a gate collecting into `set()` or `deque()` was invisible to this
# detector while the method list above carefully named their filling methods — the list was
# broader than the thing it was applied to, so half of it could never fire.
_COLLECTION_CALLS = {"list", "set", "dict", "deque", "defaultdict", "Counter", "OrderedDict"}


def _is_empty_collection(value: ast.expr) -> bool:
    """Is this expression a fresh collection a gate would collect findings into?"""
    if isinstance(value, (ast.List, ast.ListComp, ast.SetComp, ast.DictComp)):
        return True
    if isinstance(value, (ast.Set, ast.Dict)):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        return name in _COLLECTION_CALLS
    return False


def find_unread_collectors(source: str) -> list[tuple[str, int]]:
    """Names bound to a list (or list comprehension) that are never read.

    Appending to a name is not reading it — `offenders.append(x)` loads `offenders` only to
    reach its method. A collector that is only ever appended to, and never passed to an
    assert, a call, a comparison or a return, holds findings nobody looks at.

    Returns (name, lineno) per collector, sorted by line.
    """
    tree = ast.parse(source)
    collectors: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_empty_collection(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    collectors[target.id] = node.lineno

    if not collectors:
        return []

    # Name loads that only reach the container, never its contents. Two shapes:
    #   1. the receiver of a filling method:  offenders.append(x)
    #   2. the container of a store subscript:  offenders[0:0] = [x]   offenders[0] = x
    # The second was found by a sibling session and is NOT a method call, which is why a
    # method-name list alone can never be complete. Their own probe nearly misfired on it:
    # `x[len(x):] = [...]` puts a REAL read inside the slice, so that name must still count
    # as read. Handling the container separately from the slice expression keeps both right.
    container_only: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _FILLING and isinstance(node.func.value, ast.Name):
                container_only.add(id(node.func.value))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name):
                container_only.add(id(node.value))

    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if id(node) not in container_only:
                read.add(node.id)

    return sorted(
        ((name, line) for name, line in collectors.items() if name not in read),
        key=lambda pair: pair[1],
    )


# Both shapes of the defect, written out so the detector can be seen to fire. Shape (a) is
# the one that actually occurred here; shape (b) is the comprehension variant.
_PLANTED = textwrap.dedent(
    '''
    def test_shape_a_collects_and_never_checks():
        offenders = []
        for name in files():
            if bad(name):
                offenders.append(name)


    def test_shape_b_comprehension_never_read():
        problems = [f for f in files() if bad(f)]


    def test_healthy_gate_does_assert():
        found = []
        for name in files():
            if bad(name):
                found.append(name)
        assert not found, found


    def test_healthy_gate_returns_it():
        rows = [f for f in files()]
        return len(rows)


    def test_filled_by_insert_never_checked():
        v_insert = []
        for f in files():
            v_insert.insert(0, f)


    def test_filled_by_add_never_checked():
        v_add = []
        for f in files():
            v_add.add(f)


    def test_filled_by_update_never_checked():
        v_update = []
        for f in files():
            v_update.update(f)


    def test_healthy_collector_filled_then_read():
        healthy = []
        for f in files():
            healthy.append(f)
        assert not healthy, healthy


    def test_filled_by_slice_never_checked():
        v_slice = []
        v_slice[0:0] = [1]


    def test_filled_by_index_never_checked():
        v_index = []
        v_index[0] = 1


    def test_slice_with_real_read_is_healthy():
        v_lenread = []
        v_lenread[len(v_lenread):] = [1]
    '''
)


def test_controls_detector_finds_exactly_the_planted_defects():
    """Run this before trusting any null below. A zero from an instrument never seen to fire
    is not a finding — it is an unmeasured claim.

    Asserted as an EQUALITY, not as a superset. A superset assertion passes for a detector
    that flags every list it sees, which would be useless and green. The sibling session's
    first version had exactly that weakness and said so.
    """
    found = {name for name, _ in find_unread_collectors(_PLANTED)}
    expected = {
        "offenders",   # shape (a): collected in a loop, never checked
        "problems",    # shape (b): comprehension, never read
        "v_insert",    # filled by .insert — receiver must not count as a read
        "v_add",       # .add
        "v_update",    # .update
        "v_slice",     # x[0:0] = [...] — a store subscript, not a method
        "v_index",     # x[0] = ... — same shape
        # NOT expected: v_lenread. `x[len(x):] = [...]` reads the name inside the slice, so
        # something does look at it. Flagging it would be a false alarm, and the sibling
        # session nearly reported one on exactly this construction.
    }
    assert found == expected, (
        f"detector output does not match the planted set.\n"
        f"  missed:        {sorted(expected - found)}\n"
        f"  falsely flagged: {sorted(found - expected)}"
    )


# ── second arm: read, but never asserted ─────────────────────────────────────
#
# The arm above asks whether a collector is READ. That is not the same question as whether the
# test ASSERTS. `rows = [...]; return len(rows)` reads the list and checks nothing — it passes
# the arm above and passes pytest, and `test_healthy_gate_returns_it` in the planted sample
# above is exactly that shape, sitting in this file since it was written, deliberately excluded
# from the expected set. An acceptance pass on 2026-09-21 pointed at it: the file carried its
# own counter-example and called it healthy.
#
# Naming the gap in a docstring does not close it. This arm closes it.

_ASSERTING_CALLS = {"fail", "raises", "warns", "approx", "xfail", "exit", "skip"}


def _asserts_something(func: ast.AST) -> bool:
    """Does this function body contain anything that can make the test fail?"""
    for node in ast.walk(func):
        if isinstance(node, (ast.Assert, ast.Raise)):
            return True
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in _ASSERTING_CALLS:
                return True
        # `with pytest.raises(...)` is the assertion, even with no assert inside.
        if isinstance(node, ast.withitem):
            expr = node.context_expr
            if isinstance(expr, ast.Call):
                f = expr.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in _ASSERTING_CALLS:
                    return True
    return False


def find_vacuous_tests(source: str) -> list[tuple[str, int]]:
    """Test functions that cannot fail: no assert, no raise, no pytest.raises/fail."""
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not _asserts_something(node):
                out.append((node.name, node.lineno))
    return sorted(out, key=lambda pair: pair[1])


def test_controls_vacuous_detector_finds_exactly_the_planted_ones():
    """Positive control for the second arm, asserted as an equality for the same reason."""
    found = {name for name, _ in find_vacuous_tests(_PLANTED)}
    expected = {
        "test_shape_a_collects_and_never_checks",
        "test_shape_b_comprehension_never_read",
        "test_healthy_gate_returns_it",   # reads the list, asserts nothing — the new shape
        "test_filled_by_insert_never_checked",
        "test_filled_by_add_never_checked",
        "test_filled_by_update_never_checked",
        "test_filled_by_slice_never_checked",
        "test_filled_by_index_never_checked",
        "test_slice_with_real_read_is_healthy",
    }
    assert found == expected, (
        f"vacuous-test detector does not match the planted set.\n"
        f"  missed:          {sorted(expected - found)}\n"
        f"  falsely flagged: {sorted(found - expected)}"
    )


def test_the_filling_list_names_only_methods_that_exist():
    """Control arm for `_FILLING` that does NOT come from `_FILLING`.

    The planted sample above uses exactly the methods the list already names, so it can
    confirm the list but never expose a hole in it. This asks the interpreter instead: every
    name must exist on some real collection type. `appendright` stood in the list for two days
    and exists nowhere — that is what a list checked against its own probe looks like.
    """
    import collections

    types = (list, set, dict, collections.deque)
    missing = [m for m in _FILLING if not any(hasattr(t, m) for t in types)]
    assert not missing, (
        f"_FILLING names methods that exist on no collection type: {missing} — the list was "
        "extended by pattern completion rather than read off the API"
    )
    # And the other direction: a mutating method the stdlib has and we do not name.
    known_absent = {"appendleft", "extendleft"}  # deque-only, kept for collectors we may add
    for method in ("append", "extend", "insert", "add", "update"):
        assert method in _FILLING, f"{method} fills a collection but is not in _FILLING"
    assert known_absent <= _FILLING


@pytest.mark.parametrize(
    "path", sorted(_TESTS.glob("test_*.py")), ids=lambda p: p.name
)
def test_no_gate_is_vacuous(path):
    """Every test in this suite must be able to fail."""
    vacuous = find_vacuous_tests(path.read_text())
    assert not vacuous, (
        f"{path.name} contains tests that assert nothing:\n  "
        + "\n  ".join(f"line {line}: {name}" for name, line in vacuous)
        + "\nA test that cannot fail is always green."
    )


@pytest.mark.parametrize(
    "path", sorted(_TESTS.glob("test_*.py")), ids=lambda p: p.name
)
def test_no_gate_collects_without_checking(path):
    """The guard itself, over every test file in this suite — including this one."""
    unread = find_unread_collectors(path.read_text())
    assert not unread, (
        f"{path.name} builds a list that nothing ever reads:\n  "
        + "\n  ".join(f"line {line}: {name!r}" for name, line in unread)
        + "\nA collector that is never checked is a gate that cannot fail."
    )


# ── third arm: does every declared exemption exempt something? ───────────────


def test_every_declared_region_exempts_something():
    """No `gate-fixture` region anywhere may be dead weight.

    Each gate vouches for the regions in its own file, which leaves a gap: a region declared
    in a THIRD file — exempt from a prohibition whose gate lives elsewhere — would be vouched
    for by nobody. `tests/test_claims_are_bound.py` carries exactly such a region, marked so
    the novelty gate would stop flagging its control sentences.

    So this walks every region in the suite against every prohibition pattern, and requires
    each to fire for at least one. A region that no pattern matches exempts nothing, and an
    exemption that cannot fire is where a blanket starts.
    """
    from shipped import fixture_regions
    from test_claims_are_bound import _SWEEPING
    from test_no_novelty_claim import NOVELTY_CLAIM
    from test_no_submission_attribution import SUBMISSION_ATTRIBUTION

    patterns = {
        "novelty": NOVELTY_CLAIM,
        "submission": SUBMISSION_ATTRIBUTION,
        "sweeping": _SWEEPING,
    }
    seen = 0
    for path in sorted(_TESTS.glob("*.py")):
        for region in fixture_regions(path):
            seen += 1
            hits = [name for name, pat in patterns.items() if pat.search(region)]
            assert hits, (
                f"{path.name}: a declared region matches no prohibition pattern — it exempts "
                f"nothing:\n  {region[:160]!r}"
            )
    # A count, not a boolean: zero regions would pass the loop above in silence, and that is
    # indistinguishable from a marker syntax nobody is reading any more.
    assert seen >= 6, f"only {seen} declared regions found across the suite — is the marker being read?"


def test_no_gate_file_hides_behind_an_unreadable_marker():
    """Every begin has an end, in every file — an unbalanced marker would swallow the rest."""
    from shipped import FIXTURE_BEGIN, FIXTURE_END, fixture_regions

    for path in sorted(_TESTS.glob("*.py")):
        text = path.read_text()
        begins, ends = text.count(FIXTURE_BEGIN), text.count(FIXTURE_END)
        assert begins == ends, f"{path.name}: {begins} begin markers, {ends} end markers"
        fixture_regions(path)  # raises on nesting or an unclosed region


def test_the_walk_prunes_a_virtual_environment_whatever_it_is_called(tmp_path):
    """Control arm for the directory walk, which is what every prose gate stands on.

    A venv named anything but the handful on the skip list used to put site-packages inside
    the scan: third-party files reported as offenders, and one gate taking over two minutes.
    The structural test — a directory holding `pyvenv.cfg` is a virtual environment — must
    hold for a name nobody listed.
    """
    import shipped

    (tmp_path / "real.py").write_text("x = 1\n")
    for name in ("wildly-unconventional-env", "venv", ".v"):
        env = tmp_path / name
        (env / "lib").mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("home = /usr\n")
        (env / "lib" / "planted.py").write_text("y = 2\n")

    original = shipped.ROOT
    shipped.ROOT = tmp_path
    try:
        found = shipped.discover_all()
    finally:
        shipped.ROOT = original

    assert "real.py" in found, "the walk missed an ordinary file — it is pruning too much"
    leaked = [f for f in found if "planted.py" in f or "pyvenv.cfg" in f]
    assert not leaked, f"the walk descended into a virtual environment: {leaked}"


def test_the_walk_does_not_prune_a_directory_that_merely_looks_like_one(tmp_path):
    """The other direction, and the one that was wrong.

    The first fix replaced a too-narrow name list with a structural test — and kept the names
    as a fast path that fired FIRST. So `env`, `build`, `dist` and `node_modules` were pruned
    unconditionally, and a real module at `src/voteworth/env/real.py` became invisible to every
    gate in this suite. Measured 25.09.2026: 0 of 4 such files were seen.

    A name may suggest a virtual environment. Only `pyvenv.cfg` establishes one.
    """
    import shipped

    (tmp_path / "ordinary.py").write_text("z = 3\n")
    for name in ("env", "venv", "build", "dist"):
        d = tmp_path / "src" / "voteworth" / name
        d.mkdir(parents=True)
        (d / "real.py").write_text("y = 2\n")
    # A genuine environment, however it is named, still goes.
    genuine = tmp_path / "src" / "voteworth" / "env-but-real"
    genuine.mkdir(parents=True)
    (genuine / "pyvenv.cfg").write_text("home = /usr\n")
    (genuine / "hidden.py").write_text("x = 1\n")
    # Build output beside the project still goes.
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artefact.py").write_text("a = 0\n")

    original = shipped.ROOT
    shipped.ROOT = tmp_path
    try:
        found = shipped.discover_all()
    finally:
        shipped.ROOT = original

    for name in ("env", "venv", "build", "dist"):
        assert f"src/voteworth/{name}/real.py" in found, (
            f"a module directory called {name!r} was pruned on its name alone — "
            "a real file inside it is invisible to every gate"
        )
    assert "ordinary.py" in found
    assert not [f for f in found if "hidden.py" in f], (
        "a directory with pyvenv.cfg was walked — the structural test stopped working"
    )
    assert not [f for f in found if f.startswith("build/")], (
        "build output beside the project was walked"
    )
