"""`python -m voteworth.demo` — the same run as `examples/quickstart.py`.

Kept as a module so the demo works straight after `pip install voteworth`, without a
checkout of the repository.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from voteworth.aggregate import (
    agreement_confidence,
    aggregate,
    bloc_share,
    effective_votes,
    learn_independence_weights,
    naive_weights,
    pairwise_coerror,
    structural_weights,
)

DATA = Path(__file__).resolve().parent / "data" / "panel_10_voters.json"


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


# A group with no observations is not a group that measured zero. Printing 0.0 for "nothing
# co-deviated" would be an absence dressed as a measurement — and the panel shape this package
# warns about is exactly the one that produces it: a bloc that IS the plurality never deviates
# from it, so every same-model pair comes back NaN. Until 2026-09-21 the demo divided by that
# empty count and raised ZeroDivisionError *before* reaching its own warning.
_NONE = f"{'—':>6}"


def mean_or_none(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct_or_dash(x: float | None) -> str:
    return _NONE if x is None else pct(x)


def evaluate(items, votes, truth, weights, subset=None):
    """Accuracy and mean agreement-confidence over `items`, using `weights`."""
    idx = subset if subset is not None else list(range(len(weights)))
    hits, confs = 0, []
    for t in items:
        row = [votes[t][i] for i in idx]
        w = [weights[i] for i in idx]
        hits += aggregate(row, w) == truth[t]
        confs.append(agreement_confidence(row, w))
    n = len(items)
    return hits / n, sum(confs) / n


def main() -> None:
    d = json.loads(DATA.read_text())
    votes, truth, panel = d["votesByItem"], d["correctGroupByItem"], d["panel"]
    n = len(panel)

    # Only items with a gold answer can be scored. Unanswerable items are excluded and
    # reported, rather than silently counted as wrong.
    answerable = [t for t in range(len(votes)) if truth[t] >= 0]
    # Learn on one half, score on the other. Doing both on the same items would report the
    # fit, not the effect.
    history_items = [t for t in answerable if t % 2 == 0]
    test_items = [t for t in answerable if t % 2 == 1]

    print(f"\nPanel: {n} voters over {len(votes)} items "
          f"({len(answerable)} answerable, {len(votes) - len(answerable)} without a gold answer)")
    by_model: dict[str, list[str]] = defaultdict(list)
    for p in panel:
        by_model[p["model"]].append(p["name"])
    for model, names in by_model.items():
        print(f"  {len(names)}x {model}")

    # ── 1. the correlated bloc ────────────────────────────────────────────────
    m = pairwise_coerror([votes[t] for t in answerable], n)
    same, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if m[i][j] == m[i][j]:  # not NaN
                (same if panel[i]["model"] == panel[j]["model"] else cross).append(m[i][j])
    print("\n=== 1. When two voters are both wrong, do they agree on the wrong answer? ===")
    same_r, cross_r = mean_or_none(same), mean_or_none(cross)
    print(f"  same model      {pct_or_dash(same_r)}   ({len(same)} pairs)")
    print(f"  different model {pct_or_dash(cross_r)}   ({len(cross)} pairs)")

    # The same comparison under three different choices. Nothing about the votes changes; only
    # which items count, what "agreeing" means on them, and how pairs are averaged. The ratio
    # swings by a factor of seven while the absolute gap barely moves — which is why a ratio
    # quoted without its base cannot be compared to anyone else's.
    def _gap(pred):
        s_hit = s_tot = c_hit = c_tot = 0
        for t in answerable:
            for i in range(n):
                for j in range(i + 1, n):
                    c = pred(t, i, j)
                    if c is None:
                        continue
                    if panel[i]["model"] == panel[j]["model"]:
                        s_tot += 1
                        s_hit += c
                    else:
                        c_tot += 1
                        c_hit += c
        # None, not 0.0: no observed pairs is not a measured zero.
        return (s_hit / s_tot if s_tot else None), (c_hit / c_tot if c_tot else None)

    def _both_wrong(t, i, j):
        a, b = votes[t][i], votes[t][j]
        if a == -1 or b == -1 or a == truth[t] or b == truth[t]:
            return None
        return 1 if a == b else 0

    def _raw(t, i, j):
        a, b = votes[t][i], votes[t][j]
        return None if (a == -1 or b == -1) else (1 if a == b else 0)

    def _codeviate(t, i, j):
        row = votes[t]
        present = [k for k in range(n) if row[k] != -1]
        if len(present) < 2:
            return None
        from voteworth.aggregate import _plurality
        pl = _plurality([row[k] for k in present])
        if row[i] == -1 or row[j] == -1 or row[i] == pl or row[j] == pl:
            return None
        return 1 if row[i] == row[j] else 0

    # All three rows pooled over events, so the rows differ ONLY in base and definition.
    # Mixing a pair-mean row into this table would fold in the third source of variation —
    # the averaging — and hide the very thing the table is meant to separate.
    print("\n  The same question under three choices of base and definition:")
    print(f"  {'(all pooled over events)':44} {'same':>6} {'cross':>7} {'gap':>7} {'ratio':>7}")
    for label, pred in (
        ("both deviate from plurality (used here)", _codeviate),
        ("both wrong", _both_wrong),
        ("raw agreement, all items", _raw),
    ):
        sr, cr = _gap(pred)
        if sr is None or cr is None:
            print(f"  {label:44} {_NONE} {'—':>7} {'—':>7} {'—':>7}   (no pairs observed)")
        elif cr == 0:
            print(f"  {label:44} {sr:6.3f} {cr:7.3f} {sr - cr:+7.3f} {'—':>7}   (ratio undefined)")
        else:
            print(f"  {label:44} {sr:6.3f} {cr:7.3f} {sr - cr:+7.3f} "
                  f"{100 * (sr - cr) / cr:+6.0f}%")
    sr_p, cr_p = _gap(_codeviate)
    if None in (same_r, cross_r, sr_p, cr_p):
        print("\n  Third source — averaging: not shown, because one of the two groups has no "
              "observed pairs.")
        print("  That absence is itself the finding — see the warning above.")
    else:
        print(f"\n  Third source — averaging. The headline above means over voter PAIRS "
              f"({same_r:.3f} / {cross_r:.3f});")
        print(f"  pooling over events gives {sr_p:.3f} / {cr_p:.3f}, because pairs contribute "
              f"unequally.")
    print("  → The ABSOLUTE gap is stable across all of this; the ratio is not. Published")
    print("    absolute gaps sit in the same band. Match base, definition and averaging")
    print("    before comparing your panel to anyone else's number.")

    ne_raw = effective_votes([votes[t] for t in answerable], n)
    share = bloc_share([p["model"] for p in panel])
    print(f"\n  effective votes (Kish):     {ne_raw:.2f} of {n}")
    print(f"  largest provenance share:   {share:.0%}")
    if share > 0.5:
        print("  ⚠ ONE GROUP HOLDS THE MAJORITY. The shared-error statistic below cannot see it:")
        print("    a bloc that IS the plurality never deviates from it, so it is never penalised,")
        print("    while the independent minority is. Do not trust the weights in this regime.")

    # ── 2. the repair ─────────────────────────────────────────────────────────
    best_single = max(
        sum(votes[t][i] == truth[t] for t in test_items) / len(test_items) for i in range(n)
    )
    learned = learn_independence_weights([votes[t] for t in history_items], n)
    naive_acc, naive_conf = evaluate(test_items, votes, truth, naive_weights(n))
    learn_acc, learn_conf = evaluate(test_items, votes, truth, learned)

    print(f"\n=== 2. Does the panel beat its own best member? (held-out, n={len(test_items)}) ===")
    print(f"  {'':22} {'accuracy':>9} {'confidence':>11} {'overconfidence':>15}")
    print(f"  {'best single voter':22} {pct(best_single):>9} {'—':>11} {'—':>15}")
    print(f"  {'plain majority':22} {pct(naive_acc):>9} {pct(naive_conf):>11} "
          f"{pct(naive_conf - naive_acc):>15}")
    print(f"  {'independence-weighted':22} {pct(learn_acc):>9} {pct(learn_conf):>11} "
          f"{pct(learn_conf - learn_acc):>15}")
    verdict = "below" if naive_acc < best_single else "at or above"
    print(f"  → The plain majority lands {verdict} the best single voter.")
    print(f"  → Weighting moves accuracy by {100 * (learn_acc - naive_acc):+.1f} pp and "
          f"overconfidence by {100 * ((learn_conf - learn_acc) - (naive_conf - naive_acc)):+.1f} pp.")

    # ── 3. the boundary ───────────────────────────────────────────────────────
    print("\n=== 3. The limit: a panel that is one model throughout ===")
    for model, names in by_model.items():
        subset = [i for i, p in enumerate(panel) if p["model"] == model]
        if len(subset) < 3:
            continue
        w_struct = structural_weights([panel[i]["model"] for i in subset])
        a_naive, c_naive = evaluate(test_items, votes, truth, naive_weights(n), subset)
        full = naive_weights(n)
        for k, i in enumerate(subset):
            full[i] = w_struct[k]
        a_w, c_w = evaluate(test_items, votes, truth, full, subset)
        inert = abs(a_naive - a_w) < 1e-12 and abs(c_naive - c_w) < 1e-12
        tag = "INERT — weighting changed nothing" if inert else "changed the outcome"
        print(f"  {len(subset)}x {model:22} acc {pct(a_naive)} → {pct(a_w)}   {tag}")
    print("  → Provenance weighting needs voters from different sources. A same-model panel")
    print("    cannot be repaired by reweighting; it has to be made diverse.")
    print(f"  → Diversity is necessary, not sufficient: cross-model pairs here still agree on")
    print(f"    the wrong answer {100 * cross_r:.1f}% of the time, and this panel of four families is")
    print(f"    worth {ne_raw:.1f} effective votes out of {n}.")

    print("\n--- scope -------------------------------------------------------------")
    print("  57 answerable items, 4 model families, one judge. This is a screen, not a")
    print("  benchmark: it shows the mechanism on one panel, and claims no ranking between")
    print("  models. Your panel will differ — run it on your own vote history.")
    print("  Data 10.5281/zenodo.21773065 · measurement 10.5281/zenodo.21775275\n")


if __name__ == "__main__":
    main()
