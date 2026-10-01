# voteworth

**Ten judges. Worth 1.48 independent votes.**

Your LLM panel agrees — that is not the same as it being right.

When several LLM voters, judges or samples vote on an answer, the usual move is a plain
majority, and the level of agreement gets reported as confidence. Both steps assume the voters
are independent. They often are not — and when they are not, a correlated bloc can win the
vote while being wrong, with its agreement read as certainty.

This package measures that dependence three ways, corrects the vote where it can, and tells
you plainly where it cannot.

```bash
pip install voteworth
python -m voteworth.demo
```

Or from a clone of this repository — this is also the way to run the test suite, which is not
part of a wheel install:

```bash
pip install -e ".[dev]"
python -m voteworth.demo      # or: python examples/quickstart.py
pytest                        # the suite, including the mutation gate
```

## What it prints

On the bundled vote matrix — ten real LLM voters over 80 items, no network, no API key:

```
=== 1. When two voters are both wrong, do they agree on the wrong answer? ===
  same model       42.8%   (8 pairs)
  different model  18.5%   (37 pairs)

  The same question under three choices of base and definition:
  (all pooled over events)                       same   cross     gap   ratio
  both deviate from plurality (used here)       0.438   0.200  +0.237   +119%
  both wrong                                    0.643   0.462  +0.181    +39%
  raw agreement, all items                      0.721   0.624  +0.097    +16%

  Third source — averaging. The headline above means over voter PAIRS (0.428 / 0.185);
  pooling over events gives 0.438 / 0.200, because pairs contribute unequally.
  → The ABSOLUTE gap is stable across all of this; the ratio is not. Published
    absolute gaps sit in the same band. Match base, definition and averaging
    before comparing your panel to anyone else's number.

  effective votes (Kish):     1.48 of 10
  largest provenance share:   30%

=== 2. Does the panel beat its own best member? (held-out, n=29) ===
                          accuracy  confidence  overconfidence
  best single voter          58.6%           —               —
  plain majority             44.8%       62.1%           17.2%
  independence-weighted      51.7%       61.8%           10.1%
  → The plain majority lands below the best single voter.
  → Weighting moves accuracy by +6.9 pp and overconfidence by -7.2 pp.

=== 3. The limit: a panel that is one model throughout ===
  3x gpt-4o-mini            acc  37.9% →  37.9%   INERT — weighting changed nothing
  3x gpt-4.1-mini           acc  51.7% →  51.7%   INERT — weighting changed nothing
  → Provenance weighting needs voters from different sources. A same-model panel
    cannot be repaired by reweighting; it has to be made diverse.
  → Diversity is necessary, not sufficient: cross-model pairs here still agree on
    the wrong answer 18.5% of the time, and this panel of four families is
    worth 1.5 effective votes out of 10.
```

Four things worth sitting with:

**Ten voters were worth 1.48.** Kish's effective sample size, `n_eff = m/(1+(m−1)·r̄)`, counts
how many independent voters you actually have. A published nine-judge panel spanning seven
model families measures about 2.2. Adding voters from the same pool buys far less than the
count suggests.

**The plain majority scored 13.8 points below the panel's best single member**, and reported
62.1% confidence while being right 44.8% of the time. Agreement was measuring redundancy, not
correctness.

**Read the absolute gap, not the ratio.** Same-model pairs here agree on the wrong answer
2.3× as often as cross-model pairs — and that ratio is mostly an artefact of how it was
measured. On this one panel, without changing a single vote, three choices move it:

| base and definition (all pooled over events) | same | cross | gap | ratio |
|---|---|---|---|---|
| both voters **deviate from the plurality** — what this package measures | 0.438 | 0.200 | +0.237 | +119% |
| both voters are **wrong** | 0.643 | 0.462 | +0.181 | +39% |
| **raw agreement** over all items | 0.721 | 0.624 | +0.097 | +16% |

Three separate things vary here, and comparisons across studies routinely confuse them: the
**base** (which items count), the **definition** (what "agreeing" means on them) and the
**averaging** (the headline 42.8 / 18.5 is a mean over voter *pairs*; pooling over events gives
0.438 / 0.200, because pairs contribute unequally). The absolute gap stays between +0.097 and
+0.237 through all of it; the ratio swings by a factor of seven. Published measurements sit in
the same absolute band (+0.05 to +0.11). **The difference travels between studies; the ratio
does not.** Match all three choices before comparing your panel to a published number — the
demo prints the whole table so you can.

**On a single-model panel the provenance correction does nothing at all** — the demo prints
`INERT`. Not "a little": every voter carries the same weight, normalisation makes it uniform,
and the result is bit-identical to the naive vote.

## Using it on your own panel

```python
from voteworth import (
    effective_votes, bloc_share, learn_independence_weights,
    aggregate, agreement_confidence, pairwise_coerror,
)

history = [[0, 0, 1, 0], [1, 1, 0, 2], ...]   # past items, one cluster id per voter (-1 = abstained)
sources = ["gpt-4o-mini", "gpt-4o-mini", "claude-haiku-4-5", "mistral-small-latest"]

effective_votes(history, n_voters=4)   # how many independent voters you really have
bloc_share(sources)                    # ⚠ check this FIRST — see the boundary below
pairwise_coerror(history, n_voters=4)  # which pairs share their errors (NaN = never co-deviated)

weights = learn_independence_weights(history, n_voters=4)
winner  = aggregate([0, 0, 1, 2], weights)
conf    = agreement_confidence([0, 0, 1, 2], weights)
```

No vote history yet? Use what you already know about where the voters came from:

```python
from voteworth import structural_weights
structural_weights(["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini", "claude-haiku-4-5"])
# [0.333, 0.333, 0.333, 1.0]  — the three-voter bloc now counts once
```

Votes are cluster ids. For free text, turn them into ids first — `cluster_answers` (embedding
cosine, good for paraphrase) or `cluster_answers_by_fn` (an equality callback, usually an LLM
judge — the right choice for short factual answers, where embeddings merge two different dates).

## How the weight is computed

For every pair of voters, look only at items where **both deviated** from the panel's
plurality, and count how often they deviated **with the same value**. Then

```
weight[i] = 1 / (1 + Σ_{j≠i} sharedError[i][j])
```

A voter that never co-deviates keeps weight 1.0; one locked to a single partner drops to 0.5.
An empty history yields uniform weights — nothing observed, nothing claimed.

## ⚠️ The boundary: when one bloc holds the plurality, this method is blind

**Check `bloc_share` before you trust the weights.** The shared-error rate is measured among
voters that *deviate from the plurality*. A bloc large enough to **be** the plurality never
deviates from it, so it never registers as correlated — and the independent minority, which
does co-deviate, is penalised instead. The correction inverts.

Measured on a synthetic ten-voter panel with a six-voter bloc that is wrong together 45% of
the time, across twenty seeds: the bloc keeps **full weight 1.00** in 20/20 runs, the
independents drop to 0.43, the weighted vote is **identical to the naive one** in 20/20, and
both land a median **17 points below the best single voter**. The test that asserts this
failure is in the suite (`test_dominant_bloc_is_invisible_to_the_weights`) — it is a
documented boundary, not a bug awaiting a fix.

`effective_votes` still sees it: n_eff keeps falling as the bloc grows (1.91 → 1.68 → 1.55 →
1.26 for blocs of 3, 5, 6, 8 of 10) while the weights go flat. **A low n_eff beside uniform
weights is the signature.** The demo prints both and warns above 50%.

## What this package does not do

- **It cannot repair a dominated panel** — see the boundary above.
- **It is not a benchmark and ranks no models.** The bundled matrix is 57 answerable items, 4
  model families, one judge. A screen, not a benchmark. Your numbers will differ.
- **It does not make a bad panel good.** On the demo panel the weighted accuracy of 51.7% is
  still below the best single voter's 58.6%. The honest conclusion there is that voting was
  the wrong move, and the tool says so.
- **It is not a calibration method.** The overconfidence figures above come from the held-out
  half (n=29, no interval). On the full 57-item set, the note this package cites reports the
  same learned weighting leaves overconfidence at **15.3 points, from 15.3** — the provenance
  weight moves it to 14.9. Read the −7.2 pp above as a direction, not a size. **Decision
  repair and calibration repair are different things**, and this package only does the first.
- **Diversity is necessary, not sufficient.** Cross-model pairs in the demo still agree on the
  wrong answer 18.5% of the time, and a nine-judge panel of seven families measures ~2.2
  effective votes. Measure your panel; do not assume it from the roster.

## Is the test suite real?

A passing suite proves nothing on its own; it proves something once you have seen it fail for
the right reason. `tests/test_mutation.py` breaks the mechanism three ways — kills the
weighting, reverses the documented tie-break, blinds the shared-error detector — and requires
the suite to catch each one, with an unmutated control run so that failures cannot come from
the harness itself. `tests/test_readme_claims.py` parses the numbers out of *this file* and
checks them against a live run, so a stale README is a red test.

Both have already paid for themselves: the control arm caught an assertion demanding exact
float equality where only equality-to-rounding holds, and the withdrawn-figure guard was
found matching its own source text.

```bash
pytest          # 157 tests, a few seconds, including the mutation gate
```

## Related work

The phenomenon is established and the size of it is contested — the package takes no side.

**Same-model error correlation.** Kim, Garg, Peng & Garg, *Correlated Errors in Large Language
Models* (ICML 2025, [arXiv:2506.07962](https://arxiv.org/abs/2506.07962)) measure exactly this
package's primitive across 350+ models and find a significant same-developer and
same-architecture effect — **and** that cross-provider pairs already agree 0.42–0.60 when both
are wrong. Kohli, *Nine Judges, Two Effective Votes* ([arXiv:2605.29800](https://arxiv.org/abs/2605.29800))
finds the family effect small and the most correlated pairs *cross*-family. Goel et al.,
*Great Models Think Alike* ([arXiv:2502.04313](https://arxiv.org/abs/2502.04313)) propose CAPA,
a chance-corrected similarity, and report that judges favour models similar to themselves.

**Panels.** Verga et al., *Replacing Judges with Juries* ([arXiv:2404.18796](https://arxiv.org/abs/2404.18796))
is the reference for panels of disjoint model families.

**Agreement is not accuracy.** Ding ([arXiv:2607.08065](https://arxiv.org/abs/2607.08065))
audits 265k samples; Mukherjee et al. ([arXiv:2606.03043](https://arxiv.org/abs/2606.03043))
show inter-judge consensus is not human alignment; Wang et al. ([arXiv:2203.11171](https://arxiv.org/abs/2203.11171))
introduced self-consistency.

**Dependence-aware aggregation, with a model fit.** Balasubramanian et al.
([arXiv:2601.22336](https://arxiv.org/abs/2601.22336)) use Ising couplings; *CARE*
([arXiv:2603.00039](https://arxiv.org/abs/2603.00039)) separates latent confounders; Jaffe et
al. ([arXiv:1510.05830](https://arxiv.org/abs/1510.05830)) detect conditional-independence
violations without labels. These fit a model. A fit need not be iterative: FlyingSquid (Fu et
al., ICML 2020, [arXiv:2002.11955](https://arxiv.org/abs/2002.11955)) solves a latent-variable
label model in closed form from unlabelled votes, given a dependency graph that the user
supplies or estimates beforehand; Firebolt (Kuang et al., AISTATS 2022,
[PMLR](https://proceedings.mlr.press/v151/kuang22a.html)) learns the class balance and
class-specific accuracies of such a model by one least-squares solve and closed-form steps from
the votes' means and their two- and three-way covariances, given a user-supplied dependency
graph; and CROWDLAB (Goh et al., [arXiv:2210.06812](https://arxiv.org/abs/2210.06812)) weights
annotators in closed form by their agreement, with a trained classifier's probabilities as a
required input. This package fits no latent model and no classifier, and its learned weight
takes no dependency graph: it is a function of how often two voters deviate from the plurality
*with the same value*.

**Classical.** Weighted majority under dependence: Nitzan & Paroush (1982, 1984), Shapley &
Grofman (1984); jury theorems with correlated votes: Ladha (1992), Boland (1989); ensemble
diversity and the limits of majority voting: Kuncheva & Whitaker (2003), Kuncheva et al.,
*Limits on the Majority Vote Accuracy* (2003), Hansen & Salamon (1990); discounting sources
that share false values: Dong, Berti-Équille & Srivastava (PVLDB 2009); truth inference:
Dawid & Skene (1979), Zheng et al. (PVLDB 2017).

**Judges as instruments.** A separate line treats judges the way educational measurement
treats human raters. Sunkavalli, *LLM Judges as Raters*
([arXiv:2608.29517](https://arxiv.org/abs/2608.29517)) runs a pre-registered rater-effects
battery over 2,377 essays and 12 judges: severity spans 219 points on a 0–1000 scale, and
judge-human correlations sit in an undiscriminating .47–.56 band. It measures each judge
against humans rather than aggregating a panel, so it neither overlaps nor competes with
this package — but if your judges disagree, its vocabulary (severity, halo, drift) names
causes that correlation alone cannot.

**The opposite move.** Flynt, *GroundEval*
([arXiv:2606.22737](https://arxiv.org/abs/2606.22737)) argues for replacing LLM judges with
deterministic checks on what an agent actually fetched and cited, rather than hardening the
panel. Worth reading before you invest in any panel at all: where a deterministic check is
available, it beats every weighting scheme including this one.

**Tools that build panels without measuring them.** `llm-judge-panel` runs multi-provider
consensus with unanimous / majority / at-least-*m* rules and no correlation term at all;
`llm-judge-kit` offers a typed judge primitive; `judgesync` aligns judges to human preferences.
None of those three asks whether the votes being counted are independent. That is a statement
about three packages read at source in September 2026, not about the field — a prior-art sweep
of this package's own claims initially missed all three, because the PyPI search sits behind a
bot challenge and the sweep fell back to looking up names instead of descriptions. Treat every
"no tool does X" here, including this one, as *not found by a search whose limits are on
record*.

**Neighbouring tools.** `judgepanel` fits Dawid–Skene to judge panels and reports κ, but
assumes conditional independence. *CARE* corrects for latent confounders with a fitted
decomposition. Inspect AI's `multi_scorer` offers mode/majority reducers and a Krippendorff
statistic, unweighted. This package sits in a different spot: a closed-form weight from shared
deviations, no labels, no latent model and no classifier, one dependency — and its own failure
case shipped in the tests.

## What you can check yourself, and what you have to take on trust

Asked as a downloader would, with nothing but the files in this package:

**Where these run.** Everything below runs from the source: a clone of the repository, or the
source distribution (`pip download --no-binary :all: voteworth`, then unpack it). A plain
`pip install voteworth` gives you a wheel, which carries the package, the licence and the
`NOTICE` — but not the test suite, so the first three rows have nothing to run there. That is
normal for a wheel and was worth saying: an earlier version of this table promised the checks
without saying which artefact they live in.

**Verifiable offline, from what ships here:**

| | how |
|---|---|
| every number in this README | `pytest tests/test_readme_claims.py` parses them out of the file and compares them to a live run |
| every behavioural claim | `pytest` — 157 tests, including the boundary case where this method fails, and the panel shape that makes the demo's own warning fire |
| that the tests can fail | `pytest tests/test_mutation.py` breaks the mechanism three ways and requires each to be caught |
| that the bundled dataset is unaltered | its SHA-256 is in `NOTICE`, together with the one command that reaches a matching hash on Zenodo — the record page itself lists an MD5, under a different filename |

**Not verifiable from this package — you are trusting us, or the cited record:**

- The three DOIs resolve to what we say they do. Read them; they are open access.
- The statements about other people's work in *Related work*. The recent papers carry an arXiv id or a proceedings link so you can check the claim against the source rather than against us; the classical references are cited by author and year, and the tools are named, not linked.
- That the prior-art search behind "none of those three" was run as described. Its limits are stated in that paragraph, including a gap it initially had.
- **The account this package gives of its own history.** Several passages here, in `NOTICE`
  and in the test docstrings say how a check was once found wanting, or what an earlier
  version got wrong. That is our report on our own process, and nothing that ships here can
  establish it. It is kept because it says why a guard has the shape it has — read it as the
  reason for a design, not as evidence for a result. What the guards *do* is in the table
  above, and that part you can run.
- The panel behind the bundled votes was run as the cited note describes. The note reports its own scope: 57 answerable items, 4 model families, one judge.

## Provenance

| | |
|---|---|
| Measurement note | [10.5281/zenodo.21775275](https://doi.org/10.5281/zenodo.21775275) — *Reweighting Correlated Voters Does Not Repair Agreement Confidence* |
| Vote matrix (bundled here) | [10.5281/zenodo.21773065](https://doi.org/10.5281/zenodo.21773065) — CC BY 4.0 |
| Independence specification | [10.5281/zenodo.22012189](https://doi.org/10.5281/zenodo.22012189) — *Four Independence Invariants for Judgment-Quality Measurement* |

## License

Apache-2.0. Bundled vote matrix: CC BY 4.0, cite the DOI above.
