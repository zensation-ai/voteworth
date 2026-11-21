# voteworth

**Ten LLM voters, 1.48 effective votes by raw agreement.**

Your LLM panel agrees — that is not the same as it being right.

When several LLM voters, judges or samples vote on an answer, the usual move is a plain
majority, and the level of agreement gets reported as confidence. Both steps assume the voters
are independent. They often are not — and when they are not, a correlated bloc can win the
vote while being wrong, with its agreement read as certainty.

This package reports three quantities, none of which needs labels and none of which isolates
that dependence on its own: how often two voters that both break from the plurality pick the
same answer, Kish's effective number of votes by raw agreement, and the share of the panel's
largest source. It corrects the vote where it can, and tells you plainly where it cannot.

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
=== 1. When two voters both break from the plurality, do they pick the same answer? ===
  same model       42.8%   (8 pairs)
  different model  18.5%   (37 pairs)

  The same question under three choices of base and definition:
  (all pooled over events)                       same   cross     gap   ratio
  both deviate from plurality (used here)       0.438   0.200  +0.237   +119%
  both wrong                                    0.643   0.462  +0.181    +39%
  raw agreement, all 57 items                   0.721   0.624  +0.097    +16%

  Third source — averaging. The headline above means over voter PAIRS (0.428 / 0.185);
  pooling over events gives 0.438 / 0.200, because pairs contribute unequally.
  → The ABSOLUTE gap moves less than the ratio across all of this. Published
    studies measure related but different quantities: match base, definition
    and averaging before comparing your panel to anyone else's number.

  effective votes (Kish, raw agreement):   1.48 of 10
    on the 57 items with a gold answer; on all 80 items: 1.49
  independent voters with these accuracies, if every item were equally hard:
    wrong answers coinciding as on each item:   2.03
    no two wrong answers alike:                 2.35
  items all 10 voters got right:           20 of 57
    independent voters with these accuracies on equally hard items: 0.34
  → Raw agreement rises with accuracy once voters are right more often than not, with
    differences in item difficulty and with shared errors; n_eff alone cannot tell them apart.
  largest provenance share:                30%

=== 2. What does weighting change on held-out items? (n=29) ===
                          accuracy  confidence  overconfidence
  best voter, post hoc       58.6%           —               —
  plain majority             44.8%       62.1%           17.2%
  independence-weighted      51.7%       61.8%           10.1%
  → The best voter is picked on these same items, after the fact: no rival you could
    have chosen in advance.
  → Weighting moves accuracy by +6.9 pp and overconfidence by -7.2 pp.

=== 3. The limit: a panel that is one model throughout ===
  3x gpt-4o-mini            acc  37.9% →  37.9%   INERT — weighting changed nothing
  3x gpt-4.1-mini           acc  51.7% →  51.7%   INERT — weighting changed nothing
  → Provenance weighting needs voters from different sources. A same-model panel
    cannot be repaired by provenance weighting; it has to be made diverse.
```

Four things worth sitting with:

**Ten voters came to 1.48 effective votes by raw agreement, on the 57 items with a gold answer
(1.49 on all 80).** Kish's effective sample size, `n_eff = m/(1+(m−1)·r̄)`, here takes raw
pairwise agreement as r̄, and raw agreement rises with differences in difficulty between items,
with shared errors, and with accuracy once voters are right more often than not. On equally hard
items, ten independent voters that are each right 80% of the time and never give the same wrong
answer also score 1.48, with every learned weight at 1.0. With two answer options, where every
wrong answer is the same one, they score 1.40, and every weight drops to 0.10.

The demo prints what independent voters with this panel's accuracies would score if every item
were equally hard: 2.35 if no two wrong answers coincide, 2.03 if wrong answers coincide as they
do on each item. The second figure counts an item without wrong answers as no coincidence and an
item on which only one voter is wrong as full coincidence. Nor do these votes fit independent
voters on equally hard items: all ten voters got 20 of the 57 right, where such voters with these
accuracies would get 0.34. Unequal items and voters that are right together both produce that
excess, and the votes cannot say which. Item by item, a model fitted to these same votes accounts
for all of the panel's mean agreement, as it would for any panel without abstentions, so n_eff
alone cannot tell differences in item difficulty from shared errors. A lower n_eff does not mark
the more dependent panel. A nine-judge panel spanning seven model families is reported at about
2.2 (Kohli, preprint), but from the correlation of error indicators against gold labels rather
than from raw agreement, so the two figures are not on one scale.

Voters from the same model that both break from the plurality pick the same answer more often
than voters from different models (42.8% against 18.5%, means over voter pairs). The items do not
explain that gap: shuffling the votes within each item keeps every item's answers and drops who
gave them, and in 2,000 such shuffles same-model pairs came out as far ahead less than once in a
thousand, in that rate (+0.243) or in raw agreement (+0.097). The shuffle needs no labels. So
adding voters from the same model buys less than the count suggests.

**On the held-out half the plain majority reported 62.1% confidence while being right 44.8% of
the time.** Agreement was not measuring correctness.

**Read the absolute gap, not the ratio.** Same-model pairs that both break from the plurality
pick the same answer 2.3× as often as cross-model pairs — and that ratio is mostly an artefact of how it was
measured. On this one panel, without changing a single vote, three choices move it:

| base and definition (all pooled over events) | same | cross | gap | ratio |
|---|---|---|---|---|
| both voters **deviate from the plurality** — what this package measures | 0.438 | 0.200 | +0.237 | +119% |
| both voters are **wrong** | 0.643 | 0.462 | +0.181 | +39% |
| **raw agreement** over all 57 items | 0.721 | 0.624 | +0.097 | +16% |

Three separate things vary here, and comparisons across studies routinely confuse them: the
**base** (which items count), the **definition** (what "agreeing" means on them) and the
**averaging** (the headline 42.8 / 18.5 is a mean over voter *pairs*; pooling over events gives
0.438 / 0.200, because pairs contribute unequally). The absolute gap stays between +0.097 and
+0.243 through all of it; the ratio swings between +16% and +131%, a factor of eight. Published
studies report related but different quantities: Kohli a same-family excess of +0.047 to +0.109
in the correlation of error indicators, Kim et al. a same-company coefficient of +0.022 to
+0.066 in agreement when both are wrong. Neither is this package's co-deviation rate, so neither
can be set beside the table above as it stands. Match all three choices before comparing your
panel to a published number; the demo prints the whole table so you can.

**On a single-model panel the provenance correction does nothing at all** — the demo prints
`INERT`. Not "a little": every voter carries the same weight, normalisation makes it uniform,
and the result is bit-identical to the naive vote.

## Using it on your own panel

```python
from voteworth import (
    effective_votes, bloc_share, learn_independence_weights,
    aggregate, agreement_confidence, pairwise_coerror,
)

history = [[0, 0, 1, 0], [1, 1, 0, 2], [0, 0, 2, 2], [1, 1, 1, -1]]   # past items, one cluster id per voter (-1 = abstained)
sources = ["gpt-4o-mini", "gpt-4o-mini", "claude-haiku-4-5", "mistral-small-latest"]

effective_votes(history, n_voters=4)   # Kish's n_eff by raw agreement, not a count of independent voters
bloc_share(sources)                    # ⚠ check this FIRST — see the boundary below
pairwise_coerror(history, n_voters=4)  # how often two voters give the same answer when both deviate (NaN = never both)

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
judge — the right choice for short factual answers, where embeddings can merge two different
dates).

## How the weight is computed

For every pair of voters, look only at items where **both deviated** from the panel's
plurality, and take the share of those items on which they deviated **with the same value**.
Then

```
weight[i] = 1 / (1 + Σ_{j≠i} sameAnswer[i][j])
```

A voter that never co-deviates keeps weight 1.0; one locked to a single partner drops to 0.5.
An empty history yields uniform weights — nothing observed, nothing claimed.

## ⚠️ The boundary: when one bloc holds the plurality, this method is blind

**Check `bloc_share` before you trust the weights.** The co-deviation rate is measured among
voters that *deviate from the plurality*. A bloc large enough to **be** the plurality never
deviates from it, so it never registers as correlated — and the independent minority, which
does co-deviate, is penalised instead. The correction inverts.

Measured on a synthetic ten-voter panel with a six-voter bloc that is wrong together 45% of
the time, across twenty seeds: the bloc keeps **full weight 1.00** in 20/20 runs, the
independents drop to 0.43, the weighted vote is **identical to the naive one** in 20/20, and
both land a median **17 points below the best single voter**. The test that asserts this
failure is in the suite (`test_dominant_bloc_is_invisible_to_the_weights`) — it is a
documented boundary, not a bug awaiting a fix.

In the same experiment n_eff keeps falling as the bloc grows (1.91 → 1.68 → 1.55 → 1.26 for
blocs of 3, 5, 6, 8 of 10), while the bloc's weight returns to full once it never deviates from
the plurality. **The signature is a group of voters that vote together, kept at full weight
while the others are down-weighted.** If the group at full weight is the independent one, the
weighting is doing its job. `bloc_share` tells you whether the largest provenance group holds
half the panel or more. A low n_eff comes with it but is no evidence on its own: accurate voters and
unequal items lower n_eff with every weight at full when no two voters give the same wrong
answer, and with few answer options independent voters score low with every weight down. The
demo prints n_eff and the largest provenance share and warns from 50% up; the weights come from
`learn_independence_weights`.

## What this package does not do

- **It cannot repair a dominated panel** — see the boundary above.
- **It is not a benchmark and ranks no models.** The bundled matrix holds 80 items, 57 of them
  answerable, from 4 models and one judge. The judge is gpt-4o-mini, the model of three of the
  ten voters. A screen, not a benchmark. Your numbers will differ.
- **It does not make a bad panel good.** On the demo panel the weighted vote is right on 51.7% of
  the 29 held-out items, against 44.8% for the plain majority. The demo also prints the best
  single voter on those items, but picks it after the fact, on the same items: it is no rival
  you could have chosen in advance, and it does not show whether voting was the right move.
- **It is not a calibration method.** The overconfidence figures above come from the held-out
  half (n=29, no interval). On the full 57-item set, the note this package cites reports the
  same learned weighting leaves overconfidence at **15.3 points, from 15.3** — the provenance
  weight moves it to 14.9. Read the −7.2 pp above as what happened on these 29 items, not as a
  direction: on the full set the learned weighting leaves overconfidence where it was. **Decision
  repair and calibration repair are different things**, and this package only does the first.
- **Diversity is not shown to be enough.** Cross-model pairs in the demo that both break from
  the plurality pick the same answer 18.5% of the time. That is one panel of four models; a
  nine-judge panel of seven families is reported at about 2.2 effective votes on a different
  measure (Kohli, preprint). Measure your panel; do not assume it from the roster.

## Is the test suite real?

A passing suite proves nothing on its own; it proves something once you have seen it fail for
the right reason. `tests/test_mutation.py` breaks the mechanism three ways — kills the
weighting, reverses the documented tie-break, blinds the co-deviation detector — and requires
the suite to catch each one, with an unmutated control run so that failures cannot come from
the harness itself. `tests/test_readme_claims.py` parses the decimal figures, the test count,
the span sentence and the headline out of *this file* and checks them against a live run or the
boundary experiment, or names the test that recomputes them, so a stale README is a red test.
Figures from cited work, including our own measurement note, are listed there with their source.

Both have already paid for themselves: the control arm caught an assertion demanding exact
float equality where only equality-to-rounding holds, and the withdrawn-figure guard was
found matching its own source text.

```bash
pytest          # 410 tests, under a minute, including the mutation gate
```

## Related work

The phenomenon is established and the size of it is contested — the package takes no side. The
statements below about other people's work were checked against their sources on 21 November 2026.

**Same-model error correlation.** Kim, Garg, Peng & Garg, *Correlated Errors in Large Language
Models* (ICML 2025, [arXiv:2506.07962](https://arxiv.org/abs/2506.07962)) measure, across 350+
models, how often two models give the same wrong answer when both are wrong: a labelled relative
of this package's label-free primitive. They find a significant same-company and
same-architecture effect, **and** that model pairs agree on average 0.42–0.60 of the time when
both are wrong (0.423 on the HuggingFace leaderboard, 0.6 on Helm, mean over all pairs). Kohli,
*Nine Judges, Two Effective Votes* (preprint, [arXiv:2605.29800](https://arxiv.org/abs/2605.29800))
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

**Classical.** Weighted majority and its behaviour under dependence: Nitzan & Paroush (1982,
1984), Shapley &
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
consensus with unanimous / majority / at-least-*m* rules and reports Fleiss' κ as inter-judge
reliability, without letting it change any vote's weight;
`llm-judge-kit` offers a typed judge primitive; `judgesync` aligns judges to human preferences.
None of those three asks whether the votes being counted are independent. That is a statement
about three packages read at source in September 2026 and again on 21 November 2026, not about
the field — a prior-art sweep
of this package's own claims initially missed all three, because the PyPI search sits behind a
bot challenge and the sweep fell back to looking up names instead of descriptions. Treat every
"no tool does X" here, including this one, as *not found by a search whose limits are on
record*.

**Neighbouring tools.** `judgepanel` fits Dawid–Skene to judge panels and reports κ. Dawid–Skene
assumes conditional independence; since version 0.2.0 (November 2026) the package also tests that
assumption. *CARE* corrects for latent confounders with a fitted
decomposition. Inspect AI's `multi_scorer` offers mode/majority reducers and a Krippendorff
statistic, unweighted. This package sits in a different spot: a closed-form weight from shared
deviations, no labels, no latent model and no classifier, one dependency — and its own failure
case shipped in the tests.

## What you can check yourself, and what you have to take on trust

Asked as a downloader would, with nothing but the files in this package:

**Where these run.** Everything below runs from the source, that is, a clone of the repository.
Installing from PyPI (`pip install voteworth`)
gives you the wheel, which carries the package, the licence and the
`NOTICE` — but not the test suite, so the first three rows have nothing to run there. That is
normal for a wheel and was worth saying: an earlier version of this table promised the checks
without saying which artefact they live in.

**Verifiable offline, from what ships here:**

| | how |
|---|---|
| every decimal figure in this README, the test count, the span of the absolute gap, the headline and the code examples | `pytest tests/test_readme_claims.py` parses them out of the file and compares them to a live run or to the boundary experiment, or names the test that recomputes them; figures from cited work, including our own measurement note, are listed there with their source |
| the behaviour of the ten public functions, including their error paths | `pytest` — 410 tests, including the boundary case where this method fails, and the panel shape that makes the demo's own warning fire |
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
- The panel behind the bundled votes was run as the cited note describes. The note reports its own scope as “one cached panel, 57 answerable items, four model families, one judging rubric”; this README counts the four as models.

## Provenance

| | |
|---|---|
| Measurement note | [10.5281/zenodo.21775275](https://doi.org/10.5281/zenodo.21775275) — *Reweighting Correlated Voters Does Not Repair Agreement Confidence* |
| Vote matrix (bundled here) | [10.5281/zenodo.21773065](https://doi.org/10.5281/zenodo.21773065) — CC BY 4.0 |
| Independence specification | [10.5281/zenodo.22012189](https://doi.org/10.5281/zenodo.22012189) — *Four Independence Invariants for Judgment-Quality Measurement* |

## License

Apache-2.0. Bundled vote matrix: CC BY 4.0, cite the DOI above.
