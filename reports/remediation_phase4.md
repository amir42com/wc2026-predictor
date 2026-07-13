# Remediation Phase 4 — canonical frozen-state benchmark (WC 2014/2018/2022)

Recorded: 2026-07-13, branch `fix/scientific-audit`.

## Protocol

The benchmark runs `blend_weight_search.fold_frozen_predictions` VERBATIM
(imported by `backtest.frozen_fold_run`) — the exact evaluator validated on
the 2006/2010 tune folds in Phase 3. Per fold: fresh model on canonical
matches strictly before the tournament's first fixture; ONE frozen
pre-tournament team-state/H2H snapshot (no within-tournament updates);
recorded venue flags (host rows single-orientation + HFA_ELO 125.58: Brazil
2014 ×7, Russia 2018 ×5, Qatar 2022 ×3); production prior (draw 0.227). The
Elo baseline runs on the SAME frozen ratings with the same HFA on the same
rows and its fold-specific draw rate. Fold best_iterations: 145/176/123.

`backtest.PROTOCOL_COMPLETE` was set True in the same change that made the
frozen evaluator the default path. The rolling evaluation survives only as
the explicitly labelled sensitivity variant inside
`src/protocol_comparison.py` / `backtest.deployed_model_and_blend`.

## Headline numbers — old vs new (Combined, 192)

| System | Accuracy old → new | Log-loss old → new | Brier old → new |
| --- | --- | --- | --- |
| Raw XGBoost | 56.25% → **53.65%** | 0.9762 → **0.9862** | 0.5758 → **0.5841** |
| Blend | 56.25% → **53.13%** | 0.9720 → **0.9818** | 0.5735 → **0.5816** |
| Elo baseline | 57.81% → **56.25%** | 0.9769 → **0.9859** | 0.5770 → **0.5836** |

Paired tests (10,000 resamples, seed 12345, via paired_tests.py):

| Statistic | Old | New |
| --- | --- | --- |
| McNemar exact (Blend vs Elo) | p = 0.5811 (b=5, c=8) | **p = 0.2101 (b=5, c=11)** |
| Bootstrap LL diff (Blend−Elo) | −0.0049 [−0.0259, +0.0175] | **−0.0041 [−0.0252, +0.0181]** |

Per tournament (new): Elo wins accuracy in ALL three folds (59.4/56.2/53.1 vs
blend 56.2/53.1/50.0). Blend has lower log-loss in 2014 and 2022; Elo lower
in 2018.

## Does "didn't beat Elo" still hold?

**Yes — and more starkly.** Under the corrected protocol the blend's accuracy
deficit vs Elo widens from 1.56pp to 3.13pp (102 vs 108 correct of 192), the
discordant picks now split 11–5 in Elo's favour (was 8–5), and Elo leads
accuracy in every individual tournament. Still not statistically significant
(McNemar p = 0.21; the blend's log-loss edge of −0.004 has a bootstrap CI
straddling zero). The published claim's direction survives; its magnitude was
flattered by the old protocol. Everything got worse in absolute terms because
the frozen snapshot is strictly less information than in-tournament rolling
state — that is the honest deployment condition.

## Protocol decomposition (reports/evaluation_protocol_comparison.csv)

Blend accuracy path: 56.25 (A published) → 56.25 (B0: canonical data,
legacy inference) → 54.69 (B1: + venue contract) → 53.13 (C: + frozen state).

* **A→B0 data/ordering delta: ~zero.** Accuracy identical for all three
  systems; log-loss moves < 0.003. The Phase 1 nondeterminism perturbed
  features broadly but the benchmark metrics barely felt it.
* **B0→B1 venue-contract delta: −1.0 to −1.6pp accuracy, +0.007 LL (all
  systems).** Modelling real host advantage made the 15 host rows HARDER, not
  easier — host-favourite priors were often wrong in these tournaments
  (Brazil's 2014 collapse; Qatar losing all three). This is the cost of
  honesty, borne symmetrically by blend and baseline (information parity).
* **B1→C state-protocol delta: −1.5pp accuracy, +0.004 LL.** The rolling
  evaluation was leaking in-tournament form into "pre-match" features.

## Diagnostics (Combined, 192; in backtest_metrics_summary.csv)

* Wilson 95% CIs: XGB [46.6%, 60.6%], Blend [46.1%, 60.1%], Elo [49.2%,
  63.1%] — heavily overlapping.
* ECE (10 equal-width bins, top-class confidence): Blend 0.040 (best),
  XGB 0.047, Elo 0.054.
* Draw picks: XGB 1, Blend 0, Elo 0 — of 41 actual draws. No system
  meaningfully predicts draws; a known class-imbalance pathology, now on the
  record.
* Picks-flip (audit C6, benchmark): Blend vs raw XGB differ on **2 of 192**
  hard picks (0/1/1 by fold) — the blend is probabilistic smoothing, not a
  different predictor. Blend vs Elo differ on 17 of 192.

## HFA per-fold refit sensitivity (promised at Phase 2)

Refit on each fold's own training window: 2014 = 130.2, 2018 = 128.6,
2022 = 127.8 (production constant 125.58 — the era-declining trend continues
into the constant's favour). Metric impact: Blend LL 0.98189 vs 0.98182
(+0.00007); Elo accuracy 55.73% vs 56.25% (one pick). The HFA window choice
is immaterial to every benchmark conclusion.

## Phase 8 directives (recorded at Checkpoint 4 approval)

1. The asymmetric protocol effect (blend −3.12pp vs Elo −1.56pp accuracy)
   gets its own NAMED paragraph in the article: the rolling evaluation
   specifically flattered the recency-feature model. A finding, not a
   footnote.
2. The p=0.58 → 0.21 shift is protocol change, not evidence accumulation —
   the article must not narrate it as a trend toward significance. The
   zero-draws claim updates to 1/0/0 (XGB/Blend/Elo hard draw picks of 192,
   41 actual draws).

## Not done here (by design)

Figures (Phase 7) — `figure5_by_tournament.png` and the make_figures set were
NOT regenerated; `backtest_by_tournament.csv` was rebuilt via the table
functions only. Article untouched. No WC 2026 data of any kind was read.
`make_by_tournament.py`'s hardcoded 2014/2018 narrative text is now stale
against the new numbers — flagged for Phase 7.
