# Remediation Phase 5 — rho fit + simulator rebuild

Recorded: 2026-07-13, branch `fix/scientific-audit`.

## Dixon-Coles rho (scoreline layer)

Recipe rewritten to canonical inputs (`scorelines.fit_rho_from_data`): the old
fit merged the raw pre-dedupe file by (date, home, away) key — cross-joining
the kept Tahiti double-header — while the new fit aligns the canonical loader
1:1 with the canonical features.csv (asserted). Poisson-regression lambdas on
the Elo gap (+ home-advantage term for non-neutral rows), tau MLE on the four
low-score cells.

**Fitted rho = −0.050006 on 49,404 canonical pre-cutoff matches** (cell
counts: 0-0 ×3,965, 0-1 ×3,455, 1-0 ×5,098, 1-1 ×4,908). The frozen
`DIXON_COLES_RHO = -0.05` is the fit after rounding — the constant stands, now
as a measurement, not a convention. Artifact: `reports/scoreline_rho_fit.csv`
(header records date, canonical features hash, recipe). Refit is re-run by
`tests/test_simulator.py`. `models/model-manifest.json` regenerated with the
new `dixon_coles_rho` field (bundle hash unchanged — no retrain; rho is
bundle-adjacent, not a training input).

## What e12cb98 actually implemented (reported before fixing)

Commit e12cb98 ("Fix group tie-break: shuffle only teams still tied after
head-to-head", 2026-06-19): primary sort by the OVERALL (pts, gd, gf) triple;
teams tied on all three form a cluster ranked by an H2H mini-table of points
then GD only; runs still tied after that are shuffled. Defects vs the FIFA
2026 regulations: the mini-table omitted goals-scored-among-tied (criterion
f); no reapplication of the among-tied block to subsets that remain tied
after a partial separation; and `resolve_r32`'s third-place ranking used a
stable sort on (pts, gd, gf) whose ties fell through to dict insertion order
— i.e. GROUP-LETTER order, silently favouring early-alphabet groups in every
simulation.

## Tiebreak rebuild (simulate._rank_group / _rank_tied)

Implemented as an ordered criteria hierarchy: overall points → overall GD →
overall GF → among tied teams: points → GD → GF (mini-table over the recorded
scorelines), REAPPLIED recursively to any subset still tied after a partial
separation → documented RANDOM DRAW standing in for the fair-play-conduct and
drawing-of-lots criteria a simulator cannot know. Third-place ranking
(points → GD → GF → random draw) never falls through to group-letter order.

**Flag for Checkpoint 5:** the phase instruction paraphrased the order as
"H2H … then overall GD/goals", but the FIFA 2026 regulations rank overall
GD/GF ABOVE the head-to-head block (as in 2018/2022; UEFA is the H2H-first
body). The regulation order is implemented, verified by
`test_overall_gd_ranks_above_h2h`; the criteria are an ordered list, so
flipping is a one-line change if ruled otherwise.

## Scoreline sampling (the _score_for_outcome kill)

`_score_for_outcome` — outcome-independent Poisson score cosmetics whose
draws averaged 1-1 regardless of teams and whose margins ignored the model —
is deleted. Scorelines now sample from the Dixon-Coles grid CONDITIONAL on
the sampled W/D/L (`sample_scoreline`: restrict the grid to the outcome's
region, renormalise, sample a cell), with the grid solved per unordered pair
from the predictor's own W/D/L and cached (`_scoreline_grid_for`). Points,
GD and GF all derive from the one sampled scoreline. Law-of-total-probability
check: outcome-then-conditional-scoreline reproduces the grid's own cell
probabilities (tested on the 0-0 cell, 40k draws, 4σ band).

## Tournament-state conditioning (the Spain fix)

`simulate_tournament(..., state=...)` / `monte_carlo(..., state=...)`: played
matches enter as FACTS (group results with real scores; knockout winners per
slot; optionally the REAL R32 pairings, since reality has already resolved the
tiebreak randomness a re-simulation would re-sample), only remaining fixtures
simulate, ratings stay frozen at the pre-tournament snapshot. Output is
labelled "remaining-tournament forecast". The mechanism is fully tested with
synthetic states; it was NOT wired to live WC 2026 results this phase (hard
boundary) — the serving cutover phase connects it to the tracker's recorded
results.

## Verification

`tests/test_simulator.py` (12 tests): decisive outcomes always carry
consistent scores (1,500 conditional draws per outcome class); marginals
match the grid; group tables reconstruct from integer scores; FIFA order
(overall GD above H2H); H2H breaks full overall ties; reapplication resolves
the cyclic three-way tie deterministically (Z>X>Y for every seed); random
fallback reachable and actually random; third-place qualifiers vary across
seeds under full ties (no letter-order); completed results enter as facts;
fully-conditioned tournament is seed-independent; seed determinism; rho refit
matches the constant. Full suite 46/46; smoke test all checks pass (~19 s —
the grid cache keeps the Monte Carlo fast). The smoke Monte Carlo's top team
shifted (Spain → Argentina at n=100) — expected under the corrected
scoreline-driven tiebreaks and de-biased third-place ranking; simulator
outputs remain non-quotable until serving cutover.

Figure 8 untouched (Phase 7). Article untouched. No WC 2026 data read.
