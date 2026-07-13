# Remediation Phase 2 — venue/inference contract

Recorded: 2026-07-13, branch `fix/scientific-audit`.
Approved at Checkpoint 2 (proposal) with three decisions: (1) the Elo baseline
receives the same venue information as the blend; (2) the schedule table is
drafted now, PROVISIONAL until an API cross-check and a manual fifa.com
spot-check both pass; (3) HFA is a fixed production constant fitted once on
Phase-1 canonical pre-cutoff data, with per-fold refits reported by Phase 4 as
a sensitivity line. Scope addition: backtest inference refuses to run until
the Phase 4 frozen-state protocol lands.

## What the contract replaces

`predict_neutral_proba` treated every match as neutral and symmetry-averaged
both orientations, while its consumers disagreed about venue: the backtest fed
recorded rows whose true neutral flags mark 15 of the 192 fold matches as host
matches (Brazil 2014 ×7, Russia 2018 ×5, Qatar 2022 ×3; `home_team == country`
verified on all); serving hardcoded `neutral=1` for every WC 2026 fixture,
including the Mexico/USA/Canada hosts; the Streamlit UI carried its own
hand-rolled home-advantage mode (coupled to `is_world_cup = int(neutral)`)
that existed nowhere else. The Elo prior was venue-blind everywhere.

## The contract (train.py)

* `MatchContext(home_team, away_team, neutral, venue_country=None,
  is_world_cup=True)` — frozen dataclass. Invariant: `neutral=False` ⇒
  `home_team` IS the side with home advantage (upstream martj42 semantics).
  `venue_country` is derivation evidence, never a model input.
* `predict_match_proba(model, build_x, ctx, elo_home, elo_away, blend_weight,
  draw_rate, hfa_elo)` — the single deployed inference. Neutral path: both
  orientations with `neutral=1`, averaged, symmetric prior (unchanged
  procedure). Non-neutral path: ONE orientation, host at home, `neutral=0`,
  prior shifted by `hfa_elo` (symmetry-averaging would erase exactly the
  advantage being modelled). `build_x` gains the neutral argument.
* `predict_neutral_proba` remains as a DEPRECATED shim over the neutral path —
  bit-identical outputs (tested), so the frozen serving path (Predictor,
  tracker, app) is untouched until the serving rebuild phase.

## HFA_ELO — fixed production constant

**HFA_ELO = 125.58** Elo points. Recipe (documented in `train.fit_hfa_elo`,
re-run by `tests/test_match_context.py`): maximum likelihood over the 36,346
non-neutral rows of the canonical features.csv (hashes in
`remediation_phase1.md`), home win probability = Elo expectation with the home
rating shifted by h; draws are uninformative under the prior's fixed draw
share, leaving 28,038 decided matches; h* maximizes the likelihood on
h ∈ [0, 400]. Disclosure mirrors ELO_DRAW_RATE: a frozen constant with
provenance, NOT tuned against any published metric. Era sensitivity (fit on
chronological halves): 132.1 / 117.9 — home advantage declines over time,
consistent with the literature; Phase 4 reports per-fold refits as the
formal sensitivity line.

## Population per consumer (staging)

| Consumer | Status this phase |
| --- | --- |
| Historical evaluation (`backtest.py`) | IMPLEMENTED: per-row `MatchContext` from the recorded neutral flag; the Elo baseline and the production blend prior apply the same fixed HFA on the same host rows (information parity). GATED: see below. |
| 2026 tracker (`fetch_results.py`) | DEFERRED to serving rebuild (frozen path). Will derive contexts from the schedule table. |
| Simulator (`simulate.py`) | DEFERRED to serving rebuild. Group fixtures via host rows; knockout slots via the slot→venue mapping, which lands WITH its own validation then (the R32 seq in the schedule table is deliberately NOT the `R32_SLOTS` index yet). Predictor cache key gains the home-advantaged side. |
| Streamlit UI | DEFERRED to serving rebuild; its checkbox maps onto MatchContext, retiring the hand-rolled duplicate. |

## PROTOCOL-INCOMPLETE gate

`backtest.PROTOCOL_COMPLETE = False`; `deployed_model_and_blend`,
`elo_baseline_proba`, `production_blend_prior` and `backtest()` all refuse to
run (RuntimeError naming the phase) until Phase 4 flips the flag. Verified
end-to-end: `python src/backtest.py` and `python src/export_backtest.py` exit
1 at the gate with no file written; `experiments.py` is covered through the
same functions. `make_by_tournament.py`/`paired_tests.py` only re-read the
baseline CSVs in `reports/` (Phase 0 artifacts), which stay untouched.

## Schedule table (data/wc2026_schedule.csv) — PROVISIONAL

104 rows (72 group / 16 R32 / 8 R16 / 4 QF / 2 SF / third place / final).
Drafting policy: **no guessed venues** — rows I could not assert were left
blank, and blank `venue_country` already means "fall back to neutral and log"
under the approved contract semantics. Filled tiers:

* The 9 host group rows (A: Mexico, B: Canada, D: United States) —
  `venue_country` certain by construction (hosts play group matches at home);
  stadium/city best-effort for the cross-check.
* QF ×4 (all United States), SF ×2 (Dallas 07-14, Atlanta 07-15), third place
  (Miami 07-18), final (MetLife 07-19) — high confidence.
* Everything else (non-host group rows, all R32/R16 venue-by-slot): blank,
  `confidence=unknown`, to be completed by verification, not by recall.

Verification plan (both required before the PROVISIONAL flag is lifted):
`src/check_schedule_venues.py` cross-checks drafted venues against
football-data.org venue fields **at the next routine tracker refresh** (it
performs its own fixtures GET because the cached tracker JSON stores no venue
fields; it reads only date/stage/group/venue — team names solely to locate
host group fixtures — and never touches scores; knockout rows are matched by
stage+date+venue without consulting teams). Matches already played (groups,
R32, R16, QF) get machine-verified there; the remaining fixtures (SF, third
place, final) get the manual fifa.com spot-check. The script was NOT run this
phase (no API call, no refresh triggered).

## Tests

`tests/test_match_context.py` (8 tests, all pass; full suite 33/33):
shim byte-equivalence, neutral orientation invariance, host single-orientation
(spy on build_x), HFA direction, HFA fit reproduces the frozen constant on
canonical data, gate blocks all four backtest entry points, baseline venue
parity (gate lifted locally in-test only), schedule table structural
invariants (104 rows, stage counts, host rows host==country, PROVISIONAL:
no `verified` stamps, blanks ⇔ `confidence=unknown`).

`src/smoke_test.py`: all checks pass — serving output is byte-identical
through the deprecated shim; the frozen bundle and serving paths were not
touched.

## Quotability

Per the Checkpoint 1B pinned rule and the Phase 2 gate: no number in this
report is a performance claim; HFA_ELO is a protocol parameter with
provenance. No backtest, tracker, or simulator number can be produced until
Phase 4 lifts the gate on the frozen-state protocol.
