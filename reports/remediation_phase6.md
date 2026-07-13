# Remediation Phase 6 — prediction ledger + two-tier live evaluation

Recorded: 2026-07-13, branch `fix/scientific-audit`.
Checkpoint 5 tiebreak ruling recorded: the REGULATION order (overall
points → GD → GF, then among-tied with reapplication, then documented random
fallback) stands as implemented; the instruction's H2H-first paraphrase was
the UEFA convention, withdrawn.

## 6a. Prediction ledger (src/prediction_ledger.py, commit 6d8050f)

Append-only from birth: after creation the file is only ever opened in append
mode; existing fixture_ids are never rewritten (byte-stability enforced by
tests/test_ledger.py). Each entry: stable fixture ID (official match numbers
M101–M104), kickoff date (UTC kickoff times are not held in-repo; the
before-kickoff bound is date-level), both teams, FULL-precision Blend and
Elo-baseline vectors (repr floats), venue context + source (PROVISIONAL
schedule; SF Dallas/Atlanta, bronze Miami, final MetLife are asserted rows;
a host in the pairing gets the home slot and the HFA prior), bundle SHA,
features hash, code SHA, ELO_BLEND_W / prior + baseline draw rates / HFA /
rho, generation timestamp, generated_before_kickoff, retrospective=false.
Inference is the Phase 2 contract on the frozen serving state.

**BLOCKER FOUND: the tracker cache is STALE** — `data/wc2026_results.json`
was fetched 2026-06-22 and holds 40 group-stage results only; no R32/R16/QF
records exist locally, so the SF pairings are NOT derivable from cached
results and the boundary forbids this remediation from calling the API. Per
the phase instruction's own rule ("if a pairing is not yet determined, ledger
it the moment it is"), the ledger machinery is committed with ZERO entries.
One operator action ledgers the semifinals:

    uv run python src/prediction_ledger.py --refresh
    git add reports/prediction_ledger.csv && git commit -m "Ledger semifinals"
    git push <remote> fix/scientific-audit

(`--refresh` runs the routine fetch_results refresh first — also the moment
`src/check_schedule_venues.py` can machine-verify the schedule table. If the
QF date-window grouping is ambiguous the script refuses and accepts explicit
`--fixture M101 "A" "B"` pairings.) M103/M104 are ledgered the same way the
moment the semifinals decide them.

## 6b. Two-tier live evaluation (src/live_evaluation.py, commit 92c2aac)

Tier 1 prospective: read FROM THE LEDGER. Tier 2 retrospective: finished
fixtures without a ledger entry recomputed from the frozen bundle at full
precision, labeled retrospective=true; tiers are never silently pooled (every
row carries its tier; counts reported). Results are a separate table joined
by fixture_id. Target: after-extra-time W/D/L, shootouts scored as draws —
the shootout strip is re-validated (a PENALTY_SHOOTOUT record with an unlevel
score is rejected as invalid, not silently trusted). Metrics per system: n,
correct, accuracy, pooled log-loss, pooled Brier; McNemar exact + paired
bootstrap (10,000 resamples, seed 12345) via paired_tests' own functions.
FINAL MODE refuses unless EXACTLY 104 valid finished fixtures exist; only
then is reports/live_evaluation.csv written — interim runs print to the
terminal and produce no artifact.

## Verification

tests/test_ledger.py (2): append-only byte-stability (re-append with changed
numbers is a no-op; new entries extend the file with the original bytes as a
strict prefix; full regeneration idempotent), CSV escaping.
tests/test_live_evaluation.py (7, all SYNTHETIC): home/draw/away outcomes,
extra-time decisive, shootout-stripped draw, missing/unfinished, bad shootout
strip, incomplete record, duplicate fixture_ids (all copies dropped), no-id
record, orientation mapping (including reversed home/away), join + hand-
checked pooled log-loss + tier counts + McNemar/bootstrap wiring, team-
mismatch flagging, final-mode 103/104/105 gate, final report written only in
final mode. Full suite 55/55; smoke test all checks pass.

No live WC 2026 result was read at any point: the only cache access this
phase read structure, dates, and (attempted) knockout pairings — the QF
window returned zero records. live_evaluation's live path was never executed.
