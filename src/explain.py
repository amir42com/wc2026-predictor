"""
Plain-language grouping of SHAP feature contributions for the Match Predictor.

The model's SHAP values explain its raw (pre-Elo-blend) Win/Draw/Loss output.
This module groups the 31 raw features into a FIXED, documented set of reasons
and SUMS their SHAP values per group. SHAP is additive, so the grouped sums
still reconcile to the model output: base_value[class] + sum(all SHAP[class])
== the raw margin for that class, and softmax(margins) == the model's W/D/L.

Descriptions are generated PER MATCHUP from the actual feature values and the
SHAP sign/direction — never hardcoded per team. SHAP is oriented to the HOME
class here, so a positive group sum favours the home team (blue), a negative
sum favours the away team (amber).

This module does NOT touch the model or the W/D/L predictions; it only reshapes
how the existing SHAP contributions are presented.
"""

from __future__ import annotations

# ── fixed feature -> group mapping (covers EVERY model feature) ─────────────
GROUP_ORDER = [
    "Team quality", "Recent form", "Competition strength",
    "Head-to-head", "Neutral venue", "Other factors",
]

FEATURE_GROUPS: dict[str, str] = {
    # Team quality — Elo gap + each team's Elo
    "elo_diff": "Team quality", "home_elo": "Team quality", "away_elo": "Team quality",
    # Recent form — win rate + avg goal diff (last 5/10), both teams
    "home_win_rate_5": "Recent form", "away_win_rate_5": "Recent form",
    "home_gd_5": "Recent form", "away_gd_5": "Recent form",
    "home_win_rate_10": "Recent form", "away_win_rate_10": "Recent form",
    "home_gd_10": "Recent form", "away_gd_10": "Recent form",
    # Competition strength — confederation strength + membership indicators
    "home_conf_elo": "Competition strength", "away_conf_elo": "Competition strength",
    # Head-to-head
    "h2h_n": "Head-to-head", "h2h_home_wr": "Head-to-head",
    # Neutral venue
    "neutral": "Neutral venue",
    # Other factors — is_world_cup + anything unmapped (residual)
    "is_world_cup": "Other factors",
}
for _c in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "Other", "UEFA"):
    FEATURE_GROUPS[f"h_conf_{_c}"] = "Competition strength"
    FEATURE_GROUPS[f"a_conf_{_c}"] = "Competition strength"


def group_of(feature: str) -> str:
    """Group for a feature; anything unmapped falls into Other factors (residual)."""
    return FEATURE_GROUPS.get(feature, "Other factors")


def group_contributions(shap_home: dict[str, float]) -> dict[str, float]:
    """Sum SHAP (home-class oriented) per group. {feature: shap} -> {group: sum}."""
    out = {g: 0.0 for g in GROUP_ORDER}
    for feat, val in shap_home.items():
        out[group_of(feat)] += float(val)
    return out


# ── per-feature labels for the deeper-detail breakdown ──────────────────────

def _feat_label(f: str, v: float, ctx: dict) -> str:
    home, away = ctx["home_team"], ctx["away_team"]
    if f == "elo_diff":
        stronger = home if v > 0 else away
        return f"Elo gap: {stronger} +{abs(v):.0f}"
    if f == "home_elo":
        return f"{home} Elo: {v:.0f}"
    if f == "away_elo":
        return f"{away} Elo: {v:.0f}"
    side = away if f.startswith(("away_", "a_conf_")) else home
    if "_win_rate_" in f:
        return f"{side} win rate (last {f.rsplit('_', 1)[1]}): {v*100:.0f}%"
    if "_gd_" in f:
        return f"{side} goal diff (last {f.rsplit('_', 1)[1]}): {v:+.1f}"
    if f == "h2h_n":
        return f"Head-to-head games played: {v:.0f}"
    if f == "h2h_home_wr":
        return f"Head-to-head: {home} win rate {v*100:.0f}%"
    if f == "home_conf_elo":
        return f"{home} confederation strength: {v:.0f}"
    if f == "away_conf_elo":
        return f"{away} confederation strength: {v:.0f}"
    if f == "neutral":
        return f"Neutral venue: {'yes' if v else 'no'}"
    if f == "is_world_cup":
        return f"World Cup match: {'yes' if v else 'no'}"
    return f


def detail_entries(group: str, shap_home: dict[str, float], feat_vals: dict[str, float],
                   ctx: dict) -> list[tuple[str, float]]:
    """
    Raw features behind a group with their individual SHAP contributions, sorted
    by |contribution| desc. The many confederation one-hot indicators are
    collapsed into a single net entry to keep the breakdown readable.
    """
    feats = [f for f in shap_home if group_of(f) == group]
    entries: list[tuple[str, float]] = []
    onehot_sum, onehot_present = 0.0, False
    for f in feats:
        if f.startswith(("h_conf_", "a_conf_")):
            onehot_sum += float(shap_home[f])
            onehot_present = True
            continue
        entries.append((_feat_label(f, float(feat_vals[f]), ctx), float(shap_home[f])))
    if onehot_present:
        entries.append(("Confederation indicators (net)", onehot_sum))
    entries.sort(key=lambda e: abs(e[1]), reverse=True)
    return entries


# ── dynamic one-line descriptions per group ─────────────────────────────────

def _describe(group: str, shap_sum: float, fv: dict[str, float], ctx: dict) -> str:
    home, away = ctx["home_team"], ctx["away_team"]
    home_conf, away_conf = ctx["home_conf"], ctx["away_conf"]
    home_favoured = shap_sum > 0          # SHAP oriented to home class
    edge = home if home_favoured else away
    other = away if home_favoured else home

    if group == "Team quality":
        ediff = fv["elo_diff"]
        stronger = home if ediff > 0 else away
        hi, lo = (fv["home_elo"], fv["away_elo"]) if ediff > 0 else (fv["away_elo"], fv["home_elo"])
        return f"{stronger}'s Elo is {abs(ediff):.0f} points higher ({hi:.0f} vs {lo:.0f})."

    if group == "Recent form":
        # Describe the side the contribution actually favours (faithful to the
        # SHAP sign), citing that side's own recent record as context.
        wr = fv["home_win_rate_10"] if home_favoured else fv["away_win_rate_10"]
        gd = fv["home_gd_10"] if home_favoured else fv["away_gd_10"]
        return (f"{edge}'s recent form tips the balance their way "
                f"({wr*100:.0f}% win rate, {gd:+.1f} goal diff over the last 10).")

    if group == "Competition strength":
        ch, ca = fv["home_conf_elo"], fv["away_conf_elo"]
        if home_conf == away_conf:
            return (f"Both teams come from the same confederation ({home_conf}); "
                    f"a near-neutral factor with a slight net edge to {edge}.")
        stronger = home if ch >= ca else away
        s_conf, w_conf = (home_conf, away_conf) if ch >= ca else (away_conf, home_conf)
        if stronger == edge:
            return f"{edge}'s confederation ({s_conf}) is rated stronger than {w_conf}."
        return f"Confederation strength nets a slight edge to {edge}."

    if group == "Head-to-head":
        n = fv["h2h_n"]
        if n <= 0:
            return f"Little head-to-head history; a slight edge to {edge}."
        return f"Their head-to-head record over {n:.0f} past meetings leans to {edge}."

    if group == "Neutral venue":
        return (f"Played at a neutral venue, so {home} gets no home advantage "
                f"— a slight edge to {edge}.")

    # Other factors (is_world_cup + residual, possibly folded H2H)
    return f"Small net edge to {edge} from minor factors."


def _headline(group: str) -> str:
    return group


# ── assemble the reasons ────────────────────────────────────────────────────

def build_reasons(shap_home: dict[str, float], feat_vals: dict[str, float], ctx: dict,
                  max_reasons: int = 5, fold_tau: float = 0.06) -> list[dict]:
    """
    Group SHAP -> ranked plain-language reasons.

    ctx must contain: home_team, away_team, home_conf, away_conf.
    Returns reasons sorted by |contribution| desc (largest first), each:
      {group, headline, description, shap_sum, home_favoured, magnitude (0..1),
       detail: [(label, shap_value), ...]}
    Tiny Head-to-head is folded into Other factors; a tiny Other factors is
    dropped. `magnitude` is |shap_sum| normalised to the largest shown reason.
    """
    groups = group_contributions(shap_home)
    total_abs = sum(abs(v) for v in groups.values()) or 1.0

    # fold a tiny head-to-head effect into Other factors
    if abs(groups.get("Head-to-head", 0.0)) < fold_tau * total_abs:
        groups["Other factors"] = groups.get("Other factors", 0.0) + groups.pop("Head-to-head", 0.0)
    # drop a tiny Other factors entirely
    if "Other factors" in groups and abs(groups["Other factors"]) < fold_tau * total_abs:
        groups.pop("Other factors")

    max_abs = max((abs(v) for v in groups.values()), default=1.0) or 1.0
    reasons = []
    for g, s in groups.items():
        # if H2H was folded, its detail rolls into Other factors
        detail_groups = [g] + (["Head-to-head"] if g == "Other factors"
                               and "Head-to-head" not in groups else [])
        detail: list[tuple[str, float]] = []
        for dg in detail_groups:
            detail += detail_entries(dg, shap_home, feat_vals, ctx)
        detail.sort(key=lambda e: abs(e[1]), reverse=True)
        reasons.append({
            "group": g,
            "headline": _headline(g),
            "description": _describe(g, s, feat_vals, ctx),
            "shap_sum": float(s),
            "home_favoured": s > 0,
            "magnitude": abs(s) / max_abs,
            "detail": detail,
        })
    reasons.sort(key=lambda r: abs(r["shap_sum"]), reverse=True)
    return reasons[:max_reasons]
