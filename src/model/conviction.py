"""
conviction.py — NBA port of the MLB edge conviction-tier system.

Faithful to the MLB engine's architecture (mlb_edge/edge_calculator.py +
config.py): a pick is only "live" when multiple INDEPENDENT signals converge,
each signal has a reliability gate (minimum sample before it may fire), the
tier maps to a Kelly stake multiplier, and edges outside a fixed band are
skipped with a named reason. Basketball-native signals replace the pitching
ones:

    S_QUALITY    season net-rating edge favors the pick by >= 3.0 pts/100.
                 Gate: both teams >= 15 season games (season NetRtg SE at
                 n=15 ≈ 3.4 -> a 3-point read is ~1 SE; below that it's noise).
    S_FORM       last-7 net-rating edge favors the pick by >= 5.0 pts/100.
                 Gate: full 7-game windows on both sides. Threshold is wider
                 than S_QUALITY because 7-game NetRtg SE ≈ 5.
    S_SCHEDULE   rest asymmetry favors the pick: opponent on a back-to-back
                 while pick side is not, or pick side has >= 2 extra rest
                 days. No sample gate — schedule facts are exact.
    S_AGREEMENT  win-prob model and spread model choose the same side AND
                 |projected margin| >= 4.0 (a real lean, not rounding).
    S_VALUE      market-dependent (in-season only): devigged edge for the
                 pick within the MLB band [MIN_EDGE, MAX_EDGE] = [4pp, 15pp].
                 Above 15pp the model is more likely wrong than the market
                 is generous — same lesson the MLB backtests encoded.

    Tier = signals fired:  3+ DIAMOND (size 1.00) | 2 PLATINUM (0.30)
                           1 GOLD (0.00 — logged, not staked) | 0 SKIP
    Stake = quarter-Kelly * tier size, clamped at 5% bankroll (MLB values).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

# ---- MLB-inherited constants (provenance: mlb_edge/config.py) --------------
MIN_EDGE_PP: float = 4.0
MAX_EDGE_PP: float = 15.0
KELLY_FRACTION: float = 0.25
MAX_STAKE: float = 0.05
TIER_SIZES: dict[str, float] = {"DIAMOND": 1.00, "PLATINUM": 0.30,
                                "GOLD": 0.00, "SKIP": 0.00}

# ---- NBA signal thresholds + reliability gates ------------------------------
QUALITY_EDGE_MIN: float = 3.0        # pts/100
QUALITY_MIN_GAMES: int = 15
FORM_EDGE_MIN: float = 5.0           # pts/100 over last 7
REST_EDGE_DAYS: float = 2.0
AGREEMENT_MARGIN_MIN: float = 4.0


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100.0 if odds > 0 else 100.0 / (-odds))


def expected_value(prob: float, decimal_odds: float) -> float:
    """EV per $1 risked."""
    return prob * (decimal_odds - 1) - (1 - prob)


def kelly_stake(prob: float, decimal_odds: float,
                fraction: float = KELLY_FRACTION,
                max_stake: float = MAX_STAKE) -> float:
    if decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1
    raw = (b * prob - (1 - prob)) / b
    return float(min(fraction * raw, max_stake)) if raw > 0 else 0.0


def devig_two_way(dec_a: float, dec_b: float) -> tuple[float, float]:
    """Multiplicative devig of a two-way market -> fair probabilities."""
    ia, ib = 1.0 / dec_a, 1.0 / dec_b
    s = ia + ib
    return ia / s, ib / s


@dataclass
class Conviction:
    tier: str
    signals: list[str] = field(default_factory=list)
    why_skipped: str = ""
    stake_frac: float = 0.0
    edge_pp: float | None = None
    ev_per_dollar: float | None = None


def score(row: dict) -> Conviction:
    """Score one game. `row` needs (home-perspective, pick-side resolved):

    pick_is_home, p_pick, d_net_rtg_std, d_net_rtg_r7, games_h, games_a,
    h_rest, a_rest, h_b2b, a_b2b, pred_margin_home,
    optional market: pick_dec, opp_dec (decimal odds).
    """
    sgn = 1.0 if row["pick_is_home"] else -1.0
    signals: list[str] = []
    notes: list[str] = []

    # S_QUALITY (gated on season sample)
    if min(row["games_h"], row["games_a"]) >= QUALITY_MIN_GAMES:
        if sgn * row["d_net_rtg_std"] >= QUALITY_EDGE_MIN:
            signals.append("QUALITY")
    else:
        notes.append("quality gate: <15 games")

    # S_FORM (gated on full windows — r7 columns are NaN otherwise upstream)
    if sgn * row["d_net_rtg_r7"] >= FORM_EDGE_MIN:
        signals.append("FORM")

    # S_SCHEDULE (exact facts, no gate)
    pick_rest = row["h_rest"] if row["pick_is_home"] else row["a_rest"]
    opp_rest = row["a_rest"] if row["pick_is_home"] else row["h_rest"]
    pick_b2b = row["h_b2b"] if row["pick_is_home"] else row["a_b2b"]
    opp_b2b = row["a_b2b"] if row["pick_is_home"] else row["h_b2b"]
    if (opp_b2b and not pick_b2b) or (pick_rest - opp_rest >= REST_EDGE_DAYS):
        signals.append("SCHEDULE")

    # S_AGREEMENT
    margin_side_home = row["pred_margin_home"] > 0
    if margin_side_home == row["pick_is_home"] \
            and abs(row["pred_margin_home"]) >= AGREEMENT_MARGIN_MIN:
        signals.append("AGREEMENT")

    # S_VALUE — only when a market exists (in-season)
    edge_pp = ev = None
    stake = 0.0
    if row.get("pick_dec") and row.get("opp_dec"):
        fair_pick, _ = devig_two_way(row["pick_dec"], row["opp_dec"])
        edge_pp = (row["p_pick"] - fair_pick) * 100.0
        ev = expected_value(row["p_pick"], row["pick_dec"])
        if MIN_EDGE_PP <= edge_pp <= MAX_EDGE_PP:
            signals.append("VALUE")
        elif edge_pp > MAX_EDGE_PP:
            notes.append(f"edge {edge_pp:.1f}pp > {MAX_EDGE_PP:.0f}pp cap — "
                         "model likelier wrong than market generous")

    n = len(signals)
    tier = "DIAMOND" if n >= 3 else "PLATINUM" if n == 2 \
        else "GOLD" if n == 1 else "SKIP"
    why = "" if n >= 2 else (
        "; ".join(notes) if notes else
        f"{n} signal{'s' if n != 1 else ''} — below conviction floor (2)")

    if row.get("pick_dec") and TIER_SIZES[tier] > 0:
        stake = kelly_stake(row["p_pick"], row["pick_dec"]) * TIER_SIZES[tier]

    return Conviction(tier=tier, signals=signals, why_skipped=why,
                      stake_frac=round(stake, 4),
                      edge_pp=None if edge_pp is None else round(edge_pp, 2),
                      ev_per_dollar=None if ev is None else round(ev, 4))
