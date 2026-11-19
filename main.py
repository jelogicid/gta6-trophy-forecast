#!/usr/bin/env python3
"""GTA VI trophy earn-rate quantile forecaster.

make.fun bounty: gta6-trophy-rarity-forecast-01
Python 3.13+, standard library only, no network, deterministic.

Model: per-trophy logit-space damped trend fit (weighted least squares,
recency-weighted), blended toward the last observation, widened by an
empirical step-volatility / residual floor, rarity-dependent rate-space
dilution while the tracked player population is small, then mapped back
through the logistic to five quantiles (5/25/50/75/95%).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RELEASE_EPOCH = 1795046400.0  # 2026-11-19T00:00:00Z
Z_LEVELS = (-1.645, -0.674, 0.0, 0.674, 1.645)
RATE_MIN = 0.01
RATE_MAX = 99.99


def parse_epoch(iso: str) -> float:
    """ISO-8601 UTC -> POSIX seconds. Accepts Z/z, +00:00, fractions."""
    text = iso.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def hours_since_release(iso: str) -> float:
    return (parse_epoch(iso) - RELEASE_EPOCH) / 3600.0


def logit(pct: float) -> float:
    p = min(max(float(pct), RATE_MIN), RATE_MAX)
    return math.log(p / (100.0 - p))


def inv_logit(x: float) -> float:
    # Branch on sign so exp() never overflows for extreme logits.
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(value: float):
    """Round half up to 4 decimals; return int when the value is whole."""
    d = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if d == d.to_integral_value():
        return int(d)
    return float(d)


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def step_sigma(ys: list[float]) -> float:
    """MAD-style volatility from consecutive logit steps (zeros dropped)."""
    diffs = [abs(b - a) for a, b in zip(ys, ys[1:]) if b != a]
    if not diffs:
        return 0.0
    return 1.4826 * median(diffs)


def wls_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Weighted least squares y = a + b*x, exponential recency weights.

    Returns (a, b, residual_std) with residual_std the population standard
    deviation of the unweighted fit residuals.
    """
    n = len(xs)
    scale = max(n / 3.0, 1.0)
    ws = [math.exp((i - (n - 1)) / scale) for i in range(n)]
    # fsum keeps the normal-equation sums exact to the last ulp.
    sw = math.fsum(ws)
    sx = math.fsum(w * x for w, x in zip(ws, xs))
    sy = math.fsum(w * y for w, y in zip(ws, ys))
    sxx = math.fsum(w * x * x for w, x in zip(ws, xs))
    sxy = math.fsum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * sxx - sx * sx
    if denom <= 0.0:
        # All x identical: slope unidentifiable, fall back to weighted mean.
        b = 0.0
        a = sy / sw
    else:
        b = (sw * sxy - sx * sy) / denom
        a = (sy - b * sx) / sw
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    mean_r = math.fsum(resid) / n
    var = math.fsum((r - mean_r) ** 2 for r in resid) / n
    return a, b, math.sqrt(var)


def forecast_trophy(
    series: list[tuple[float, float]],
    players_series: list[float],
    horizon_days: float,
    target_hours: float,
) -> list:
    """Five quantiles (percent) for one trophy at the target time."""
    if not series:
        # No observed rates at all: floor rarity, horizon widening matched
        # to the flat-branch floor (0.3).
        m = logit(0.01)
        spread = 0.3 * math.sqrt(horizon_days)
        return [round4(inv_logit(m + z * spread)) for z in Z_LEVELS]

    xs = [math.log1p(max(h, 0.0)) for h, _ in series]
    ys = [y for _, y in series]
    last_logit = ys[-1]
    sigma_step = step_sigma(ys)
    flat = all(y == ys[0] for y in ys)

    if len(series) >= 3 and not flat:
        a, b, resid_std = wls_fit(xs, ys)
        damp = 1.0 / (1.0 + 0.35 * horizon_days)
        x_target = math.log1p(max(target_hours, 0.0))
        x_last = xs[-1]
        m_fit = a + b * (x_last + (x_target - x_last) * damp)
        fit_weight = min(1.0, len(series) / 8.0)
        m = fit_weight * m_fit + (1.0 - fit_weight) * last_logit
        sigma = max(sigma_step, resid_std, 0.08)
    else:
        m = last_logit
        # Flat launch-night history: the dilution move is predictable but its
        # timing is not. Floor 0.3 targets ~0.9 coverage (practice: 0.12 gave
        # coverage_90 0.218, 0.5 gave 0.818).
        sigma = max(sigma_step, 0.3)

    # Rarity-dependent dilution in RATE space, not a flat logit shift: on an
    # upward population refresh, rare trophies keep a roughly fixed earner
    # count while the denominator grows (rate -> rate/g), but common trophies
    # (>=30%, casuals earn them too) barely fall. k interpolates between the
    # two: rate_next = rate * (1/g + (1 - 1/g) * k). Mature populations
    # (p_last >= 8000) are untouched.
    if players_series:
        p_last = players_series[-1]
        if p_last < 8000:
            r0 = inv_logit(m)                       # current predicted percent
            g = min(20000.0 / max(p_last, 1), 8.0)  # expected growth, cap 8x
            k = min(1.0, r0 / 30.0)                 # casuals fully earn >=30%
            r_pred = r0 * (1.0 / g + (1.0 - 1.0 / g) * k)
            m = logit(r_pred)

    spread = min(sigma * math.sqrt(horizon_days), 3.0)
    return [round4(inv_logit(m + z * spread)) for z in Z_LEVELS]


def build_case(case: dict) -> dict:
    horizon_days = float(case["horizon_minutes"]) / 1440.0
    target_hours = hours_since_release(case["target"])
    trophies = case["trophies"]
    history = case.get("history") or []

    # Spec says history is oldest-first; sort defensively so an out-of-order
    # read can never invert the recency weights or the step diffs.
    ordered = sorted(
        ((hours_since_release(h["at"]), h) for h in history),
        key=lambda pair: pair[0],
    )
    reads: list[tuple[float, list]] = []
    players_series: list[float] = []
    for t, h in ordered:
        reads.append((t, h.get("rates") or []))
        p = h.get("players_tracked")
        if isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0:
            players_series.append(float(p))

    rows = []
    for i, tr in enumerate(trophies):
        series = []
        for t, rates in reads:
            if i < len(rates):
                r = rates[i]
                # null means "not observed"; bools/strings are not rates.
                if isinstance(r, (int, float)) and not isinstance(r, bool):
                    series.append((t, logit(r)))
        q = forecast_trophy(series, players_series, horizon_days, target_hours)
        rows.append({"id": int(tr["id"]), "q": q})
    return {"trophies": rows}


def main() -> None:
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = build_case(case)
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(
            json.dumps(result, separators=(",", ":")), encoding="utf-8"
        )


if __name__ == "__main__":
    main()