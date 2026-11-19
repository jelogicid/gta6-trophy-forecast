# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jelogicid/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jelogicid/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Deterministic quantile forecaster for GTA VI trophy earn rates, built for the
[gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun. Python 3.13+, standard library only, no network.

## Model

Each trophy's observed rarity history is mapped into logit space. With three
or more non-flat reads, a recency-weighted least-squares line is fit against
ln(1 + hours since release) and extrapolated to the target time with damping
1/(1 + 0.35 * horizon_days), then blended toward the last observation
(weight min(1, n/8)). Uncertainty is the max of empirical step volatility
(1.4826 * median nonzero |diff|), the fit's residual std, and a 0.08 floor;
flat or short series keep the last logit with a 0.5 volatility floor. While
the tracked player population is under 8000, a dilution nudge of
-0.5 * ln(20000 / players) (floored at -2.5) pulls rates down, since early
rare-trophy percentages deflate as casual players sync. The spread
min(sigma * sqrt(horizon_days), 3) is applied at normal quantile z-levels
(-1.645, -0.674, 0, 0.674, 1.645) and mapped back through the logistic to
five non-decreasing percentages, rounded half-up to 4 decimals.

## Run

```bash
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 -P -s main.py
```

Reads every `*.in` (JSON case) from the input dir in sorted order and writes
`<id>.out` with `{"trophies":[{"id":<int>,"q":[q05,q25,q50,q75,q95]}, ...]}`.