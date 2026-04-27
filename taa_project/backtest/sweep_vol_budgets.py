"""Sweep standalone TAA vol budgets and audit IPS compliance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from taa_project.compliance import audit_ips_compliance, compliance_stats
from taa_project.config import OUTPUT_DIR
from taa_project.backtest import run_backtest


SWEEP_RESULTS_FILENAME = "vol_budget_sweep.csv"
TRADING_DAYS_PER_YEAR = 252


def _annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float((1.0 + clean.mean()) ** TRADING_DAYS_PER_YEAR - 1.0)


def _annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    wealth = (1.0 + clean).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def run_sweep(
    start: str = "2003-01-01",
    end: str | None = None,
    output_csv: Path = OUTPUT_DIR / SWEEP_RESULTS_FILENAME,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for vol_budget in [round(value, 2) for value in np.arange(0.05, 0.1601, 0.01)]:
        weights, regimes, returns = run_backtest.run(
            start=start,
            end=end,
            use_timesfm=False,
            vol_budget=vol_budget,
            enforce_vol_ceiling=False,
        )
        del regimes

        breaches = audit_ips_compliance(weights, returns)
        stats = compliance_stats(returns)
        rows.append(
            {
                "target_vol": vol_budget,
                "annual_return": _annualized_return(returns),
                "volatility": _annualized_volatility(returns),
                "max_drawdown": _max_drawdown(returns),
                "ips_breach_count": int(len(breaches)),
                "ips_breach_rules": "" if breaches.empty else "|".join(sorted(breaches["rule"].unique())),
                "max_rolling_21d_vol": stats["max_rolling_21d_vol"],
                "compliance_max_drawdown": stats["max_drawdown"],
            }
        )

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    breached = results.loc[results["ips_breach_count"] > 0]
    if breached.empty:
        print("IPS compliance audit passed for all sweep configurations.")
    else:
        print("IPS compliance audit found breaches:")
        print(breached[["target_vol", "ips_breach_count", "ips_breach_rules"]].to_string(index=False))

    qualified = results.loc[(results["annual_return"] >= 0.08) & (results["volatility"] <= 0.15)]
    if qualified.empty:
        print("No configurations met annualized return >= 8% and volatility <= 15%.")
    else:
        print("Configurations with annualized return >= 8% and volatility <= 15%:")
        print(qualified.to_string(index=False))
    print(f"Sweep results written to {output_csv}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep standalone TAA vol budgets without TimesFM.")
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-csv", default=str(OUTPUT_DIR / SWEEP_RESULTS_FILENAME))
    args = parser.parse_args()

    run_sweep(start=args.start, end=args.end, output_csv=Path(args.output_csv))


if __name__ == "__main__":
    main()
