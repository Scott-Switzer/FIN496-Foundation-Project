"""Smoke tests for the Task 12 pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from taa_project import main as pipeline_main


def test_run_pipeline_orchestrates_with_stubbed_dependencies(monkeypatch, tmp_path) -> None:
    captured: dict[str, float] = {}

    def fake_run_data_audit(output_dir):
        return {"output_dir": output_dir}

    def fake_build_saa_portfolio(start_date, end_date, output_dir):
        return None

    def fake_build_benchmarks(start_date, end_date, output_dir):
        return None

    def fake_run_walkforward(start, end, folds, use_timesfm, vol_budget, output_dir):
        captured["walkforward_vol_budget"] = vol_budget
        return {"folds": pd.DataFrame(), "oos_returns": pd.DataFrame(), "oos_weights": pd.DataFrame(), "oos_regimes": pd.DataFrame()}

    def fake_build_attribution(start, end, folds, use_timesfm, vol_budget, output_dir):
        captured["attribution_vol_budget"] = vol_budget
        return {"per_signal": pd.DataFrame()}

    def fake_build_reporting(start, end, folds, use_timesfm, vol_budget, output_dir, figure_dir):
        captured["reporting_vol_budget"] = vol_budget
        return {
            "ips_compliance": pd.DataFrame(columns=["portfolio", "date", "rule", "value", "bound"]),
            "metrics": pd.DataFrame(
                [
                    {
                        "portfolio": "SAA+TAA",
                        "annualized_return": 0.10,
                        "annualized_volatility": 0.09,
                        "max_drawdown": -0.12,
                        "sharpe_rf_2pct": 0.90,
                        "sortino_rf_2pct": 1.10,
                        "calmar": 0.80,
                    }
                ]
            ),
        }

    def fake_build_notebook(output_dir, notebook_dir):
        path = notebook_dir / "diagnostics.ipynb"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_build_report(output_dir, figure_dir, report_dir):
        md = report_dir / "report.md"
        pdf = report_dir / "report.pdf"
        md.write_text("# report", encoding="utf-8")
        pdf.write_text("pdf", encoding="utf-8")
        return md, pdf

    def fake_build_deck(output_dir, figure_dir, report_dir):
        pdf = report_dir / "deck.pdf"
        pdf.write_text("pdf", encoding="utf-8")
        return pdf

    monkeypatch.setattr(pipeline_main, "run_data_audit", fake_run_data_audit)
    monkeypatch.setattr(pipeline_main, "build_saa_portfolio", fake_build_saa_portfolio)
    monkeypatch.setattr(pipeline_main, "build_benchmarks", fake_build_benchmarks)
    monkeypatch.setattr(pipeline_main, "run_walkforward", fake_run_walkforward)
    monkeypatch.setattr(pipeline_main, "build_attribution", fake_build_attribution)
    monkeypatch.setattr(pipeline_main, "build_reporting", fake_build_reporting)
    monkeypatch.setattr(pipeline_main, "build_diagnostics_notebook", fake_build_notebook)
    monkeypatch.setattr(pipeline_main, "build_report", fake_build_report)
    monkeypatch.setattr(pipeline_main, "build_deck", fake_build_deck)
    monkeypatch.setattr(pipeline_main, "TRIAL_LEDGER_CSV", tmp_path / "TRIAL_LEDGER.csv")

    artifacts = pipeline_main.run_pipeline(
        start="2003-01-01",
        end="2003-06-30",
        folds=2,
        use_timesfm=False,
        vol_budget=0.08,
        output_dir=tmp_path / "outputs",
        figure_dir=tmp_path / "figures",
        report_dir=tmp_path / "reports",
        notebook_dir=tmp_path / "notebooks",
    )

    assert Path(artifacts["notebook_path"]).exists()
    assert Path(artifacts["report_markdown_path"]).exists()
    assert Path(artifacts["deck_pdf_path"]).exists()
    assert (tmp_path / "TRIAL_LEDGER.csv").exists()
    assert captured["walkforward_vol_budget"] == 0.08
    assert captured["attribution_vol_budget"] == 0.08
    assert captured["reporting_vol_budget"] == 0.08


def test_run_pipeline_raises_when_timesfm_requested_but_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline_main, "timesfm_is_available", lambda: False)

    with pytest.raises(RuntimeError, match="TimesFM was requested"):
        pipeline_main.run_pipeline(
            use_timesfm=True,
            output_dir=tmp_path / "outputs",
            figure_dir=tmp_path / "figures",
            report_dir=tmp_path / "reports",
            notebook_dir=tmp_path / "notebooks",
        )


def test_run_pipeline_raises_on_invalid_vol_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="vol_budget"):
        pipeline_main.run_pipeline(
            use_timesfm=False,
            vol_budget=0.20,
            output_dir=tmp_path / "outputs",
            figure_dir=tmp_path / "figures",
            report_dir=tmp_path / "reports",
            notebook_dir=tmp_path / "notebooks",
        )
