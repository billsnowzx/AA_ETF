# Global ETF Asset Allocation Research Platform

This repository contains the Phase 1 foundation for a global ETF asset allocation research platform. The current implementation focuses on auditable pandas-based data ingestion, cleaning, return analytics, and liquidity screening for a compact ETF universe.

## Phase 1 scope

Phase 1 includes:

- Repository scaffolding aligned with the project spec
- Config-driven ETF universe, portfolio templates, benchmarks, and rebalance rules
- Price download utilities built on `yfinance`
- Reusable price cleaning utilities
- Return and annualization analytics
- Drawdown and basic risk-adjusted analytics
- Covariance and correlation analytics
- Portfolio risk contribution and marginal contribution to risk analytics
- ETF metadata-aware scoring inputs (metadata availability, expense ratio, total assets)
- Benchmark-relative attribution and annual return tables
- Portfolio template loading and weight normalization
- Static-weight portfolio return and transaction-cost helpers
- Config-driven rebalance drift checks
- Fixed-weight pandas backtest engine
- ETF summary and Phase 1 scoring
- Liquidity screening based on dollar volume
- A simple pipeline runner for download, cleaning, and liquidity summary generation
- Pytest coverage for critical deterministic functions

Phase 1 intentionally excludes:

- Optimization modules such as risk parity and Black-Litterman
- Optimization-aware overlays and macro regime logic
- Advanced optimization workflows and production dashboarding

## Repository layout

```text
D:\AI\AAETF
|-- config/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- cache/
|   `-- macro/
|-- src/
|   |-- analytics/
|   |-- backtest/
|   |-- dashboard/
|   |-- data/
|   |-- portfolio/
|   |-- universe/
|   `-- utils/
|-- tests/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   |-- tables/
|   `-- reports/
|-- requirements.txt
`-- README.md
```

## Config files

- `config/etf_universe.yaml`: Phase 1 ETF universe and asset classification
- `config/portfolio_templates.yaml`: strategic template weights with `balanced` as the default
- `config/benchmark_config.yaml`: benchmark definitions for Phase 1 evaluation
- `config/rebalance_rules.yaml`: calendar rebalance, drift thresholds, transaction cost, and placeholder overlay settings
- `config/risk_limits.yaml`: placeholder risk-limit settings; Phase 1 records this in the manifest but does not enforce limits beyond liquidity filtering

## Data flow

1. Load the ETF universe from YAML.
2. Download per-ticker raw OHLCV data with [`src/data/fetch_prices.py`](D:/AI/AAETF/src/data/fetch_prices.py).
3. Clean and validate the raw frames with [`src/data/clean_data.py`](D:/AI/AAETF/src/data/clean_data.py).
4. Fetch macro monitoring series with [`src/data/fetch_macro_data.py`](D:/AI/AAETF/src/data/fetch_macro_data.py).
5. Fetch ETF metadata snapshots with [`src/data/fetch_etf_metadata.py`](D:/AI/AAETF/src/data/fetch_etf_metadata.py).
6. Compute returns and annualized metrics with [`src/analytics/returns.py`](D:/AI/AAETF/src/analytics/returns.py).
7. Generate liquidity audit tables and a filtered universe with [`src/universe/liquidity_filter.py`](D:/AI/AAETF/src/universe/liquidity_filter.py).

Raw prices are stored as one CSV per ticker under `data/raw/` to keep the pipeline easy to inspect and audit manually.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running tests

```bash
python -m pytest
```

## Running the Phase 1 pipeline

```bash
python run_pipeline.py --start 2020-01-01 --end 2024-12-31
```

The rolling volatility and rolling Sharpe outputs use a 63-trading-day window by default. Override it with `--rolling-window`, for example:

```bash
python run_pipeline.py --start 2020-01-01 --end 2024-12-31 --rolling-window 21
```

To increase download resilience when `yfinance` is unstable:

```bash
python run_pipeline.py --start 2020-01-01 --end 2024-12-31 --download-retries 5 --download-retry-delay 2
```

This command will:

- download per-ticker raw prices into `data/raw/`
- save per-ticker ETF metadata snapshots into `data/raw/metadata/` and `outputs/tables/etf_metadata_summary.csv`
- save cleaned per-ticker data into `data/processed/`
- download macro series into `data/macro/` and write `outputs/tables/macro_observation_summary.csv`
- generate `outputs/tables/liquidity_summary.csv`
- generate `outputs/tables/etf_summary.csv`
- generate `outputs/tables/data_quality_summary.csv`
- generate `outputs/tables/investable_universe.csv`
- run the default strategic portfolio plus configured benchmarks
- generate `performance_summary.csv`, `annual_return_table.csv`, `turnover_summary.csv`, `benchmark_comparisons.csv`, `nav_series.csv`, and `return_series.csv`
- generate `benchmark_annual_excess_returns.csv` and `benchmark_drawdown_comparisons.csv`
- generate `trend_filter_summary.csv`, `trend_filter_active.csv`, and `trend_filter_scales.csv`
- generate `rebalance_reason.csv` with per-date rebalance trigger reasons
- generate `risk_limit_checks.csv` with config-driven risk-limit audit checks
- generate `rolling_volatility.csv`, `rolling_sharpe.csv`, and `drawdown_series.csv`
- generate `covariance_matrix.csv`, `correlation_matrix.csv`, `covariance_pairs.csv`, and `correlation_pairs.csv`
- generate `top_correlation_pairs.csv` and `asset_risk_snapshot.csv`
- generate `portfolio_risk_contribution.csv` with latest weights, MCTR, absolute risk contribution, and percent risk contribution
- generate `backtest_universe_validation.csv` and `backtest_universe_policy_summary.csv`
- generate `run_configuration.csv` with CLI parameters and config file inputs for auditability
- generate `output_inventory.csv` with existence and file-size checks for generated artifacts
- generate `pipeline_manifest.json` with run parameters, selected universes, and output paths
- generate the required Phase 1 charts under `outputs/figures/`, including risk contribution and MCTR charts
- generate Markdown and HTML reports under `outputs/reports/`

## Running the local dashboard

```bash
python -m src.dashboard.app
```

This command will:

- generate `outputs/reports/dashboard.html`
- start a local server at `http://127.0.0.1:8000/reports/dashboard.html`
- serve the latest pipeline tables, figures, generated reports, and run manifest through a single local page

For environments where a persistent local server is not needed, generate and open the static dashboard directly:

```bash
python -m src.dashboard.app --no-server --open
```

Optional policy:

- `--backtest-universe-mode configured` keeps configured strategy and benchmark assets in the backtest even if some fail the liquidity screen.
- `--backtest-universe-mode liquidity_filtered` requires every configured backtest asset to pass the liquidity screen and raises an error otherwise.
- `--fail-on-missing-outputs` raises an error when `output_inventory.csv` detects missing expected artifacts.
- `--fail-on-risk-limit-breach` raises an error when enabled portfolio risk limits are breached.

Current Phase 1 config note:

- The international aggregate bond slot uses `BNDX` in the live config so the default universe can satisfy the current liquidity screen in strict mode.

## Operations

Use these commands for repeatable day-to-day operation.

### 1) Full Phase 1 run (pipeline + dashboard generation)

```bash
python scripts/run_phase1.py --start 2020-01-01 --end 2024-12-31
```

This runs:

- `run_pipeline.py` with output validation (`--fail-on-missing-outputs --fail-on-empty-outputs`)
- static dashboard generation to `outputs/reports/dashboard.html`

### 2) Deterministic run metadata

```bash
python scripts/run_phase1.py --start 2020-01-01 --end 2024-12-31 --as-of-date 2024-12-31 --seed 7
```

This records deterministic metadata in `outputs/tables/pipeline_manifest.json` under:

- `parameters.as_of_date`
- `parameters.seed`

### 3) Run tests

```bash
python -m pytest -q
```

### 4) No-network smoke test (fixture-based)

```bash
python -m pytest tests/test_smoke_phase1_fixture.py -q
```

### 5) Validate required output artifacts

```bash
python scripts/check_required_outputs.py --table-dir outputs/tables --figure-dir outputs/figures --report-dir outputs/reports
```

This fails if required tables/figures/reports are missing or empty.

### 6) Launch dashboard server

```bash
python -m src.dashboard.app --host 127.0.0.1 --port 8000
```

### Troubleshooting

- `Config schema validation failed`:
  Check YAML structure in `config/etf_universe.yaml`, `config/portfolio_templates.yaml`, `config/benchmark_config.yaml`, and `config/rebalance_rules.yaml`.
- `Output inventory validation failed; missing artifacts`:
  Re-run pipeline with `--fail-on-missing-outputs` and inspect `outputs/tables/output_inventory.csv`.
- `Output inventory validation failed; empty artifacts`:
  Re-run pipeline with `--fail-on-empty-outputs` and inspect generated file sizes in `output_inventory.csv`.
- Dashboard opens but shows missing sections:
  Ensure pipeline completed first and that `outputs/tables/*.csv` plus `outputs/figures/*.png` exist.

## Example usage

```python
from src.data.fetch_prices import fetch_prices
from src.data.clean_data import batch_clean_price_frames
from src.portfolio.weights import load_portfolio_template
from src.universe.liquidity_filter import filter_liquid_universe

raw_frames = fetch_prices(
    tickers=["VTI", "VEA", "IEMG"],
    start="2020-01-01",
    end="2024-12-31",
)

clean_frames = batch_clean_price_frames(raw_frames)
liquid_tickers, liquidity_table = filter_liquid_universe(clean_frames)
balanced_weights = load_portfolio_template("config/portfolio_templates.yaml")
```

## Design principles

- Pandas-only implementation for correctness and auditability
- Explicit handling of missing data
- Config-driven behavior where practical
- Narrow, typed, unit-testable functions
