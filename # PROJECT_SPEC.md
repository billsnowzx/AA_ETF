# PROJECT_SPEC.md

## Project Name
Global ETF Asset Allocation Research Platform

---

## 1. Project Objective

Build a robust, modular, and auditable Python research platform for global multi-asset allocation using highly liquid ETFs.

The platform must support:

1. ETF universe definition and liquidity screening
2. Strategic asset allocation (SAA)
3. Tactical overlays and rule-based rebalancing
4. Observation / monitoring system
5. Evaluation / scoring system
6. Backtesting framework
7. Dashboard and reporting outputs
8. Future extensions such as Risk Parity and Black-Litterman

This is a research and portfolio construction platform, not a retail trading bot.

The design philosophy is:

- Liquidity first
- Simplicity before complexity
- Auditability before speed
- Modular codebase
- Clear separation between data, portfolio logic, analytics, and reporting
- Support long-term maintainability and future automation

---

## 2. Investment Philosophy

The system should implement a global ETF allocation approach based on:

- High-liquidity ETFs
- Core-satellite portfolio structure
- Rule-based risk control
- Explicit benchmark comparison
- Consistent observation, evaluation, and backtesting

Portfolio structure:

- Core layer: 80%–90%
  - Long-term strategic asset allocation
  - Use only highly liquid, broad-market ETFs

- Satellite layer: 10%–20%
  - Tactical tilts
  - Risk expression
  - Trend / credit / commodity / regional views

- Overlay layer:
  - Rebalancing rules
  - Risk switch
  - Trend filter
  - Volatility / drawdown guardrails

---

## 3. Implementation Principles

The system must follow these implementation priorities:

1. First build a correct and auditable pandas-based engine
2. Then optionally build a vectorized high-performance version
3. Start with monthly and quarterly allocation logic
4. Avoid unnecessary complexity in phase 1
5. Every critical function must be unit-testable
6. All return calculations must use adjusted prices / total return proxies
7. No forward-looking bias
8. Signal date and execution date must be separated clearly
9. Transaction costs must be explicitly modeled
10. Missing data handling must be explicit and logged

---

## 4. Initial ETF Universe

Phase 1 should use a compact set of highly liquid ETFs.

### 4.1 Core universe (Phase 1)
- VTI   # US equity
- VEA   # Developed markets ex-US equity
- IEMG  # Emerging markets equity
- AGG   # US aggregate bond
- IAGG  # International aggregate bond
- GLD   # Gold
- VNQ   # US REITs

### 4.2 Extended universe (Phase 2)
- TLT   # Long-duration US Treasury
- TIP   # US TIPS
- DBC   # Broad commodities
- HYG   # High yield credit
- EMB   # Emerging market bonds
- SGOV  # Cash proxy

### 4.3 Universe requirements
Each ETF should ideally have:
- Sufficient AUM
- Strong daily trading volume
- Tight spreads
- Long enough history
- Stable data availability
- Clear asset classification

Phase 1 should focus on daily dollar volume and data stability rather than turnover ratio, because turnover may not be consistently available from all free sources.

---

## 5. Portfolio Templates

The system must support at least three strategic template portfolios.

### 5.1 Conservative portfolio
- 25% VTI
- 10% VEA
- 5% IEMG
- 30% AGG
- 15% IAGG
- 10% GLD
- 5% VNQ

### 5.2 Balanced portfolio (default)
- 35% VTI
- 15% VEA
- 10% IEMG
- 20% AGG
- 10% IAGG
- 5% GLD
- 5% VNQ

### 5.3 Growth portfolio
- 45% VTI
- 20% VEA
- 10% IEMG
- 10% AGG
- 5% IAGG
- 5% GLD
- 5% VNQ

### 5.4 Notes
- Balanced portfolio should be treated as the default strategic portfolio
- All templates must be configurable via YAML
- Later extensions may include all-weather-like or risk parity portfolios

---

## 6. Benchmark Design

The platform must always evaluate strategies relative to explicit benchmarks.

### 6.1 Benchmark A: Traditional US 60/40
- 60% VTI
- 40% AGG

### 6.2 Benchmark B: Global static diversified benchmark
- 35% VTI
- 15% VEA
- 10% IEMG
- 20% AGG
- 10% IAGG
- 5% GLD
- 5% VNQ

### 6.3 Benchmark C: Target strategy benchmark
- Core-satellite portfolio with rules-based rebalancing and overlays

### 6.4 Benchmark comparison metrics
All strategy reports must compare against Benchmarks A and B at minimum:
- CAGR
- Annualized volatility
- Sharpe ratio
- Max drawdown
- Calmar ratio
- Turnover
- Cost drag

---

## 7. Monitoring / Observation System

The observation system must be structured into three layers.

### 7.1 Market-level monitoring
Track:
- US 10Y yield
- US 2Y yield
- 2s10s slope
- USD index
- VIX
- Gold price
- Oil price
- Credit spread proxies
- Optional inflation / breakeven indicators

### 7.2 Asset-level monitoring
For each ETF track:
- Price
- Adjusted price
- Daily returns
- Volume
- Daily dollar volume
- Rolling volatility
- Rolling drawdown
- Rolling correlations
- Data completeness
- Optional AUM and expense ratio if available

### 7.3 Portfolio-level monitoring
Track:
- Current weights
- Target weights
- Weight deviations
- Portfolio NAV
- Daily returns
- Rolling volatility
- Rolling Sharpe
- Max drawdown
- Risk contribution
- Marginal contribution to risk (MCTR)
- Equity/bond correlation regime
- Rebalance trigger status

### 7.4 Monitoring frequency
- Daily: prices, vol, drawdowns, risk alerts
- Weekly: macro regime review, trend status, market condition summary
- Monthly: rebalance review, performance report, trigger evaluation

---

## 8. Evaluation / Scoring Framework

The system must evaluate both ETFs and portfolios.

### 8.1 ETF scoring (100 points)
- Liquidity: 25
- Risk-return characteristics: 25
- Diversification contribution: 20
- Cost and tax efficiency: 15
- Data quality and history length: 10
- Strategy fit: 5

### 8.2 Portfolio scoring (100 points)
- Return: 25
- Risk control: 25
- Risk-adjusted return: 20
- Stability: 15
- Executability: 15

### 8.3 Core evaluation metrics
#### Return metrics
- Cumulative return
- Annualized return
- Rolling 3Y return
- Monthly win rate

#### Risk metrics
- Annualized volatility
- Downside volatility
- Max drawdown
- 95% VaR
- Expected shortfall (optional in phase 1)

#### Risk-adjusted metrics
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Information ratio (relative to benchmark)

#### Stability metrics
- Monthly hit rate
- Annual hit rate
- Drawdown recovery time
- Rolling Sharpe stability

#### Executability metrics
- Average dollar volume
- Rolling dollar volume stability
- Data completeness
- ETF history length
- Expense ratio
- Availability of substitutes

---

## 9. Rebalancing and Trigger Rules

Rules must be deterministic and backtestable.

### 9.1 Standard rebalance
- Quarterly rebalance by default

### 9.2 Weight drift rule
- Trigger rebalance if any asset weight deviates by more than 20% of its target weight

Example:
- target = 10%
- trigger if actual > 12% or actual < 8%

### 9.3 Risk switch rule
- If 20-day annualized portfolio volatility exceeds a defined threshold, shift part of risky assets into AGG and/or GLD

### 9.4 Trend filter rule
- If an equity ETF falls below its 10-month moving average, reduce its weight by 50%
- Restore original weight when price moves back above the moving average

### 9.5 Recovery logic
- Reinstatement rules must be explicit and symmetric where possible

### 9.6 Transaction costs
- Default one-way trading cost: 5 bps
- Costs must be deducted based on turnover traded

---

## 10. Backtesting Framework

The backtesting system should have three layers.

### 10.1 Layer 1: Core static backtests
- Monthly and quarterly rebalance
- Fixed strategic weights
- Adjusted prices required
- Transaction costs included
- Output full return and drawdown series

### 10.2 Layer 2: Robustness tests
- Different rebalance frequencies
- Different start dates
- Different cost assumptions
- Different weight schemes
- Threshold-based rebalance vs calendar rebalance

### 10.3 Layer 3: Strategy extensions
- Trend filter
- Risk switch
- Inverse-volatility weighting
- Risk parity
- Black-Litterman
- Credit-spread based defensive tilt for HYG / EMB (phase 2)

### 10.4 Required backtest outputs
- NAV time series
- Daily return series
- Drawdown series
- Weight history
- Turnover history
- Transaction cost history
- Annual return table
- Rolling Sharpe
- Rolling volatility
- Relative performance vs benchmark

---

## 11. Data Handling Requirements

### 11.1 Data source
Phase 1:
- yfinance or equivalent free source

Phase 2:
- optional integration with more robust commercial source

### 11.2 Required data fields
- Open
- High
- Low
- Close
- Adj Close
- Volume

### 11.3 Data rules
- Use adjusted prices for return calculations
- Forward-fill prices only when appropriate and explicitly logged
- Never fabricate non-zero volume
- Missing values must be reported
- Data cleaning functions must be reusable and tested

### 11.4 Liquidity screening logic
At minimum compute:
- Daily dollar volume = Adj Close * Volume
- Full-sample average daily dollar volume
- 30-day rolling average daily dollar volume
- Recent liquidity stability ratio

Suggested initial filter:
- Full-sample average daily dollar volume > 50,000,000 USD
- In the most recent 60 trading days, at least 80% of days satisfy 30-day rolling average daily dollar volume > 50,000,000 USD

---

## 12. Project Structure

The repository should follow this structure:

global_etf_allocation/
├── config/
│   ├── etf_universe.yaml
│   ├── asset_mapping.yaml
│   ├── portfolio_templates.yaml
│   ├── benchmark_config.yaml
│   ├── rebalance_rules.yaml
│   └── risk_limits.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── macro/
├── src/
│   ├── data/
│   │   ├── fetch_prices.py
│   │   ├── fetch_macro_data.py
│   │   ├── fetch_etf_metadata.py
│   │   └── clean_data.py
│   ├── universe/
│   │   ├── liquidity_filter.py
│   │   ├── universe_builder.py
│   │   └── etf_scoring.py
│   ├── analytics/
│   │   ├── returns.py
│   │   ├── risk.py
│   │   ├── drawdown.py
│   │   ├── correlation.py
│   │   └── attribution.py
│   ├── portfolio/
│   │   ├── weights.py
│   │   ├── saa.py
│   │   ├── rebalancer.py
│   │   ├── transaction_cost.py
│   │   ├── risk_parity.py
│   │   └── black_litterman.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── vectorized_engine.py
│   │   ├── scenarios.py
│   │   └── stress_test.py
│   ├── dashboard/
│   │   ├── plots.py
│   │   ├── rolling_metrics.py
│   │   └── macro_dashboard.py
│   └── utils/
│       ├── logger.py
│       ├── dates.py
│       └── validators.py
├── tests/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
├── requirements.txt
├── README.md
└── run_pipeline.py

---

## 13. Module Requirements

### 13.1 Data module
Must:
- Download price and volume data
- Save raw data locally
- Clean missing data
- Return standardized DataFrames

### 13.2 Universe module
Must:
- Apply liquidity filters
- Build valid investable universe
- Generate liquidity summary table
- Support ETF scoring

### 13.3 Analytics module
Must:
- Compute returns
- Compute annualized statistics
- Compute Sharpe / Sortino / drawdown
- Compute covariance and correlation matrices
- Compute rolling metrics
- Compute risk contribution and MCTR

### 13.4 Portfolio module
Must:
- Load portfolio templates
- Normalize weights
- Apply rebalancing rules
- Support static SAA
- Support later risk parity / Black-Litterman extensions

### 13.5 Backtest module
Must:
- Run pandas-based auditable backtests
- Support calendar and threshold-based rebalance
- Include transaction costs
- Export all key time series

### 13.6 Dashboard module
Must generate:
- NAV chart
- Drawdown chart
- Correlation heatmap
- Rolling correlation chart
- Rolling Sharpe chart
- Annual return bar chart
- Risk contribution chart
- MCTR chart

---

## 14. Development Phases

### Phase 1: Foundation
Build:
- project structure
- config files
- data downloader
- data cleaner
- returns functions

### Phase 2: Universe and metrics
Build:
- liquidity filter
- ETF summary table
- annualized metrics
- drawdown and Sharpe calculations
- covariance / correlation outputs

### Phase 3: Core backtesting
Build:
- fixed-weight pandas backtest engine
- quarterly rebalance
- transaction cost handling
- benchmark comparisons

### Phase 4: Reporting
Build:
- charts
- tables
- reporting outputs
- pipeline runner

### Phase 5: Strategy overlays
Build:
- trend filter
- risk switch
- threshold rebalance
- scenario testing

### Phase 6: Optimization
Build:
- risk parity
- Black-Litterman
- sensitivity and robustness analysis

---

## 15. Coding Standards

All code generated must follow these standards:

1. Use Python type hints
2. Write docstrings for public functions
3. Use logging for key pipeline steps
4. Use pure functions where practical
5. Keep modules narrowly scoped
6. Avoid hidden side effects
7. Raise informative exceptions
8. Write pytest unit tests for critical functions
9. Keep plotting functions separate from analytics logic
10. Keep config-driven behavior in YAML, not hard-coded

---

## 16. Testing Requirements

At minimum test:
- return calculations
- drawdown calculations
- annualization logic
- liquidity filter logic
- rebalance logic
- transaction cost deduction
- weight normalization
- benchmark comparison calculations

Tests should include:
- simple deterministic toy examples
- missing data cases
- edge cases (single asset, zero returns, extreme drawdown)

---

## 17. Phase 1 Deliverables

Phase 1 must produce:

1. Working repository structure
2. Clean price data pipeline
3. Liquidity summary table
4. Balanced portfolio backtest
5. Benchmark A and B backtests
6. Output tables:
   - performance summary
   - annual return table
   - turnover summary
7. Output charts:
   - NAV
   - drawdown
   - annual return bars
   - correlation heatmap
8. Unit tests for core functions

---

## 18. Future Extensions

The architecture should support future additions including:
- vectorbt implementation
- Streamlit dashboard
- macro regime classification
- Black-Litterman views
- risk parity optimization
- walk-forward testing
- parameter sweep framework
- optional multi-currency support
- optional tax-aware implementation
- integration with portfolio execution tools

---

## 19. Immediate Task for Codex

Start with Phase 1 only.

Do not implement all advanced features at once.

Priority order:
1. Project scaffolding
2. Config files
3. Data downloader
4. Data cleaner
5. Liquidity filter
6. Return/risk analytics
7. Pandas backtest engine
8. Basic reporting

Risk parity and Black-Litterman should be added only after the core system works correctly.
