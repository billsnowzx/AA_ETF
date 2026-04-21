"""Minimal local dashboard app for browsing generated research outputs."""

from __future__ import annotations

import argparse
import json
import webbrowser
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer

import pandas as pd


def _read_csv_if_exists(path: Path, index_col: int | str | None = 0) -> pd.DataFrame:
    """Read a CSV when it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=index_col)


def _read_json_if_exists(path: Path) -> dict:
    """Read a JSON file when it exists, otherwise return an empty dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_manifest_summary(manifest: dict) -> pd.DataFrame:
    """Build a compact dashboard table from the pipeline manifest."""
    if not manifest:
        return pd.DataFrame()

    date_range = manifest.get("date_range", {})
    parameters = manifest.get("parameters", {})
    universes = manifest.get("universes", {})
    strategy = manifest.get("strategy", {})
    config_files = manifest.get("config_files", {})
    rows = {
        "run_completed_at": manifest.get("run_completed_at", "n/a"),
        "date_range": f"{date_range.get('start', 'n/a')} to {date_range.get('end', 'n/a')}",
        "backtest_universe_mode": parameters.get("backtest_universe_mode", "n/a"),
        "rolling_window": parameters.get("rolling_window", "n/a"),
        "strategy": strategy.get("name", "n/a"),
        "ending_nav": strategy.get("ending_nav", "n/a"),
        "enabled_tickers": ", ".join(universes.get("enabled_tickers", [])),
        "liquid_tickers": ", ".join(universes.get("liquid_tickers", [])),
        "backtest_tickers": ", ".join(universes.get("backtest_tickers", [])),
    }
    rows.update(
        {
            f"config_{name}": path
            for name, path in sorted(config_files.items())
        }
    )
    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


def _format_percent(value: object) -> str:
    """Format a decimal value as a percent string when numeric."""
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_decimal(value: object, digits: int = 4) -> str:
    """Format a numeric value as a fixed-point decimal string when numeric."""
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_integer(value: object) -> str:
    """Format a numeric value as an integer string when numeric."""
    if pd.isna(value):
        return "n/a"
    return f"{int(round(float(value)))}"


def _format_dashboard_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Format dashboard tables into human-readable display values."""
    formatted = {name: frame.copy() for name, frame in tables.items()}

    if "performance_summary" in formatted:
        frame = formatted["performance_summary"]
        for column in [
            "annualized_return",
            "annualized_volatility",
            "downside_volatility",
            "max_drawdown",
            "total_transaction_cost_drag",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_percent)
        for column in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "ending_nav", "total_turnover"]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_decimal)

    if "benchmark_comparisons" in formatted:
        frame = formatted["benchmark_comparisons"]
        for column in [
            "strategy_annualized_return",
            "benchmark_annualized_return",
            "annualized_excess_return",
            "tracking_error",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_percent)
        if "information_ratio" in frame.columns:
            frame["information_ratio"] = frame["information_ratio"].map(_format_decimal)

    for name in ["benchmark_annual_excess_returns", "benchmark_drawdown_comparisons"]:
        if name in formatted:
            frame = formatted[name]
            for column in frame.columns:
                frame[column] = frame[column].map(_format_percent)

    if "top_correlation_pairs" in formatted:
        frame = formatted["top_correlation_pairs"]
        if "correlation" in frame.columns:
            frame["correlation"] = frame["correlation"].map(_format_decimal)

    if "asset_risk_snapshot" in formatted:
        frame = formatted["asset_risk_snapshot"]
        if "avg_correlation" in frame.columns:
            frame["avg_correlation"] = frame["avg_correlation"].map(_format_decimal)
        if "variance" in frame.columns:
            frame["variance"] = frame["variance"].map(lambda value: _format_decimal(value, digits=6))

    if "etf_summary" in formatted:
        frame = formatted["etf_summary"]
        for column in ["average_dollar_volume", "latest_rolling_average_dollar_volume"]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_integer)
        if "recent_pass_ratio" in frame.columns:
            frame["recent_pass_ratio"] = frame["recent_pass_ratio"].map(_format_percent)
        for column in ["liquidity_score", "data_quality_score", "strategy_fit_score", "phase1_total_score"]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_decimal)
        if "phase1_score_pct" in frame.columns:
            frame["phase1_score_pct"] = frame["phase1_score_pct"].map(_format_percent)
        if "observations" in frame.columns:
            frame["observations"] = frame["observations"].map(_format_integer)

    if "data_quality_summary" in formatted:
        frame = formatted["data_quality_summary"]
        for column in ["observations", "missing_adj_close", "missing_volume", "zero_volume", "missing_dollar_volume"]:
            if column in frame.columns:
                frame[column] = frame[column].map(_format_integer)

    if "rolling_volatility" in formatted:
        frame = formatted["rolling_volatility"]
        for column in frame.columns:
            frame[column] = frame[column].map(_format_percent)

    if "rolling_sharpe" in formatted:
        frame = formatted["rolling_sharpe"]
        for column in frame.columns:
            frame[column] = frame[column].map(_format_decimal)

    if "manifest_summary" in formatted:
        frame = formatted["manifest_summary"].astype(object)
        if "value" in frame.columns and "ending_nav" in frame.index:
            frame.loc["ending_nav", "value"] = _format_decimal(frame.loc["ending_nav", "value"])
        formatted["manifest_summary"] = frame

    return formatted


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame to a simple HTML table."""
    if frame.empty:
        return "<p>No data available.</p>"

    table = frame.copy()
    header_cells = "".join(f"<th>{escape(str(column))}</th>" for column in ["index", *table.columns.tolist()])
    body_rows: list[str] = []

    for index, row in table.iterrows():
        values = [index, *row.tolist()]
        body_rows.append(
            "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>"
        )

    return "<table><thead><tr>" + header_cells + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"


def build_dashboard_html(
    output_dir: str | Path = "outputs/tables",
    figure_dir: str | Path = "outputs/figures",
    report_dir: str | Path = "outputs/reports",
) -> str:
    """Build a self-contained dashboard page from generated outputs."""
    output_path = Path(output_dir)
    figure_path = Path(figure_dir)
    report_path = Path(report_dir)

    performance_summary = _read_csv_if_exists(output_path / "performance_summary.csv")
    benchmark_comparisons = _read_csv_if_exists(output_path / "benchmark_comparisons.csv")
    annual_excess = _read_csv_if_exists(output_path / "benchmark_annual_excess_returns.csv")
    drawdown_comparisons = _read_csv_if_exists(output_path / "benchmark_drawdown_comparisons.csv")
    top_correlations = _read_csv_if_exists(output_path / "top_correlation_pairs.csv", index_col=None)
    asset_risk_snapshot = _read_csv_if_exists(output_path / "asset_risk_snapshot.csv")
    etf_summary = _read_csv_if_exists(output_path / "etf_summary.csv")
    data_quality_summary = _read_csv_if_exists(output_path / "data_quality_summary.csv")
    run_configuration = _read_csv_if_exists(output_path / "run_configuration.csv")
    rolling_volatility = _read_csv_if_exists(output_path / "rolling_volatility.csv")
    rolling_sharpe = _read_csv_if_exists(output_path / "rolling_sharpe.csv")
    manifest_summary = _build_manifest_summary(_read_json_if_exists(output_path / "pipeline_manifest.json"))
    formatted_tables = _format_dashboard_tables(
        {
            "performance_summary": performance_summary,
            "benchmark_comparisons": benchmark_comparisons,
            "benchmark_annual_excess_returns": annual_excess,
            "benchmark_drawdown_comparisons": drawdown_comparisons,
            "top_correlation_pairs": top_correlations,
            "asset_risk_snapshot": asset_risk_snapshot,
            "etf_summary": etf_summary,
            "data_quality_summary": data_quality_summary,
            "run_configuration": run_configuration,
            "rolling_volatility": rolling_volatility.tail(5),
            "rolling_sharpe": rolling_sharpe.tail(5),
            "manifest_summary": manifest_summary,
        }
    )
    performance_summary = formatted_tables["performance_summary"]
    benchmark_comparisons = formatted_tables["benchmark_comparisons"]
    annual_excess = formatted_tables["benchmark_annual_excess_returns"]
    drawdown_comparisons = formatted_tables["benchmark_drawdown_comparisons"]
    top_correlations = formatted_tables["top_correlation_pairs"]
    asset_risk_snapshot = formatted_tables["asset_risk_snapshot"]
    etf_summary = formatted_tables["etf_summary"]
    data_quality_summary = formatted_tables["data_quality_summary"]
    run_configuration = formatted_tables["run_configuration"]
    rolling_volatility = formatted_tables["rolling_volatility"]
    rolling_sharpe = formatted_tables["rolling_sharpe"]
    manifest_summary = formatted_tables["manifest_summary"]

    report_links = []
    for report_name in ["balanced_phase1_report.html", "balanced_phase1_report.md"]:
        report_file = report_path / report_name
        if report_file.exists():
            report_links.append(
                f'<li><a href="../reports/{escape(report_name)}" target="_blank">{escape(report_name)}</a></li>'
            )
    report_links_html = "".join(report_links) if report_links else "<li>No reports available.</li>"

    figure_cards = []
    for title, filename in [
        ("NAV", "balanced_nav.png"),
        ("Drawdown", "balanced_drawdown.png"),
        ("Annual Returns", "balanced_annual_returns.png"),
        ("Rolling Volatility", "balanced_rolling_volatility.png"),
        ("Rolling Sharpe", "balanced_rolling_sharpe.png"),
        ("Correlation Heatmap", "correlation_heatmap.png"),
    ]:
        figure_file = figure_path / filename
        if figure_file.exists():
            figure_cards.append(
                f"""
                <section class="card chart">
                  <h3>{escape(title)}</h3>
                  <img src="../figures/{escape(filename)}" alt="{escape(title)}" />
                </section>
                """
            )
    figure_cards_html = "".join(figure_cards) if figure_cards else "<p>No figures available.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AA ETF Dashboard</title>
  <style>
    :root {{
      --paper: #f4efe5;
      --ink: #1f2933;
      --muted: #52606d;
      --panel: #fffdf8;
      --line: #d9d0bb;
      --accent: #7c3f00;
    }}
    body {{
      margin: 0;
      font-family: Georgia, 'Times New Roman', serif;
      background: radial-gradient(circle at top left, #fff8ec, var(--paper));
      color: var(--ink);
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      padding: 24px 28px;
      background: linear-gradient(135deg, #fffaf0, #efe4d1);
      border: 1px solid var(--line);
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{
      color: #102a43;
      margin-top: 0;
    }}
    p, li {{
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px;
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #efe7d3;
    }}
    .chart img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      background: white;
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>AA ETF Research Dashboard</h1>
      <p>Local dashboard for browsing the latest pipeline outputs, benchmark-relative tables, figures, and reports.</p>
      <ul>{report_links_html}</ul>
    </section>

    <div class="grid">
      <section class="card">
        <h2>Run Manifest</h2>
        {dataframe_to_html_table(manifest_summary)}
      </section>
      <section class="card">
        <h2>Run Configuration</h2>
        {dataframe_to_html_table(run_configuration)}
      </section>
      <section class="card">
        <h2>Performance Summary</h2>
        {dataframe_to_html_table(performance_summary)}
      </section>
      <section class="card">
        <h2>Benchmark Comparisons</h2>
        {dataframe_to_html_table(benchmark_comparisons)}
      </section>
      <section class="card">
        <h2>Benchmark Annual Excess Returns</h2>
        {dataframe_to_html_table(annual_excess)}
      </section>
      <section class="card">
        <h2>Benchmark Drawdown Comparisons</h2>
        {dataframe_to_html_table(drawdown_comparisons)}
      </section>
      <section class="card">
        <h2>Top Correlation Pairs</h2>
        {dataframe_to_html_table(top_correlations)}
      </section>
      <section class="card">
        <h2>Asset Risk Snapshot</h2>
        {dataframe_to_html_table(asset_risk_snapshot)}
      </section>
      <section class="card">
        <h2>ETF Summary</h2>
        {dataframe_to_html_table(etf_summary)}
      </section>
      <section class="card">
        <h2>Data Quality Summary</h2>
        {dataframe_to_html_table(data_quality_summary)}
      </section>
      <section class="card">
        <h2>Latest Rolling Volatility</h2>
        {dataframe_to_html_table(rolling_volatility)}
      </section>
      <section class="card">
        <h2>Latest Rolling Sharpe</h2>
        {dataframe_to_html_table(rolling_sharpe)}
      </section>
    </div>

    <div class="grid">
      {figure_cards_html}
    </div>
  </div>
</body>
</html>
"""


def write_dashboard_html(
    output_path: str | Path = "outputs/reports/dashboard.html",
    output_dir: str | Path = "outputs/tables",
    figure_dir: str | Path = "outputs/figures",
    report_dir: str | Path = "outputs/reports",
) -> Path:
    """Write the dashboard page to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = build_dashboard_html(output_dir=output_dir, figure_dir=figure_dir, report_dir=report_dir)
    destination.write_text(html, encoding="utf-8")
    return destination


def open_dashboard_html(output_path: str | Path = "outputs/reports/dashboard.html") -> str:
    """Open the generated dashboard HTML file and return its file URL."""
    url = Path(output_path).resolve().as_uri()
    webbrowser.open(url)
    return url


def run_dashboard_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    directory: str | Path = "outputs",
) -> None:
    """Serve the output directory locally for dashboard browsing."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    TCPServer.allow_reuse_address = True
    with ThreadingHTTPServer((host, port), Handler) as httpd:
        print(f"Serving dashboard at http://{host}:{port}/reports/dashboard.html")
        httpd.serve_forever()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the local dashboard app."""
    parser = argparse.ArgumentParser(description="Run the local AA ETF dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--dashboard-path", default="outputs/reports/dashboard.html")
    parser.add_argument("--open", action="store_true", help="Open the generated dashboard in the default browser.")
    parser.add_argument("--no-server", action="store_true", help="Generate the dashboard and exit without serving HTTP.")
    return parser


def main() -> None:
    """Generate and serve the local dashboard."""
    parser = build_argument_parser()
    args = parser.parse_args()

    write_dashboard_html(
        output_path=args.dashboard_path,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
    )
    if args.open:
        open_dashboard_html(args.dashboard_path)
    if args.no_server:
        return
    run_dashboard_server(host=args.host, port=args.port, directory=Path(args.dashboard_path).parent.parent)


if __name__ == "__main__":
    main()
