import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
RISK_FREE_RATE = 0.05
NUM_PORTFOLIOS = 10000
TOP_N = 10
TRADING_DAYS = 252
MIN_OBSERVATIONS = 120
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# DISPLAY SETTINGS
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# =========================
# HELPER FUNCTIONS
# =========================
def ensure_series(obj, name="value"):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0]
            s.name = name
            return s
        raise ValueError(f"{name} expected Series or 1-column DataFrame, but got {obj.shape[1]} columns.")
    if isinstance(obj, pd.Series):
        return obj
    raise TypeError(f"{name} must be a pandas Series or 1-column DataFrame.")

def safe_divide(numerator, denominator):
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator

def calculate_total_return(price_series: pd.Series) -> float:
    price_series = ensure_series(price_series, "price_series").dropna()
    if len(price_series) < 2:
        return np.nan
    return (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100

def annualized_volatility(daily_returns: pd.Series) -> float:
    daily_returns = ensure_series(daily_returns, "daily_returns").dropna()
    if len(daily_returns) < 2:
        return np.nan
    return daily_returns.std() * np.sqrt(TRADING_DAYS) * 100

def calculate_max_drawdown(price_series: pd.Series) -> float:
    price_series = ensure_series(price_series, "price_series").dropna()
    if len(price_series) < 2:
        return np.nan
    rolling_max = price_series.cummax()
    drawdown = (price_series / rolling_max) - 1
    return drawdown.min() * 100

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    daily_returns = ensure_series(daily_returns, "daily_returns").dropna()
    if len(daily_returns) < 2:
        return np.nan

    rf_daily = risk_free_rate / TRADING_DAYS
    excess_returns = daily_returns - rf_daily
    annual_excess_return = excess_returns.mean() * TRADING_DAYS
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

    return safe_divide(annual_excess_return, annual_volatility)

def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    daily_returns = ensure_series(daily_returns, "daily_returns").dropna()
    if len(daily_returns) < 2:
        return np.nan

    rf_daily = risk_free_rate / TRADING_DAYS
    excess_returns = daily_returns - rf_daily
    downside_excess = excess_returns[excess_returns < 0]

    downside_deviation = downside_excess.std() * np.sqrt(TRADING_DAYS)
    annual_excess_return = excess_returns.mean() * TRADING_DAYS

    return safe_divide(annual_excess_return, downside_deviation)

def calculate_information_ratio(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    stock_returns = ensure_series(stock_returns, "stock_returns").dropna()
    benchmark_returns = ensure_series(benchmark_returns, "benchmark_returns").dropna()

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return np.nan

    aligned.columns = ["stock", "benchmark"]
    active_returns = aligned["stock"] - aligned["benchmark"]

    annual_active_return = active_returns.mean() * TRADING_DAYS
    tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)

    return safe_divide(annual_active_return, tracking_error)

def calculate_beta_alpha_treynor(stock_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE):
    stock_returns = ensure_series(stock_returns, "stock_returns").dropna()
    benchmark_returns = ensure_series(benchmark_returns, "benchmark_returns").dropna()

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return np.nan, np.nan, np.nan

    aligned.columns = ["stock", "benchmark"]

    cov_matrix = np.cov(aligned["stock"], aligned["benchmark"])
    benchmark_variance = cov_matrix[1, 1]

    if benchmark_variance == 0:
        beta = np.nan
    else:
        beta = cov_matrix[0, 1] / benchmark_variance

    rf_daily = risk_free_rate / TRADING_DAYS
    stock_excess_annual = (aligned["stock"] - rf_daily).mean() * TRADING_DAYS
    benchmark_excess_annual = (aligned["benchmark"] - rf_daily).mean() * TRADING_DAYS

    alpha = stock_excess_annual - (beta * benchmark_excess_annual) if pd.notna(beta) else np.nan
    treynor = safe_divide(stock_excess_annual, beta)

    return beta, alpha * 100 if pd.notna(alpha) else np.nan, treynor

def calculate_calmar_ratio(total_return_pct: float, max_drawdown_pct: float) -> float:
    if pd.isna(total_return_pct) or pd.isna(max_drawdown_pct) or max_drawdown_pct == 0:
        return np.nan
    return (total_return_pct / 100) / abs(max_drawdown_pct / 100)

def save_table(df: pd.DataFrame, filename: str, index: bool = True):
    df.to_csv(OUTPUT_DIR / filename, index=index)

def save_barh(series: pd.Series, title: str, xlabel: str, filename: str, figsize=(10, 6), ascending=True):
    plt.figure(figsize=figsize)
    series.sort_values(ascending=ascending).plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()

def save_bar(series: pd.Series, title: str, ylabel: str, filename: str, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()

# =========================
# 1. LOAD S&P 500 COMPANY LIST
# =========================
print("Loading S&P 500 company list...")

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]

sp500_table["Symbol"] = sp500_table["Symbol"].str.replace(".", "-", regex=False)
tickers = sp500_table["Symbol"].dropna().unique().tolist()
loaded_ticker_count = len(tickers)

print(f"Total S&P 500 tickers loaded: {loaded_ticker_count}")

# =========================
# 2. DOWNLOAD PRICE DATA
# =========================
print("Downloading historical price data...")

raw_data = yf.download(
    tickers,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
    group_by="column"
)

if "Close" not in raw_data.columns.get_level_values(0):
    raise ValueError("Close prices could not be found in downloaded data.")

close_prices = raw_data["Close"].copy()
close_prices = close_prices.dropna(axis=1, how="all").sort_index()

valid_price_tickers = close_prices.columns.tolist()
failed_tickers = sorted(list(set(tickers) - set(valid_price_tickers)))

print(f"Valid tickers with price data: {len(valid_price_tickers)}")
print(f"Failed tickers: {len(failed_tickers)}")

if failed_tickers:
    failed_df = pd.DataFrame({"Failed Tickers": failed_tickers})
    save_table(failed_df, "failed_tickers.csv", index=False)

# =========================
# 3. DOWNLOAD BENCHMARK (SPY)
# =========================
print("Downloading SPY benchmark data...")

spy_raw = yf.download(
    "SPY",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
    group_by="column"
)

spy_close = ensure_series(spy_raw["Close"], name="SPY Close").dropna().sort_index()
spy_daily_returns = spy_close.pct_change().dropna()

if len(spy_daily_returns) < MIN_OBSERVATIONS:
    raise ValueError("SPY benchmark has insufficient observations.")

# =========================
# 4. DAILY RETURNS
# =========================
daily_returns = close_prices.pct_change().dropna(how="all")
daily_returns = daily_returns.dropna(axis=1, how="all")

sufficient_obs_tickers = [
    col for col in daily_returns.columns
    if daily_returns[col].dropna().shape[0] >= MIN_OBSERVATIONS
]

daily_returns = daily_returns[sufficient_obs_tickers]
close_prices = close_prices[sufficient_obs_tickers]
final_analysis_tickers = sufficient_obs_tickers

print(f"Tickers with sufficient observations: {len(final_analysis_tickers)}")

# =========================
# 5. COVERAGE SUMMARY
# =========================
coverage_summary = pd.DataFrame({
    "Stage": [
        "Loaded from Wikipedia",
        "Valid price data",
        "Failed downloads / unavailable",
        "Passed minimum observation filter",
    ],
    "Count": [
        loaded_ticker_count,
        len(valid_price_tickers),
        len(failed_tickers),
        len(final_analysis_tickers),
    ]
})
save_table(coverage_summary, "coverage_summary.csv", index=False)

# =========================
# 6. STOCK-LEVEL METRICS
# =========================
print("Calculating stock-level metrics...")

metrics = []

for ticker in final_analysis_tickers:
    price_series = close_prices[ticker].dropna()
    returns_series = daily_returns[ticker].dropna()

    if len(price_series) < 2 or len(returns_series) < MIN_OBSERVATIONS:
        continue

    total_return_pct = calculate_total_return(price_series)
    vol_pct = annualized_volatility(returns_series)
    sharpe = calculate_sharpe_ratio(returns_series, RISK_FREE_RATE)
    sortino = calculate_sortino_ratio(returns_series, RISK_FREE_RATE)
    max_dd_pct = calculate_max_drawdown(price_series)
    calmar = calculate_calmar_ratio(total_return_pct, max_dd_pct)
    info_ratio = calculate_information_ratio(returns_series, spy_daily_returns)
    beta, alpha_pct, treynor = calculate_beta_alpha_treynor(
        returns_series, spy_daily_returns, RISK_FREE_RATE
    )

    metrics.append({
        "Ticker": ticker,
        "Total Return (%)": total_return_pct,
        "Volatility (%)": vol_pct,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd_pct,
        "Calmar Ratio": calmar,
        "Beta": beta,
        "Alpha (%)": alpha_pct,
        "Treynor Ratio": treynor,
        "Information Ratio": info_ratio
    })

results = pd.DataFrame(metrics).set_index("Ticker")

sector_map = sp500_table.drop_duplicates("Symbol").set_index("Symbol")["GICS Sector"]
subindustry_map = sp500_table.drop_duplicates("Symbol").set_index("Symbol")["GICS Sub-Industry"]

results["Sector"] = results.index.map(sector_map)
results["Sub-Industry"] = results.index.map(subindustry_map)

essential_cols = [
    "Total Return (%)",
    "Volatility (%)",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)",
    "Beta",
    "Alpha (%)",
    "Information Ratio"
]
results = results.dropna(subset=essential_cols)

# =========================
# 7. RANKING MODEL
# =========================
print("Building ranking model...")

results["Return Score"] = results["Total Return (%)"].rank(pct=True)
results["Sharpe Score"] = results["Sharpe Ratio"].rank(pct=True)
results["Sortino Score"] = results["Sortino Ratio"].rank(pct=True)
results["Drawdown Score"] = results["Max Drawdown (%)"].rank(pct=True)
results["Alpha Score"] = results["Alpha (%)"].rank(pct=True)
results["Info Ratio Score"] = results["Information Ratio"].rank(pct=True)
results["Calmar Score"] = results["Calmar Ratio"].rank(pct=True)

results["Composite Score"] = (
    0.22 * results["Return Score"] +
    0.20 * results["Sharpe Score"] +
    0.15 * results["Sortino Score"] +
    0.12 * results["Drawdown Score"] +
    0.13 * results["Alpha Score"] +
    0.08 * results["Info Ratio Score"] +
    0.10 * results["Calmar Score"]
)

# =========================
# 8. SCREENING TABLES
# =========================
best_10_total_return = results.sort_values("Total Return (%)", ascending=False).head(TOP_N)
worst_10_total_return = results.sort_values("Total Return (%)", ascending=True).head(TOP_N)
most_volatile_10 = results.sort_values("Volatility (%)", ascending=False).head(TOP_N)
least_volatile_10 = results.sort_values("Volatility (%)", ascending=True).head(TOP_N)
best_sharpe_10 = results.sort_values("Sharpe Ratio", ascending=False).head(TOP_N)
best_sortino_10 = results.sort_values("Sortino Ratio", ascending=False).head(TOP_N)
worst_drawdown_10 = results.sort_values("Max Drawdown (%)", ascending=True).head(TOP_N)
highest_alpha_10 = results.sort_values("Alpha (%)", ascending=False).head(TOP_N)
highest_beta_10 = results.sort_values("Beta", ascending=False).head(TOP_N)
best_calmar_10 = results.sort_values("Calmar Ratio", ascending=False).head(TOP_N)
best_composite_10 = results.sort_values("Composite Score", ascending=False).head(TOP_N)

# =========================
# 9. SECTOR ANALYSIS
# =========================
sector_summary = (
    results.groupby("Sector")
    .agg({
        "Total Return (%)": "mean",
        "Volatility (%)": "mean",
        "Sharpe Ratio": "mean",
        "Sortino Ratio": "mean",
        "Max Drawdown (%)": "mean",
        "Calmar Ratio": "mean",
        "Alpha (%)": "mean",
        "Information Ratio": "mean",
        "Composite Score": "mean"
    })
    .rename(columns={
        "Total Return (%)": "Avg Total Return (%)",
        "Volatility (%)": "Avg Volatility (%)",
        "Sharpe Ratio": "Avg Sharpe Ratio",
        "Sortino Ratio": "Avg Sortino Ratio",
        "Max Drawdown (%)": "Avg Max Drawdown (%)",
        "Calmar Ratio": "Avg Calmar Ratio",
        "Alpha (%)": "Avg Alpha (%)",
        "Information Ratio": "Avg Information Ratio",
        "Composite Score": "Avg Composite Score"
    })
    .sort_values("Avg Composite Score", ascending=False)
)

sector_counts = results.groupby("Sector").size().rename("Stock Count")
sector_summary = sector_summary.join(sector_counts)

# =========================
# 10. BENCHMARK SUMMARY
# =========================
spy_total_return = float(calculate_total_return(spy_close))
spy_volatility = float(annualized_volatility(spy_daily_returns))
spy_sharpe = float(calculate_sharpe_ratio(spy_daily_returns, RISK_FREE_RATE))
spy_sortino = float(calculate_sortino_ratio(spy_daily_returns, RISK_FREE_RATE))
spy_drawdown = float(calculate_max_drawdown(spy_close))
spy_calmar = float(calculate_calmar_ratio(spy_total_return, spy_drawdown))

benchmark_summary = pd.DataFrame({
    "Metric": [
        "Total Return (%)",
        "Volatility (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Calmar Ratio"
    ],
    "SPY": [
        spy_total_return,
        spy_volatility,
        spy_sharpe,
        spy_sortino,
        spy_drawdown,
        spy_calmar
    ]
})

# =========================
# 11. BENCHMARK COMPARISON
# =========================
benchmark_comparison = pd.DataFrame({
    "Statistic": [
        "Stocks analyzed",
        "Stocks beating SPY by Total Return",
        "Stocks beating SPY by Sharpe",
        "Stocks with positive Alpha",
        "Stocks with lower volatility than SPY",
        "Stocks with smaller drawdown than SPY"
    ],
    "Value": [
        len(results),
        int((results["Total Return (%)"] > spy_total_return).sum()),
        int((results["Sharpe Ratio"] > spy_sharpe).sum()),
        int((results["Alpha (%)"] > 0).sum()),
        int((results["Volatility (%)"] < spy_volatility).sum()),
        int((results["Max Drawdown (%)"] > spy_drawdown).sum())
    ]
})

# =========================
# 12. EFFICIENT FRONTIER
# =========================
print("Running portfolio simulation...")

ef_candidates = results.sort_values("Composite Score", ascending=False).head(25).index.tolist()
ef_returns = daily_returns[ef_candidates].dropna()

rf_daily = RISK_FREE_RATE / TRADING_DAYS
annual_mean_returns = ef_returns.mean() * TRADING_DAYS
annual_excess_returns = (ef_returns.mean() - rf_daily) * TRADING_DAYS
annual_cov = ef_returns.cov() * TRADING_DAYS

portfolio_results = []

for _ in range(NUM_PORTFOLIOS):
    weights = np.random.random(len(ef_candidates))
    weights /= weights.sum()

    portfolio_return = np.dot(weights, annual_mean_returns)
    portfolio_excess_return = np.dot(weights, annual_excess_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    portfolio_sharpe = safe_divide(portfolio_excess_return, portfolio_volatility)

    portfolio_results.append([
        portfolio_return * 100,
        portfolio_volatility * 100,
        portfolio_sharpe,
        *weights
    ])

portfolio_columns = [
    "Portfolio Return (%)",
    "Portfolio Volatility (%)",
    "Portfolio Sharpe Ratio"
] + ef_candidates

portfolio_df = pd.DataFrame(portfolio_results, columns=portfolio_columns).dropna()

max_sharpe_portfolio = portfolio_df.sort_values("Portfolio Sharpe Ratio", ascending=False).head(1)
min_vol_portfolio = portfolio_df.sort_values("Portfolio Volatility (%)", ascending=True).head(1)

# =========================
# 13. EXPORT TABLES
# =========================
print("Saving CSV outputs...")

save_table(results.sort_values("Composite Score", ascending=False), "sp500_screening_results.csv")
save_table(best_10_total_return, "top_10_total_return.csv")
save_table(worst_10_total_return, "top_10_worst_total_return.csv")
save_table(most_volatile_10, "top_10_most_volatile.csv")
save_table(least_volatile_10, "top_10_least_volatile.csv")
save_table(best_sharpe_10, "top_10_sharpe.csv")
save_table(best_sortino_10, "top_10_sortino.csv")
save_table(worst_drawdown_10, "top_10_worst_drawdown.csv")
save_table(highest_alpha_10, "top_10_alpha.csv")
save_table(highest_beta_10, "top_10_beta.csv")
save_table(best_calmar_10, "top_10_calmar.csv")
save_table(best_composite_10, "top_10_composite.csv")
save_table(sector_summary, "sector_summary.csv")
save_table(benchmark_summary, "benchmark_summary.csv", index=False)
save_table(benchmark_comparison, "benchmark_comparison.csv", index=False)
save_table(portfolio_df, "efficient_frontier_portfolios.csv", index=False)
save_table(max_sharpe_portfolio, "max_sharpe_portfolio.csv", index=False)
save_table(min_vol_portfolio, "min_volatility_portfolio.csv", index=False)

# =========================
# 14. CHARTS
# =========================
print("Saving charts...")

save_barh(
    best_10_total_return["Total Return (%)"],
    "Top 10 S&P 500 Stocks by Total Return",
    "Total Return (%)",
    "top_10_total_return.png",
    ascending=True
)

save_barh(
    best_sharpe_10["Sharpe Ratio"],
    "Top 10 Stocks by Sharpe Ratio",
    "Sharpe Ratio",
    "top_10_sharpe.png",
    ascending=True
)

save_barh(
    best_sortino_10["Sortino Ratio"],
    "Top 10 Stocks by Sortino Ratio",
    "Sortino Ratio",
    "top_10_sortino.png",
    ascending=True
)

save_barh(
    highest_alpha_10["Alpha (%)"],
    "Top 10 Stocks by Alpha",
    "Alpha (%)",
    "top_10_alpha.png",
    ascending=True
)

save_barh(
    best_calmar_10["Calmar Ratio"],
    "Top 10 Stocks by Calmar Ratio",
    "Calmar Ratio",
    "top_10_calmar.png",
    ascending=True
)

save_barh(
    best_composite_10["Composite Score"],
    "Top 10 Stocks by Composite Score",
    "Composite Score",
    "top_10_composite.png",
    ascending=True
)

save_bar(
    sector_summary["Avg Total Return (%)"].sort_values(ascending=False),
    "Average Sector Total Return",
    "Average Total Return (%)",
    "sector_avg_total_return.png"
)

save_bar(
    sector_summary["Avg Sharpe Ratio"].sort_values(ascending=False),
    "Average Sector Sharpe Ratio",
    "Average Sharpe Ratio",
    "sector_avg_sharpe.png"
)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    portfolio_df["Portfolio Volatility (%)"],
    portfolio_df["Portfolio Return (%)"],
    c=portfolio_df["Portfolio Sharpe Ratio"],
    cmap="viridis",
    alpha=0.45
)

plt.scatter(
    max_sharpe_portfolio["Portfolio Volatility (%)"],
    max_sharpe_portfolio["Portfolio Return (%)"],
    marker="*",
    s=250,
    label="Max Sharpe Portfolio"
)

plt.scatter(
    min_vol_portfolio["Portfolio Volatility (%)"],
    min_vol_portfolio["Portfolio Return (%)"],
    marker="D",
    s=120,
    label="Min Volatility Portfolio"
)

plt.colorbar(scatter, label="Portfolio Sharpe Ratio")
plt.title("Efficient Frontier")
plt.xlabel("Portfolio Volatility (%)")
plt.ylabel("Portfolio Return (%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "efficient_frontier.png", dpi=200)
plt.close()

heatmap_tickers = best_composite_10.index.tolist()
corr_matrix = daily_returns[heatmap_tickers].corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(heatmap_tickers)), heatmap_tickers, rotation=90)
plt.yticks(range(len(heatmap_tickers)), heatmap_tickers)
plt.title("Correlation Matrix (Top Composite Score Stocks)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
results["Total Return (%)"].plot(kind="hist", bins=40)
plt.title("Distribution of Total Returns")
plt.xlabel("Total Return (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_return_distribution.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
results["Sharpe Ratio"].plot(kind="hist", bins=40)
plt.title("Distribution of Sharpe Ratios")
plt.xlabel("Sharpe Ratio")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sharpe_distribution.png", dpi=200)
plt.close()

# =========================
# 15. TEXT REPORT
# =========================
print("Writing summary report...")

report_lines = []
report_lines.append("S&P 500 STOCK SCREENING & PORTFOLIO ANALYTICS REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Analysis Period: {START_DATE} to {END_DATE}")
report_lines.append(f"Risk-Free Rate Assumption: {RISK_FREE_RATE:.2%}")
report_lines.append(f"Trading Days Assumption: {TRADING_DAYS}")
report_lines.append(f"Loaded from Wikipedia: {loaded_ticker_count}")
report_lines.append(f"Valid price data: {len(valid_price_tickers)}")
report_lines.append(f"Failed downloads: {len(failed_tickers)}")
report_lines.append(f"Passed minimum observation filter: {len(final_analysis_tickers)}")
report_lines.append(f"Final stocks in results table: {len(results)}")
report_lines.append("")

report_lines.append("IMPORTANT NOTE")
report_lines.append("-" * 80)
report_lines.append("This analysis uses the current S&P 500 constituent list from Wikipedia.")
report_lines.append("Some tickers are excluded due to missing historical price data or insufficient observations.")
report_lines.append("This project should be interpreted as a practical stock screening and portfolio analytics system.")
report_lines.append("")

report_lines.append("SPY BENCHMARK METRICS")
report_lines.append("-" * 80)
report_lines.append(f"Total Return (%): {spy_total_return:.2f}")
report_lines.append(f"Volatility (%): {spy_volatility:.2f}")
report_lines.append(f"Sharpe Ratio: {spy_sharpe:.2f}")
report_lines.append(f"Sortino Ratio: {spy_sortino:.2f}")
report_lines.append(f"Max Drawdown (%): {spy_drawdown:.2f}")
report_lines.append(f"Calmar Ratio: {spy_calmar:.2f}")
report_lines.append("")

report_lines.append("BENCHMARK COMPARISON")
report_lines.append("-" * 80)
for _, row in benchmark_comparison.iterrows():
    report_lines.append(f"{row['Statistic']}: {row['Value']}")
report_lines.append("")

report_lines.append("TOP 5 STOCKS BY COMPOSITE SCORE")
report_lines.append("-" * 80)
for ticker, row in best_composite_10.head(5).iterrows():
    report_lines.append(
        f"{ticker}: Total Return={row['Total Return (%)']:.2f}%, "
        f"Sharpe={row['Sharpe Ratio']:.2f}, "
        f"Sortino={row['Sortino Ratio']:.2f}, "
        f"Alpha={row['Alpha (%)']:.2f}%, "
        f"Vol={row['Volatility (%)']:.2f}%, "
        f"MaxDD={row['Max Drawdown (%)']:.2f}%, "
        f"Sector={row['Sector']}"
    )
report_lines.append("")

report_lines.append("TOP 5 STOCKS BY SHARPE RATIO")
report_lines.append("-" * 80)
for ticker, row in best_sharpe_10.head(5).iterrows():
    report_lines.append(
        f"{ticker}: Sharpe={row['Sharpe Ratio']:.2f}, "
        f"Total Return={row['Total Return (%)']:.2f}%, "
        f"Vol={row['Volatility (%)']:.2f}%, "
        f"Alpha={row['Alpha (%)']:.2f}%"
    )
report_lines.append("")

report_lines.append("TOP 5 STOCKS BY ALPHA")
report_lines.append("-" * 80)
for ticker, row in highest_alpha_10.head(5).iterrows():
    report_lines.append(
        f"{ticker}: Alpha={row['Alpha (%)']:.2f}%, "
        f"Beta={row['Beta']:.2f}, "
        f"Total Return={row['Total Return (%)']:.2f}%, "
        f"Sharpe={row['Sharpe Ratio']:.2f}"
    )
report_lines.append("")

report_lines.append("TOP 5 SECTORS BY COMPOSITE SCORE")
report_lines.append("-" * 80)
for sector, row in sector_summary.head(5).iterrows():
    report_lines.append(
        f"{sector}: Avg Total Return={row['Avg Total Return (%)']:.2f}%, "
        f"Avg Sharpe={row['Avg Sharpe Ratio']:.2f}, "
        f"Avg Alpha={row['Avg Alpha (%)']:.2f}%, "
        f"Stock Count={int(row['Stock Count'])}"
    )
report_lines.append("")

report_lines.append("MAX SHARPE PORTFOLIO")
report_lines.append("-" * 80)
ms = max_sharpe_portfolio.iloc[0]
report_lines.append(f"Portfolio Return (%): {ms['Portfolio Return (%)']:.2f}")
report_lines.append(f"Portfolio Volatility (%): {ms['Portfolio Volatility (%)']:.2f}")
report_lines.append(f"Portfolio Sharpe Ratio: {ms['Portfolio Sharpe Ratio']:.2f}")
report_lines.append("Top 10 Weights:")
weight_series = ms[ef_candidates].sort_values(ascending=False).head(10)
for ticker, weight in weight_series.items():
    report_lines.append(f"  {ticker}: {weight:.2%}")
report_lines.append("")

report_lines.append("MINIMUM VOLATILITY PORTFOLIO")
report_lines.append("-" * 80)
mv = min_vol_portfolio.iloc[0]
report_lines.append(f"Portfolio Return (%): {mv['Portfolio Return (%)']:.2f}")
report_lines.append(f"Portfolio Volatility (%): {mv['Portfolio Volatility (%)']:.2f}")
report_lines.append(f"Portfolio Sharpe Ratio: {mv['Portfolio Sharpe Ratio']:.2f}")
report_lines.append("Top 10 Weights:")
weight_series = mv[ef_candidates].sort_values(ascending=False).head(10)
for ticker, weight in weight_series.items():
    report_lines.append(f"  {ticker}: {weight:.2%}")
report_lines.append("")

report_path = OUTPUT_DIR / "summary_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    for line in report_lines:
        f.write(line + "\n")

# =========================
# 16. SHORT CONSOLE SUMMARY
# =========================
print("\nCoverage Summary:")
print(coverage_summary.to_string(index=False))

print("\nTop 10 Stocks by Composite Score:")
print(best_composite_10[[
    "Total Return (%)",
    "Volatility (%)",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)",
    "Alpha (%)",
    "Calmar Ratio",
    "Composite Score",
    "Sector"
]].to_string())

print("\nTop 10 Stocks by Sharpe Ratio:")
print(best_sharpe_10[[
    "Total Return (%)",
    "Volatility (%)",
    "Sharpe Ratio",
    "Alpha (%)",
    "Sector"
]].to_string())

print("\nSector Summary:")
print(sector_summary.to_string())

print("\nSPY Benchmark:")
print(benchmark_summary.to_string(index=False))

print("\nBenchmark Comparison:")
print(benchmark_comparison.to_string(index=False))

print(f"\nAll analysis completed successfully.")
print(f"Outputs saved in folder: {OUTPUT_DIR.resolve()}")