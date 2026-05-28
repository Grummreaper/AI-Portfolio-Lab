import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="AI Portfolio Analytics Lab", layout="wide")

st.title("AI Portfolio Analytics Lab")
st.write("Upload holdings or select tickers to analyze portfolio performance, risk, optimization, backtesting, Monte Carlo simulations, and market regime signals.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Portfolio Inputs")

default_tickers = ["AAPL", "SPY", "NVDA", "MSFT"]
manual_tickers = st.sidebar.multiselect(
    "Select tickers",
    ["AAPL", "SPY", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "QQQ", "VOO", "IWM"],
    default=default_tickers
)

uploaded_file = st.sidebar.file_uploader("Or upload portfolio CSV", type=["csv"])

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

risk_free_rate = st.sidebar.number_input("Risk-free rate", value=0.04, step=0.01)

# -----------------------------
# Load Portfolio
# CSV should have: Ticker, Weight
# -----------------------------
if uploaded_file is not None:
    portfolio_df = pd.read_csv(uploaded_file)
    tickers = portfolio_df["Ticker"].str.upper().tolist()
    weights = portfolio_df["Weight"].astype(float).values
    weights = weights / weights.sum()
else:
    tickers = manual_tickers
    if len(tickers) > 0:
        weights = np.array([1 / len(tickers)] * len(tickers))
    else:
        st.warning("Select at least one ticker.")
        st.stop()

if len(tickers) < 2:
    st.warning("Please select at least two assets.")
    st.stop()

# -----------------------------
# Download Data
# -----------------------------
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()

prices = load_data(tickers, start_date, end_date)
returns = prices.pct_change().dropna()

# -----------------------------
# Portfolio Calculations
# -----------------------------
portfolio_returns = returns.dot(weights)
cumulative_returns = (1 + portfolio_returns).cumprod()

annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

rolling_max = cumulative_returns.cummax()
drawdown = cumulative_returns / rolling_max - 1
max_drawdown = drawdown.min()

# -----------------------------
# 1. Allocation Breakdown
# -----------------------------
st.header("1. Portfolio Allocation Breakdown")

allocation_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight": weights
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(allocation_df)

with col2:
    fig_alloc, ax_alloc = plt.subplots()
    ax_alloc.pie(weights, labels=tickers, autopct="%1.1f%%")
    ax_alloc.set_title("Portfolio Allocation")
    st.pyplot(fig_alloc)

# -----------------------------
# 2. Performance Metrics
# -----------------------------
st.header("2. Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annual Return", f"{annual_return:.2%}")
col2.metric("Annual Volatility", f"{annual_volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
ax_perf.plot(cumulative_returns.index, cumulative_returns, label="Portfolio")
ax_perf.set_title("Portfolio Growth of $1")
ax_perf.set_ylabel("Growth")
ax_perf.legend()
st.pyplot(fig_perf)

# -----------------------------
# 3. Risk Analytics
# -----------------------------
st.header("3. Risk Analytics")

st.subheader("Correlation Matrix")
corr = returns.corr()
st.dataframe(corr)

fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
im = ax_corr.imshow(corr)
ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45)
ax_corr.set_yticklabels(corr.columns)
ax_corr.set_title("Asset Correlation Matrix")
fig_corr.colorbar(im)
st.pyplot(fig_corr)

asset_risk = pd.DataFrame({
    "Ticker": returns.columns,
    "Annual Return": returns.mean() * 252,
    "Annual Volatility": returns.std() * np.sqrt(252)
}).sort_values("Annual Volatility", ascending=False)

st.subheader("Asset-Level Risk")
st.dataframe(asset_risk)

# -----------------------------
# 4. Efficient Frontier
# -----------------------------
st.header("4. Efficient Frontier")

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(tickers)

def portfolio_performance(w):
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def negative_sharpe(w):
    return -portfolio_performance(w)[2]

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = np.array([1 / num_assets] * num_assets)

optimal = minimize(
    negative_sharpe,
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

optimal_weights = optimal.x
opt_return, opt_vol, opt_sharpe = portfolio_performance(optimal_weights)

st.subheader("Max Sharpe Portfolio")
opt_df = pd.DataFrame({
    "Ticker": tickers,
    "Optimized Weight": optimal_weights
}).sort_values("Optimized Weight", ascending=False)

st.dataframe(opt_df)

col1, col2, col3 = st.columns(3)
col1.metric("Optimized Return", f"{opt_return:.2%}")
col2.metric("Optimized Volatility", f"{opt_vol:.2%}")
col3.metric("Optimized Sharpe", f"{opt_sharpe:.2f}")

# Simulate random portfolios
sim_returns = []
sim_vols = []
sim_sharpes = []

for _ in range(3000):
    w = np.random.random(num_assets)
    w = w / np.sum(w)
    r, v, s = portfolio_performance(w)
    sim_returns.append(r)
    sim_vols.append(v)
    sim_sharpes.append(s)

fig_frontier, ax_frontier = plt.subplots(figsize=(10, 6))
scatter = ax_frontier.scatter(sim_vols, sim_returns, c=sim_sharpes)
ax_frontier.scatter(opt_vol, opt_return, marker="*", s=300, label="Max Sharpe Portfolio")
ax_frontier.set_xlabel("Volatility")
ax_frontier.set_ylabel("Return")
ax_frontier.set_title("Efficient Frontier Simulation")
ax_frontier.legend()
fig_frontier.colorbar(scatter, label="Sharpe Ratio")
st.pyplot(fig_frontier)

# -----------------------------
# 5. Backtesting Engine
# -----------------------------
st.header("5. Backtesting Engine")

benchmark_ticker = st.selectbox("Benchmark", ["SPY", "QQQ", "VOO"], index=0)
benchmark_prices = load_data([benchmark_ticker], start_date, end_date)
benchmark_returns = benchmark_prices.pct_change().dropna()

benchmark_growth = (1 + benchmark_returns.iloc[:, 0]).cumprod()

fig_backtest, ax_backtest = plt.subplots(figsize=(10, 5))
ax_backtest.plot(cumulative_returns.index, cumulative_returns, label="Portfolio")
ax_backtest.plot(benchmark_growth.index, benchmark_growth, label=benchmark_ticker)
ax_backtest.set_title("Portfolio vs Benchmark Backtest")
ax_backtest.legend()
st.pyplot(fig_backtest)

# -----------------------------
# 6. Monte Carlo Simulation
# -----------------------------
st.header("6. Monte Carlo Simulation")

num_simulations = st.slider("Number of simulations", 100, 2000, 500)
num_days = st.slider("Forecast days", 30, 756, 252)

last_value = 1
daily_mean = portfolio_returns.mean()
daily_std = portfolio_returns.std()

simulation_results = []

for _ in range(num_simulations):
    simulated_returns = np.random.normal(daily_mean, daily_std, num_days)
    price_path = last_value * np.cumprod(1 + simulated_returns)
    simulation_results.append(price_path)

simulation_results = np.array(simulation_results)

fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
for i in range(min(num_simulations, 100)):
    ax_mc.plot(simulation_results[i], alpha=0.2)
ax_mc.set_title("Monte Carlo Portfolio Simulation")
ax_mc.set_xlabel("Days")
ax_mc.set_ylabel("Portfolio Value")
st.pyplot(fig_mc)

ending_values = simulation_results[:, -1]

col1, col2, col3 = st.columns(3)
col1.metric("Median Ending Value", f"{np.median(ending_values):.2f}")
col2.metric("5th Percentile", f"{np.percentile(ending_values, 5):.2f}")
col3.metric("95th Percentile", f"{np.percentile(ending_values, 95):.2f}")

# -----------------------------
# 7. AI Market Regime Detection
# -----------------------------
st.header("7. AI Market Regime Detection")

rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252)
rolling_return = portfolio_returns.rolling(21).mean() * 252

latest_vol = rolling_vol.dropna().iloc[-1]
latest_return = rolling_return.dropna().iloc[-1]

if latest_return > 0 and latest_vol < annual_volatility:
    regime = "Bullish / Stable"
elif latest_return > 0 and latest_vol >= annual_volatility:
    regime = "Bullish but Volatile"
elif latest_return < 0 and latest_vol >= annual_volatility:
    regime = "Bearish / High Volatility"
else:
    regime = "Defensive / Low Momentum"

st.metric("Detected Market Regime", regime)

fig_regime, ax_regime = plt.subplots(figsize=(10, 5))
ax_regime.plot(rolling_vol.index, rolling_vol, label="Rolling Volatility")
ax_regime.plot(rolling_return.index, rolling_return, label="Rolling Return")
ax_regime.set_title("Rolling Return and Volatility Regime Signals")
ax_regime.legend()
st.pyplot(fig_regime)

# -----------------------------
# 8. AI Investment Insights
# -----------------------------
st.header("8. AI Investment Insights")

top_weight = allocation_df.sort_values("Weight", ascending=False).iloc[0]
highest_vol = asset_risk.iloc[0]
avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()

insights = []

if top_weight["Weight"] > 0.4:
    insights.append(
        f"The portfolio is highly concentrated in {top_weight['Ticker']}, which represents {top_weight['Weight']:.1%} of total allocation."
    )
else:
    insights.append("The portfolio does not appear overly concentrated in a single asset.")

if annual_volatility > 0.25:
    insights.append("Portfolio volatility is relatively high, suggesting meaningful exposure to market swings.")
else:
    insights.append("Portfolio volatility is moderate based on the selected time period.")

if avg_corr > 0.6:
    insights.append("Average asset correlation is high, meaning diversification may be limited.")
else:
    insights.append("Average asset correlation is moderate/low, suggesting some diversification benefit.")

insights.append(
    f"The highest-volatility asset is {highest_vol['Ticker']}, with annualized volatility of {highest_vol['Annual Volatility']:.2%}."
)

insights.append(
    f"The current regime is classified as: {regime}."
)

for insight in insights:
    st.write("- " + insight)

# -----------------------------
# 9. Downloadable Report
# -----------------------------
st.header("9. Download Executive Summary")

report = f"""
AI Portfolio Analytics Lab - Executive Summary

Portfolio:
{allocation_df.to_string(index=False)}

Performance:
Annual Return: {annual_return:.2%}
Annual Volatility: {annual_volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.2%}

Optimized Portfolio:
{opt_df.to_string(index=False)}

Market Regime:
{regime}

AI Investment Insights:
{chr(10).join(["- " + i for i in insights])}

Limitations:
This app is a portfolio analytics prototype and should not be treated as investment advice. Results depend on historical data, assumptions about returns, and simplified optimization methods. Future returns may differ materially from historical performance.
"""

st.download_button(
    label="Download Executive Summary",
    data=report,
    file_name="portfolio_analytics_summary.txt",
    mime="text/plain"
)