# AI Portfolio Analytics Lab

AI-powered portfolio analytics platform built with Python and Streamlit. This application allows users to upload portfolio datasets or manually select assets to generate advanced investment analytics, optimization models, risk metrics, and AI-generated portfolio insights.

---

## Features

### Portfolio Allocation Analysis
- Upload portfolio CSV files
- Manually select assets such as:
  - AAPL
  - SPY
  - NVDA
  - MSFT
  - AMZN
  - META
  - TSLA
- Visual portfolio allocation breakdowns

### Performance Metrics
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Portfolio growth visualization

### Risk Analytics
- Correlation matrix
- Asset-level volatility analysis
- Diversification analysis

### Efficient Frontier Optimization
- Monte Carlo portfolio simulations
- Maximum Sharpe ratio optimization
- Portfolio optimization using SciPy
- Efficient frontier visualization

### Backtesting Engine
- Compare portfolio performance against:
  - SPY
  - QQQ
  - VOO
- Historical portfolio benchmarking

### Monte Carlo Simulation
- Thousands of forward portfolio simulations
- Future value distribution modeling
- Risk scenario forecasting

### AI Market Regime Detection
- Bullish / Bearish regime classification
- Rolling volatility analysis
- Rolling return signals

### AI Investment Insights
Automatically generated investment commentary including:
- Portfolio concentration analysis
- Diversification insights
- Volatility interpretation
- Risk observations
- Market regime summaries

### Executive Summary Export
- Downloadable portfolio analytics report

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- yFinance
- Machine Learning & Quantitative Finance Concepts

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Portfolio-Lab.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python3 -m streamlit run portfolio_lab_app.py
```

---

# Example CSV Format

```csv
Ticker,Weight
AAPL,0.30
MSFT,0.25
NVDA,0.20
SPY,0.25
```

---

# Project Purpose

This project was designed to simulate a professional portfolio analytics workflow used in:
- Asset management
- Wealth management
- Quantitative finance
- Investment research
- Financial analytics

The platform demonstrates how machine learning, optimization, and financial modeling can be combined into an interactive AI-driven analytics tool.

---

# Future Improvements

Planned future enhancements include:
- Live market data streaming
- LSTM price forecasting models
- Sentiment analysis integration
- Factor model analysis
- Black-Litterman optimization
- VaR and CVaR modeling
- PDF report exports
- User authentication
- Cloud database integration
- AI chatbot investment assistant

---

# Disclaimer

This application is for educational and research purposes only and should not be considered financial advice. Historical performance does not guarantee future results.

---

# Author

Justin Grumm

GitHub Repository:
https://github.com/Grummreaper/AI-Portfolio-Lab
