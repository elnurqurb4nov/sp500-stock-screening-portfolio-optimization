# S&P 500 Stock Screening & Portfolio Optimization System

This project implements a **quantitative stock screening and portfolio optimization system** for the S&P 500 using Python.

The system downloads historical market data, calculates multiple **risk-adjusted performance metrics**, ranks stocks using a **multi-factor scoring model**, performs **sector-level analysis**, and constructs optimized portfolios using **Modern Portfolio Theory and Efficient Frontier simulation**.

---

# Key Results

- **503 S&P 500 companies loaded**
- **498 stocks analyzed after data validation**
- **164 stocks outperformed SPY by total return**
- **73 stocks outperformed SPY by Sharpe ratio**
- **190 stocks generated positive Alpha**

### Best Performing Stocks (Composite Score)

- META
- GE
- FICO
- AVGO
- CRWD

### Best Performing Sector

Information Technology achieved the highest average composite performance score.

---

# Portfolio Optimization Results

### Maximum Sharpe Portfolio

- **Expected Return:** 77.28%
- **Volatility:** 17.71%
- **Sharpe Ratio:** 4.08

Top allocations included:

GE, VST, UBER, FICO, META, NVDA, PANW, CRM, AVGO

---

### Minimum Volatility Portfolio

- **Expected Return:** 67.83%
- **Volatility:** 16.07%
- **Sharpe Ratio:** 3.91

Top allocations included:

NRG, VST, GE, PANW, PHM, DECK, CRM, TDG, CPRT, DELL

---

# Features

The system performs the following analyses:

### Data Collection

- Downloads historical S&P 500 stock prices using **Yahoo Finance API**
- Uses the official **Wikipedia S&P 500 constituent list**

### Risk & Performance Metrics

For each stock the following metrics are calculated:

- Total Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Alpha
- Beta
- Treynor Ratio
- Information Ratio
- Maximum Drawdown
- Calmar Ratio

### Stock Screening Model

Stocks are ranked using a **multi-factor composite score** based on:

- Return
- Sharpe Ratio
- Sortino Ratio
- Drawdown
- Alpha
- Information Ratio
- Calmar Ratio

### Sector Analysis

- Average sector return
- Average Sharpe ratio
- Sector composite score ranking

### Portfolio Optimization

Monte Carlo simulation is used to generate **10,000 portfolios** and construct the **Efficient Frontier**.

The model identifies:

- Maximum Sharpe portfolio
- Minimum volatility portfolio

---

# Example Visualizations

### Efficient Frontier

![Efficient Frontier](plots/efficient_frontier.png)

### Correlation Heatmap

![Correlation Heatmap](plots/correlation_heatmap.png)

### Sector Average Total Return

![Sector Returns](plots/sector_avg_total_return.png)

### Top 10 Stocks by Sharpe Ratio

![Top Sharpe](plots/top_10_sharpe.png)

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Yahoo Finance API

---

# Installation

Clone the repository:
