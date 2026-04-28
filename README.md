# S&P 500 Stock Screening & Portfolio Optimization System

## Overview

This project builds a Python-based quantitative stock screening and portfolio optimization system for the S&P 500 universe.

The system collects historical market data, calculates risk-adjusted performance metrics, ranks stocks through a multi-factor scoring model, compares sector-level performance, and simulates optimized portfolios using Modern Portfolio Theory.

The purpose of this project is not to produce a trading signal, but to demonstrate how Python can be used to structure a systematic equity analysis workflow from raw price data to portfolio-level interpretation.

---

## Research Question

**Can historical risk-adjusted performance metrics help identify stronger stocks and construct more efficient portfolios within the S&P 500 universe?**

The project explores this question through three main layers:

- Stock-level screening
- Sector-level performance comparison
- Portfolio-level optimization

---

## Key Results

The current version of the system produced the following results:

- **503 S&P 500 companies loaded**
- **498 stocks analyzed after data validation**
- **164 stocks outperformed SPY by total return**
- **73 stocks outperformed SPY by Sharpe ratio**
- **190 stocks generated positive Alpha**

### Best Performing Stocks by Composite Score

```text
META, GE, FICO, AVGO, CRWD
