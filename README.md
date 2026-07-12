# SMM921 Coursework — Trading and Market Microstructure

Group 19 coursework for SMM921 (Bayes Business School), submitted June 2026.

## What's in the report

The assignment has two parts:

**Part 1 — Liquidity analysis.** Using three months of one-minute intraday data, we pick three London-listed stocks (AstraZeneca, Next, Burberry) and build standard liquidity measures: midquotes, spreads, depth and midquote returns. We compare average liquidity across the stocks, look at how liquidity changes over the trading day, and use correlation and regression analysis to relate daily liquidity to daily volatility.

**Part 2 — Portfolio analysis.** Using 20 years of monthly country equity index data, we compare country performance (mean returns, volatility, Sharpe ratios, betas to a world portfolio), build five momentum-sorted portfolios plus a high-minus-low spread, and run a walk-forward mean-variance optimisation based on momentum alphas. We then repeat the optimisation with a constant-correlation covariance matrix and compare the two approaches.

## How it's built

The report is written in Quarto, which lets us keep the code, outputs, tables and references in a single document and render it straight to PDF.