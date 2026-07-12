#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMM921 Coursework - Part 1: Intraday Liquidity Analysis
=======================================================
Quantitative liquidity analysis of LSE intraday (1-minute) data.

Liquidity construction (mid, spread in bps, depth, midquote returns, the
intraday time-of-day grouping and the ADV/ADT normalisation) follows the
methodology in the provided lecture code:
    - smm921_liquidityStats.py   (R. Payne)  -> spread/depth/intraday means
    - smm921_priceImpact.py      (R. Payne)  -> trade-sign / price-impact idea
This module re-implements that logic for the 2026 data layout (which uses
different column names) WITHOUT modifying the original files or the raw CSV.

Author: Part-1 analysis (Claude, for Pietro). Originals (c) R. Payne.
"""

import numpy as np
import pandas as pd

DATA_FILE = "SMM921_trading_data_2026.csv"

# --- coursework parameters (from the brief) ---------------------------------
OPEN_MIN   = 8 * 60 + 15      # 08:15  -> keep observations at/after this
CLOSE_MIN  = 16 * 60 + 25     # 16:25  -> keep observations at/before this
RET_OUTLIER_BPS = 150.0       # |1-min midquote return| above this = data glitch
SEED = 921                    # reproducible random stock selection (fallback)
# Manager-specified stocks for the analysis (large / mid / small cap):
#   AZN.L  AstraZeneca (mega-cap pharma) | NXT.L Next (mid-cap retailer)
#   BRBY.L Burberry (small-cap luxury)
CHOSEN_STOCKS = ["AZN.L", "NXT.L", "BRBY.L"]


# ---------------------------------------------------------------------------
# 1. LOAD + FEATURE ENGINEERING + CLEANING
# ---------------------------------------------------------------------------
def load_and_engineer(path=DATA_FILE, verbose=True):
    """Load raw CSV and build the per-minute liquidity dataset.

    Returns (clean_df, diagnostics_dict).
    Columns added: mid, spread_bps, depth_gbp, ret_bps, gbp_vol, date, tod, hour.
    """
    raw = pd.read_csv(path)
    diag = {"raw_rows": len(raw), "n_stocks": raw["Stock"].nunique()}

    df = raw.copy()
    df["dt"]   = pd.to_datetime(df["Date and time"])
    df["date"] = df["dt"].dt.date
    df["tod"]  = df["dt"].dt.hour * 60 + df["dt"].dt.minute   # minute of day
    df["hour"] = df["dt"].dt.hour
    diag["n_days"] = df["date"].nunique()

    # "no trades this minute" -> empty Last/Volume/Num. Trades. Treat as zero
    # activity (the quote, i.e. Bid/Ask/sizes, is still valid for that minute).
    diag["no_trade_minutes"] = int(df["Last"].isna().sum())
    df["Volume"]      = df["Volume"].fillna(0.0)
    df["Num. Trades"] = df["Num. Trades"].fillna(0.0)

    # --- liquidity primitives (brief definitions) ---------------------------
    df["mid"]        = 0.5 * (df["Bid"] + df["Ask"])
    df["spread_bps"] = 1e4 * (df["Ask"] - df["Bid"]) / df["mid"]
    # Depth = average GBP value at best bid and ask. Prices are in pence (GBp),
    # so divide by 100 to express depth in pounds (GBP).
    df["depth_gbp"]  = 0.5 * (df["Bid size"] * df["Bid"]
                             + df["Ask size"] * df["Ask"]) / 100.0
    # GBP value traded in the minute (for ADV / Amihud); pence -> GBP.
    df["gbp_vol"]    = df["Volume"] * df["Last"].fillna(0.0) / 100.0

    # --- filters ------------------------------------------------------------
    # (a) trading window 08:15 - 16:25
    n0 = len(df)
    df = df[(df["tod"] >= OPEN_MIN) & (df["tod"] <= CLOSE_MIN)].copy()
    diag["dropped_time_window"] = n0 - len(df)

    # (b) non-positive (crossed/locked) spreads are non-economic -> drop
    n1 = len(df)
    diag["crossed_spreads"] = int((df["spread_bps"] < 0).sum())
    diag["locked_spreads"]  = int((df["spread_bps"] == 0).sum())
    df = df[df["spread_bps"] > 0].copy()
    diag["dropped_nonpos_spread"] = n1 - len(df)

    # --- midquote returns (within stock & day; no overnight jumps) ----------
    df = df.sort_values(["Stock", "dt"]).reset_index(drop=True)
    df["ret_bps"] = df.groupby(["Stock", "date"])["mid"].pct_change() * 1e4

    # (c) outlier returns: implausible 1-minute moves for FTSE blue chips.
    # These are vanishingly rare (e.g. 5 unrelated names all jump ~+200bps at
    # 2026-03-23 11:05 = a feed glitch) so we NULL the affected RETURNS only,
    # keeping the (sensible) quote rows for spread/depth.
    mask_out = df["ret_bps"].abs() > RET_OUTLIER_BPS
    diag["return_outliers_nulled"] = int(mask_out.sum())
    df.loc[mask_out, "ret_bps"] = np.nan

    diag["clean_rows"] = len(df)
    if verbose:
        for k, v in diag.items():
            print(f"  {k:>26}: {v}")
    return df, diag


# ---------------------------------------------------------------------------
# 2. STOCK SELECTION  (random, market-cap variety, not first/last three)
# ---------------------------------------------------------------------------
def select_stocks(df, override=None, seed=SEED, verbose=True):
    """Choose the 3 stocks for the analysis.

    By default returns the manager-specified `CHOSEN_STOCKS`
    (AZN/NXT/BRBY: a large-, mid- and small-cap that span the liquidity
    spectrum and are not simply the first/last three in the file). Pass
    override=None and a seed to instead draw one name at random per ADV
    tercile. Always returns the ADV series and tercile labels for context."""
    adv = (df.groupby("Stock")["gbp_vol"].sum() / df["date"].nunique())
    adv = adv.sort_values(ascending=False)
    # tercile labels by turnover (a market-cap proxy): large / mid / small
    tiers = pd.qcut(adv.rank(method="first", ascending=False), 3,
                    labels=["large", "mid", "small"])

    chosen = list(CHOSEN_STOCKS) if override is None else list(override)
    if chosen == "random":   # explicit random fallback
        order = df["Stock"].drop_duplicates().tolist()
        excluded = set(order[:3]) | set(order[-3:])
        rng = np.random.default_rng(seed)
        chosen = [rng.choice(sorted([s for s in adv.index[tiers == t]
                                     if s not in excluded]))
                  for t in ["large", "mid", "small"]]

    if verbose:
        print(f"  Selected stocks (large->small): {chosen}")
        print(f"  their ADV tiers: "
              f"{[ (s, str(tiers[s])) for s in chosen ]}")
    return chosen, adv, tiers


# ---------------------------------------------------------------------------
# 3. AVERAGE-LIQUIDITY SUMMARY
# ---------------------------------------------------------------------------
def summary_table(df, stocks):
    """Cross-sectional average-liquidity comparison for the chosen stocks."""
    ndays = df["date"].nunique()
    rows = {}
    for s in stocks:
        d = df[df["Stock"] == s]
        # Amihud illiquidity: |return| per GBP1m traded (trading minutes only).
        tr = d[d["gbp_vol"] > 0]
        amihud = (tr["ret_bps"].abs() / (tr["gbp_vol"] / 1e6)).mean()
        rows[s] = {
            "mean price (GBp)":       d["mid"].mean(),
            "mean spread (bps)":      d["spread_bps"].mean(),
            "median spread (bps)":    d["spread_bps"].median(),
            "mean quoted spread (GBp)": (d["Ask"] - d["Bid"]).mean(),
            "mean depth (GBP)":       d["depth_gbp"].mean(),
            "median depth (GBP)":     d["depth_gbp"].median(),
            "ADV (GBP m/day)":        d["gbp_vol"].sum() / ndays / 1e6,
            "avg trades/day":         d["Num. Trades"].sum() / ndays,
            "avg volume/day (sh)":    d["Volume"].sum() / ndays,
            "Amihud (bps per GBPm)":  amihud,
        }
    return pd.DataFrame(rows).T[[
        "mean price (GBp)", "mean spread (bps)", "median spread (bps)",
        "mean quoted spread (GBp)", "mean depth (GBP)", "median depth (GBP)",
        "ADV (GBP m/day)", "avg trades/day", "avg volume/day (sh)",
        "Amihud (bps per GBPm)"]]


# ---------------------------------------------------------------------------
# 4. INTRADAY (TIME-OF-DAY) PROFILE
# ---------------------------------------------------------------------------
def intraday_profile(df, stocks, freq=10):
    """Average liquidity by time-of-day bucket (minutes), averaged over days."""
    d = df[df["Stock"].isin(stocks)].copy()
    d["bucket"] = (d["tod"] // freq) * freq
    g = d.groupby(["Stock", "bucket"])
    prof = g.agg(spread_bps=("spread_bps", "mean"),
                 depth_gbp=("depth_gbp", "mean"),
                 volume=("Volume", "mean"),
                 trades=("Num. Trades", "mean")).reset_index()
    prof["time"] = (pd.to_timedelta(prof["bucket"], unit="m")
                    .astype(str).str.slice(7, 12))
    return prof


# ---------------------------------------------------------------------------
# 5. DAILY PANEL + LIQUIDITY-vs-VOLATILITY REGRESSION
# ---------------------------------------------------------------------------
def daily_panel(df, stocks):
    """Per stock-day: mean spread, mean depth, activity, and daily volatility
    (= mean absolute minutely midquote return within the day, per the brief)."""
    d = df[df["Stock"].isin(stocks)].copy()
    g = d.groupby(["Stock", "date"])
    daily = g.agg(
        spread_bps=("spread_bps", "mean"),
        depth_gbp=("depth_gbp", "mean"),
        gbp_vol=("gbp_vol", "sum"),
        trades=("Num. Trades", "sum"),
        vol_bps=("ret_bps", lambda x: x.abs().mean()),   # daily volatility
    ).reset_index()
    return daily


def liq_vol_regression(daily, stocks):
    """OLS of daily liquidity on daily volatility for each stock.
    Uses statsmodels if available (HC0 robust SEs), else closed-form OLS.
    Returns a tidy results DataFrame."""
    try:
        import statsmodels.api as sm
        have_sm = True
    except Exception:
        have_sm = False

    out = []
    for s in stocks:
        sub = daily[daily["Stock"] == s].dropna(subset=["vol_bps"])
        x = sub["vol_bps"].values
        for yname in ["spread_bps", "depth_gbp"]:
            y = sub[yname].values
            corr = np.corrcoef(x, y)[0, 1]
            if have_sm:
                X = sm.add_constant(x)
                res = sm.OLS(y, X).fit(cov_type="HC0")
                alpha, beta = res.params
                tstat = res.tvalues[1]
                r2 = res.rsquared
                pval = res.pvalues[1]
            else:
                import math
                beta = np.cov(x, y, bias=True)[0, 1] / np.var(x)
                alpha = y.mean() - beta * x.mean()
                r2 = corr ** 2
                yhat = alpha + beta * x
                se = np.sqrt(((y - yhat) ** 2).sum() / (len(x) - 2)
                             / ((x - x.mean()) ** 2).sum())
                tstat = beta / se
                # two-sided p-value, normal approximation (n=64 ~ large)
                pval = math.erfc(abs(tstat) / math.sqrt(2))
            out.append({"Stock": s, "liquidity": yname, "corr": corr,
                        "intercept": alpha, "slope": beta, "t(slope)": tstat,
                        "R2": r2, "p": pval, "n_days": len(sub)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=" * 70, "\n[1] LOAD + CLEAN")
    df, diag = load_and_engineer()

    print("\n[2] STOCK SELECTION")
    stocks, adv, tiers = select_stocks(df)

    print("\n[3] AVERAGE-LIQUIDITY SUMMARY")
    summ = summary_table(df, stocks)
    print(summ.round(2).to_string())

    print("\n[4] INTRADAY PROFILE (first/last buckets shown)")
    prof = intraday_profile(df, stocks)
    for s in stocks:
        p = prof[prof["Stock"] == s]
        print(f"  {s}: open spread={p.iloc[0]['spread_bps']:.2f}bps "
              f"midday(min)={p['spread_bps'].min():.2f}bps "
              f"close spread={p.iloc[-1]['spread_bps']:.2f}bps")

    print("\n[5] DAILY LIQUIDITY vs VOLATILITY")
    daily = daily_panel(df, stocks)
    reg = liq_vol_regression(daily, stocks)
    print(reg.round(4).to_string(index=False))

    return df, stocks, summ, prof, daily, reg


if __name__ == "__main__":
    main()
