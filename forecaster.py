"""
Forecasting core used by the dashboard (app.py) and usable on its own.

Model: SARIMAX(p,d,q) on the daily mode live-broiler price, with optional calendar regressors
(holiday, day-after-holiday, weekday dummies). Selected by walk-forward CV in the analysis notebook.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ConvergenceWarning)

PRICE_COL = "mode_price_lkr_per_kg"
STATUS_COL = "status"
VALID_STATUS = {"posted", "holiday", "no_post"}


# ----------------------------------------------------------------------------- data
def read_prices(src) -> pd.DataFrame:
    """Read a CSV with at least: date, mode_price_lkr_per_kg (blank = no price), status (optional)."""
    df = pd.read_csv(src)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column")
    if PRICE_COL not in df.columns:
        # accept a simpler 2-column file: date, price
        cand = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if not cand:
            raise ValueError(f"CSV must have a '{PRICE_COL}' (or a numeric price) column")
        df = df.rename(columns={cand[0]: PRICE_COL})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    if STATUS_COL not in df.columns:
        df[STATUS_COL] = np.where(df[PRICE_COL].notna(), "posted", "no_post")
    df[STATUS_COL] = df[STATUS_COL].fillna("").str.strip().str.lower()
    df.loc[df[PRICE_COL].notna(), STATUS_COL] = "posted"
    df.loc[~df[STATUS_COL].isin(VALID_STATUS), STATUS_COL] = "no_post"
    return df[["date", PRICE_COL, STATUS_COL]]


def merge_new_rows(df: pd.DataFrame, rows: list) -> pd.DataFrame:
    """rows: list of dicts {date, price or None, status}. New rows override existing dates."""
    if not rows:
        return df
    add = pd.DataFrame(rows).rename(columns={"price": PRICE_COL})
    add["date"] = pd.to_datetime(add["date"])
    out = pd.concat([df, add[["date", PRICE_COL, STATUS_COL]]], ignore_index=True)
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop_duplicates("date", keep="last").set_index("date").sort_index().asfreq("D")
    d[STATUS_COL] = d[STATUS_COL].fillna("no_post")
    d.loc[d[PRICE_COL].notna(), STATUS_COL] = "posted"
    return d


def prepare(df: pd.DataFrame, imputation="pchip"):
    d = to_daily(df)
    raw = d[PRICE_COL].astype(float)
    if raw.notna().sum() < 60:
        raise ValueError("Need at least 60 posted prices to fit the model")
    holiday = (d[STATUS_COL] == "holiday").astype(float)
    try:
        y = raw.interpolate(imputation)
    except Exception:
        y = raw.interpolate("linear")
    y = y.ffill().bfill()
    y.name = "price"
    return y, raw, holiday


def make_exog(index, holiday_flags) -> pd.DataFrame:
    hol = holiday_flags.reindex(index).fillna(0.0)
    X = pd.DataFrame({"holiday": hol.values, "after_holiday": hol.shift(1).fillna(0.0).values}, index=index)
    for d in range(1, 7):  # Monday is the baseline
        X[f"dow_{d}"] = (index.dayofweek == d).astype(float)
    return X


# ----------------------------------------------------------------------------- model
def fit(y, X, order):
    return SARIMAX(y, exog=X, order=tuple(order),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)


def retune(y, X, max_pq=3):
    best = None
    for p, q in itertools.product(range(max_pq + 1), range(max_pq + 1)):
        try:
            aic = fit(y, X, (p, 1, q)).aic
            if best is None or aic < best[0]:
                best = (aic, (p, 1, q))
        except Exception:
            pass
    return best[1], best[0]


def run_forecast(df, days=7, order=(2, 1, 2), use_exog=True, future_holidays=(), n_sims=2000,
                 imputation="pchip", seed=42):
    """Returns dict with history, daily forecast table, weekly table, model info."""
    y, raw, hol = prepare(df, imputation)
    X = make_exog(y.index, hol) if use_exog else None
    res = fit(y, X, order)

    horizon = max(int(days), 21)   # always simulate 3 weeks so weekly averages are complete
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fut_hol = pd.Series(0.0, index=idx)
    if len(future_holidays):
        fut_hol[idx.isin(pd.to_datetime(list(future_holidays)))] = 1.0
    Xf = make_exog(idx, pd.concat([hol, fut_hol])) if use_exog else None

    f = res.get_forecast(horizon, exog=Xf)
    ci80, ci95 = f.conf_int(alpha=.20), f.conf_int(alpha=.05)
    daily = pd.DataFrame({
        "date": idx, "weekday": idx.day_name(),
        "forecast": f.predicted_mean.values,
        "lo80": ci80.iloc[:, 0].values, "hi80": ci80.iloc[:, 1].values,
        "lo95": ci95.iloc[:, 0].values, "hi95": ci95.iloc[:, 1].values,
        "holiday": fut_hol.astype(int).values,
    })
    num = ["forecast", "lo80", "hi80", "lo95", "hi95"]
    daily.loc[daily["holiday"] == 1, num] = np.nan
    daily[num] = daily[num].round(1)

    # simulated paths -> intervals for weekly averages
    np.random.seed(seed)
    sims = np.asarray(res.simulate(horizon, repetitions=n_sims, anchor="end", exog=Xf)).reshape(horizon, -1)
    sims = pd.DataFrame(sims, index=idx)
    week_start = idx.to_period("W-SUN").start_time
    wk_rows = []
    for wk in pd.unique(week_start):
        sel = (week_start == wk) & (fut_hol.values == 0)
        if sel.sum() == 0:
            continue
        m = sims[sel].mean(axis=0)
        wk_rows.append({"week_start": pd.Timestamp(wk).date(),
                        "week_end": (pd.Timestamp(wk) + pd.Timedelta(days=6)).date(),
                        "forecast_days": int(sel.sum()), "avg_forecast": m.mean(),
                        "lo80": m.quantile(.10), "hi80": m.quantile(.90),
                        "lo95": m.quantile(.025), "hi95": m.quantile(.975)})
    weekly = pd.DataFrame(wk_rows).round(1)

    return {
        "y": y, "raw": raw, "holiday": hol,
        "daily": daily.head(int(days)).reset_index(drop=True),
        "weekly": weekly,
        "aic": res.aic, "order": tuple(order), "use_exog": use_exog,
        "last_date": y.index[-1], "last_posted_date": raw.dropna().index[-1],
        "last_posted_price": float(raw.dropna().iloc[-1]),
    }


def quick_backtest(df, order=(2, 1, 2), use_exog=True, n_origins=20, step=5, horizon=7, imputation="pchip"):
    """Walk-forward check on recent data vs naive. Scored on real posted days only."""
    y, raw, hol = prepare(df, imputation)
    X = make_exog(y.index, hol) if use_exog else None
    origins = y.index[-(n_origins * step + horizon):-horizon:step]
    rows = []
    for o in origins:
        tr = y.loc[:o]
        fut = pd.date_range(o + pd.Timedelta(days=1), periods=horizon, freq="D")
        try:
            r = fit(tr, X.loc[tr.index] if use_exog else None, order)
            p = r.forecast(horizon, exog=X.loc[fut] if use_exog else None).values
        except Exception:
            continue
        for k, d in enumerate(fut):
            if pd.notna(raw.get(d)):
                rows.append({"origin": o, "h": k + 1, "actual": raw[d], "model": p[k], "naive": tr.iloc[-1]})
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, pd.DataFrame()
    summ = []
    for h in sorted(bt["h"].unique()):
        s = bt[bt["h"] <= h]
        mae_m, mae_n = (s["actual"] - s["model"]).abs().mean(), (s["actual"] - s["naive"]).abs().mean()
        summ.append({"horizon_upto_days": h, "n": len(s),
                     "model_MAE": mae_m, "naive_MAE": mae_n,
                     "model_MAPE%": ((s["actual"] - s["model"]).abs() / s["actual"]).mean() * 100,
                     "accuracy% (100-MAPE)": 100 - ((s["actual"] - s["model"]).abs() / s["actual"]).mean() * 100,
                     "skill_vs_naive%": (1 - mae_m / mae_n) * 100})
    return bt, pd.DataFrame(summ).round(2)
