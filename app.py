"""
Sri Lanka Live Broiler Price — Forecast Dashboard
Run locally:  streamlit run app.py
"""
import io
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecaster as fc

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "broiler_prices.csv"
CONFIG = json.loads((ROOT / "model_config.json").read_text())

st.set_page_config(page_title="Broiler Price Forecast", page_icon="🐔", layout="wide")

# ----------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("new_rows", [])
ss.setdefault("holidays", list(CONFIG.get("future_holidays", [])))
ss.setdefault("backtest", None)


@st.cache_data(show_spinner=False)
def load_bundled():
    return fc.read_prices(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_uploaded(content: bytes):
    return fc.read_prices(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def cached_forecast(df, days, order, use_exog, holidays, n_sims, imputation):
    return fc.run_forecast(df, days=days, order=order, use_exog=use_exog,
                           future_holidays=holidays, n_sims=n_sims, imputation=imputation)


@st.cache_data(show_spinner=False)
def cached_retune(df, use_exog, imputation):
    y, raw, hol = fc.prepare(df, imputation)
    X = fc.make_exog(y.index, hol) if use_exog else None
    return fc.retune(y, X)


# ----------------------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Price history", ["Built-in data", "Upload my CSV"], label_visibility="collapsed")
    if source == "Upload my CSV":
        up = st.file_uploader("CSV with columns: date, mode_price_lkr_per_kg, status", type="csv")
        st.caption("A simple 2-column file (date, price) also works.")
        if up is None:
            st.info("Upload a file, or switch back to built-in data.")
            st.stop()
        try:
            base = load_uploaded(up.getvalue())
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        base = load_bundled()

    last_known = base["date"].max().date()

    st.header("2 · Add new prices")
    with st.form("add_price", clear_on_submit=True):
        c1, c2 = st.columns(2)
        d_new = c1.date_input("Date", value=last_known + timedelta(days=1))
        status = c2.selectbox("Status", ["posted", "holiday"])
        p_new = st.number_input("Mode price (Rs/kg)", min_value=0.0, max_value=3000.0, value=0.0, step=5.0,
                                help="The most common price on the day's board. Leave 0 for a holiday.")
        if st.form_submit_button("➕ Add day", use_container_width=True):
            if status == "posted" and p_new <= 0:
                st.warning("Enter a price above 0 for a posted day.")
            else:
                ss.new_rows = [r for r in ss.new_rows if r["date"] != str(d_new)]
                ss.new_rows.append({"date": str(d_new), "price": p_new if status == "posted" else None, "status": status})

    with st.expander("Paste several days at once"):
        bulk = st.text_area("One per line: YYYY-MM-DD, price  (use 'holiday' instead of a price)",
                            placeholder="2026-10-30, 545\n2026-10-31, 540\n2026-11-01, holiday", height=110)
        if st.button("Add pasted days", use_container_width=True) and bulk.strip():
            added, bad = 0, []
            for line in bulk.strip().splitlines():
                parts = [x.strip() for x in line.replace("\t", ",").split(",")]
                try:
                    dt_ = pd.to_datetime(parts[0]).date()
                    if parts[1].lower() == "holiday":
                        row = {"date": str(dt_), "price": None, "status": "holiday"}
                    else:
                        row = {"date": str(dt_), "price": float(parts[1]), "status": "posted"}
                    ss.new_rows = [r for r in ss.new_rows if r["date"] != str(dt_)] + [row]
                    added += 1
                except Exception:
                    bad.append(line)
            st.success(f"Added {added} day(s)")
            if bad:
                st.warning("Skipped: " + "; ".join(bad))

    if ss.new_rows:
        st.dataframe(pd.DataFrame(ss.new_rows).sort_values("date"), hide_index=True, use_container_width=True)
        if st.button("🗑 Clear added days", use_container_width=True):
            ss.new_rows = []
            st.rerun()

    st.header("3 · Upcoming holidays")
    st.caption("Poya / public holidays when no price is posted. Improves the forecast around those days.")
    hc1, hc2 = st.columns([3, 1])
    h_new = hc1.date_input("Holiday date", value=None, label_visibility="collapsed")
    if hc2.button("Add", use_container_width=True) and h_new:
        if str(h_new) not in ss.holidays:
            ss.holidays.append(str(h_new))
    if ss.holidays:
        keep = st.multiselect("Holidays used", sorted(ss.holidays), default=sorted(ss.holidays))
        ss.holidays = keep

    st.header("4 · Forecast settings")
    days = st.slider("Days to forecast", 1, 30, 7)
    if days > 14:
        st.warning("Beyond ~14 days the model is barely better than 'price stays the same'.")
    lookback = st.slider("History shown on chart (days)", 30, 700, 120, step=10)

    with st.expander("Advanced model settings"):
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("p", 0, 5, CONFIG["order"][0])
        d = c2.number_input("d", 0, 2, CONFIG["order"][1])
        q = c3.number_input("q", 0, 5, CONFIG["order"][2])
        use_exog = st.checkbox("Holiday & weekday effects (SARIMAX)", value=CONFIG["use_calendar_exog"])
        n_sims = st.select_slider("Simulations for weekly ranges", [500, 1000, 2000, 5000], value=2000)
        imputation = st.selectbox("Fill missing days with", ["pchip", "linear", "time", "akima", "ffill"],
                                  index=["pchip", "linear", "time", "akima", "ffill"].index(CONFIG["imputation"]))
        if st.button("Find best (p,1,q) by AIC", use_container_width=True):
            with st.spinner("Searching 16 models..."):
                best, aic = cached_retune(fc.merge_new_rows(base, ss.new_rows), use_exog, imputation)
            st.success(f"Best order: {best}  (AIC {aic:.1f}). Set p, d, q above to use it.")

# ----------------------------------------------------------------------------- run model
df = fc.merge_new_rows(base, ss.new_rows)
order = (int(p), int(d), int(q))
try:
    with st.spinner("Fitting model and forecasting..."):
        out = cached_forecast(df, days, order, use_exog, tuple(sorted(ss.holidays)), n_sims, imputation)
except Exception as e:
    st.error(f"Model failed: {e}")
    st.stop()

daily, weekly = out["daily"], out["weekly"]
y, raw = out["y"], out["raw"]
model_name = f"SARIMAX{order}" + (" + calendar" if use_exog else "")

# ----------------------------------------------------------------------------- header + KPIs
st.title("🐔 Sri Lanka Live Broiler Price Forecast")
st.caption(f"Daily mode price across companies (Rs/kg) · Model: **{model_name}** · "
           f"Data to {out['last_date'].date()} · Source: Alpha Broiler Farm Association price boards")

gap = (out["last_date"] - out["last_posted_date"]).days
if gap > 0:
    st.warning(f"No posted price for the last {gap} day(s). The forecast starts from an estimated value — "
               "add the latest board in the sidebar for a better forecast.")

first = daily.dropna(subset=["forecast"]).iloc[0] if daily["forecast"].notna().any() else None
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last posted price", f"Rs {out['last_posted_price']:.0f}", help=str(out["last_posted_date"].date()))
if first is not None:
    k2.metric(f"Forecast {pd.Timestamp(first['date']).strftime('%a %d %b')}", f"Rs {first['forecast']:.0f}",
              delta=f"{first['forecast'] - out['last_posted_price']:+.0f}", delta_color="off")
k3.metric(f"Average next {days} day(s)", f"Rs {daily['forecast'].mean():.0f}",
          help=f"80% range for the last day: Rs {daily['lo80'].dropna().iloc[-1]:.0f}–{daily['hi80'].dropna().iloc[-1]:.0f}"
          if daily["forecast"].notna().any() else None)
full_weeks = weekly[weekly["forecast_days"] >= 5]
if len(full_weeks):
    w = full_weeks.iloc[0]
    k4.metric(f"Week of {pd.Timestamp(w['week_start']).strftime('%d %b')} (avg)", f"Rs {w['avg_forecast']:.0f}",
              help=f"80% range Rs {w['lo80']:.0f}–{w['hi80']:.0f}")

tab_f, tab_w, tab_a, tab_d, tab_i = st.tabs(["📈 Daily forecast", "📅 Weekly forecast", "🎯 Accuracy", "🗂 Data", "ℹ️ How it works"])

# ----------------------------------------------------------------------------- daily
with tab_f:
    hist = y.iloc[-lookback:]
    posted = raw.dropna().iloc[-lookback:]
    posted = posted[posted.index >= hist.index[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name="History (gaps filled)",
                             line=dict(color="#B0B7C3", width=1.5)))
    fig.add_trace(go.Scatter(x=posted.index, y=posted, mode="markers", name="Posted price",
                             marker=dict(color="#1f4e79", size=5)))
    dd = daily.dropna(subset=["forecast"])
    x_band = list(dd["date"]) + list(dd["date"][::-1])
    fig.add_trace(go.Scatter(x=x_band, y=list(dd["hi95"]) + list(dd["lo95"][::-1]), fill="toself", mode="lines",
                             fillcolor="rgba(214,69,65,.12)", line=dict(width=0), name="95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_band, y=list(dd["hi80"]) + list(dd["lo80"][::-1]), fill="toself", mode="lines",
                             fillcolor="rgba(214,69,65,.25)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    anchor_x = [out["last_date"]] + list(dd["date"])
    anchor_y = [float(y.iloc[-1])] + list(dd["forecast"])
    fig.add_trace(go.Scatter(x=anchor_x, y=anchor_y, mode="lines+markers", name="Forecast",
                             line=dict(color="#d64541", width=2.5), marker=dict(size=6)))
    for hd in daily.loc[daily["holiday"] == 1, "date"]:
        fig.add_vline(x=hd, line_dash="dot", line_color="gray")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="Rs / kg", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    show = daily.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    show["holiday"] = show["holiday"].map({1: "holiday", 0: ""})
    show = show.rename(columns={"forecast": "Forecast (Rs/kg)", "lo80": "80% low", "hi80": "80% high",
                                "lo95": "95% low", "hi95": "95% high"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button("⬇ Download daily forecast (CSV)", daily.to_csv(index=False).encode(),
                       f"daily_forecast_{out['last_date'].date()}.csv", "text/csv")
    st.caption("Read the 80% range as: on 8 days out of 10 the real price should fall inside it.")

# ----------------------------------------------------------------------------- weekly
with tab_w:
    st.subheader("Weekly average price (Monday–Sunday)")
    st.caption("Computed from 2,000+ simulated price paths, so the ranges are correct for a weekly average. "
               "Weeks only partly inside the forecast window are averaged over the forecast days only.")
    n_weeks = st.slider("Weeks to show", 1, 6, 3)
    need_days = n_weeks * 7 + 7
    if need_days > days:
        wk_out = cached_forecast(df, need_days, order, use_exog, tuple(sorted(ss.holidays)), n_sims, imputation)["weekly"]
    else:
        wk_out = weekly
    wk = wk_out.head(n_weeks + 1)
    figw = go.Figure()
    labels = [f"{pd.Timestamp(a).strftime('%d %b')}–{pd.Timestamp(b).strftime('%d %b')}" for a, b in zip(wk.week_start, wk.week_end)]
    figw.add_trace(go.Bar(x=labels, y=wk["avg_forecast"], name="Avg forecast", marker_color="#d64541",
                          error_y=dict(type="data", symmetric=False, array=wk["hi80"] - wk["avg_forecast"],
                                       arrayminus=wk["avg_forecast"] - wk["lo80"]),
                          text=wk["avg_forecast"].round(0), textposition="inside", insidetextanchor="start",
                          textfont=dict(color="white", size=14)))
    figw.update_layout(height=380, yaxis_title="Rs / kg (error bars = 80% range)",
                       yaxis_range=[max(0, wk["lo80"].min() * .9), wk["hi80"].max() * 1.05],
                       margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figw, use_container_width=True)
    st.dataframe(wk.rename(columns={"avg_forecast": "Avg forecast (Rs/kg)", "lo80": "80% low", "hi80": "80% high",
                                    "lo95": "95% low", "hi95": "95% high"}), hide_index=True, use_container_width=True)
    st.download_button("⬇ Download weekly forecast (CSV)", wk.to_csv(index=False).encode(),
                       f"weekly_forecast_{out['last_date'].date()}.csv", "text/csv")

# ----------------------------------------------------------------------------- accuracy
with tab_a:
    st.subheader("How accurate is the model?")
    val = pd.DataFrame(CONFIG["validation"]["walk_forward"])
    st.markdown(f"**Validated in the analysis notebook** — walk-forward test, {CONFIG['validation']['n_origins']} "
                f"forecast start dates, scored only on days with a real posted price:")
    st.dataframe(val, hide_index=True, use_container_width=True)
    st.info("Under normal conditions the forecast is within about **±Rs 11 next day** and **±Rs 26 within a week**. "
            "It cannot foresee sudden supply-driven price shocks (e.g. the Aug–Oct 2026 fall from ~780 to 550), "
            "and 'price stays the same' is already a strong benchmark — the model's edge over it is ~7–9%.")

    st.markdown("**Extra checks (walk-forward, same test period)**")
    st.dataframe(pd.DataFrame(CONFIG["validation"]["extra_checks"]), hide_index=True, use_container_width=True)
    st.caption("Takeaway: the point forecast is only modestly better than 'no change' (clearly so at 7 days, not at 1 day). "
               "The main value of the model is the price ranges — they were reliable (slightly cautious) in testing.")

    st.markdown("**Re-check accuracy on the current data** (including any days you added):")
    if st.button("▶ Run quick backtest (≈20–40 s)"):
        with st.spinner("Refitting the model at 20 recent start dates..."):
            ss.backtest = fc.quick_backtest(df, order=order, use_exog=use_exog, imputation=imputation)
    if ss.backtest is not None:
        bt, summ = ss.backtest
        if summ.empty:
            st.warning("Not enough recent posted prices for a backtest.")
        else:
            st.dataframe(summ, hide_index=True, use_container_width=True)
            figb = go.Figure()
            figb.add_trace(go.Scatter(x=summ["horizon_upto_days"], y=summ["model_MAE"], name="Model", mode="lines+markers"))
            figb.add_trace(go.Scatter(x=summ["horizon_upto_days"], y=summ["naive_MAE"], name="Naive (no change)", mode="lines+markers"))
            figb.update_layout(height=320, xaxis_title="Days ahead (cumulative)", yaxis_title="MAE (Rs/kg)",
                               margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figb, use_container_width=True)

# ----------------------------------------------------------------------------- data
with tab_d:
    d_all = fc.to_daily(df).reset_index()
    c1, c2, c3 = st.columns(3)
    c1.metric("Days covered", len(d_all))
    c2.metric("Posted prices", int((d_all["status"] == "posted").sum()))
    c3.metric("Days you added this session", len(ss.new_rows))
    st.dataframe(d_all.sort_values("date", ascending=False).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")),
                 hide_index=True, use_container_width=True, height=380)
    st.download_button("⬇ Download full data incl. added days (CSV)", d_all.to_csv(index=False).encode(),
                       "broiler_prices_updated.csv", "text/csv")
    st.caption("Free hosting doesn't keep files between restarts. To make added days permanent, download this file "
               "and replace data/broiler_prices.csv in your GitHub repo.")

# ----------------------------------------------------------------------------- about
with tab_i:
    st.markdown(f"""
**Data** — daily *mode* live broiler price (the most common price across companies) read from the
"Live Price" boards of the Alpha Broiler Farm Association of Sri Lanka. Days without a board are filled with
`{imputation}` interpolation for fitting only; accuracy is always measured on real posted days.

**Model** — {model_name}: an ARIMA model on daily prices (differenced once, because the price level wanders like a
random walk) plus holiday / day-after-holiday / weekday effects. It was chosen because it had the lowest error in a
walk-forward test against naive, seasonal naive, ARIMA, SARIMA, exponential smoothing and gradient boosting.

**Every forecast refits the model on all data up to the latest day**, so adding today's price immediately updates it.

**Good use:** next 1–7 days, and next 1–2 weekly averages. **Not reliable:** a month or more ahead.

**Inputs you control:** new daily prices, upcoming holidays, forecast length, and (advanced) the model order.
""")
