# Sri Lanka Live Broiler Price Forecast — Dashboard

A Streamlit web app that forecasts the daily live broiler price (Rs/kg) using the SARIMAX model chosen in
`broiler_price_time_series.ipynb`.

## What it does
* **Inputs:** add new daily prices (one at a time or paste many), mark holidays, upload your own CSV,
  choose how many days to forecast, and (advanced) change the model order or switch the calendar effects off.
* **Outputs:** daily forecast with 80% / 95% ranges, weekly-average forecast (Mon–Sun) from simulated price paths,
  accuracy figures from the notebook, an on-demand backtest on the current data, and CSV downloads.
* **The model is refit on all data every time** you change something, so the forecast always starts from the
  latest price (fits in about 2–3 seconds and is cached).

## Files
```
app.py               Streamlit dashboard
forecaster.py        Forecasting core (also usable from Python on its own)
model_config.json    Model order, settings and validation results
data/broiler_prices.csv   Price history (date, mode_price_lkr_per_kg, status)
requirements.txt     Python packages
.streamlit/config.toml    Theme
```

## Run on your computer
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Deploy for free — Streamlit Community Cloud (recommended)
1. Create a **public** GitHub repository and upload every file and folder in this zip (keep the folder structure).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**, pick your repo and branch, and set *Main file path* to `app.py`.
4. (Optional) Under **Advanced settings** choose Python 3.11 or 3.12.
5. Click **Deploy**. After a few minutes you get a link like `https://your-app-name.streamlit.app`.

Updating the data: replace `data/broiler_prices.csv` in GitHub (the app's *Data* tab has a download button that
includes days you added). The app redeploys automatically.

## Alternative — Render (free web service)
1. Push the same files to GitHub.
2. On **render.com**, create a **New → Web Service**, connect the repo and choose the **Free** instance type.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Notes on free hosting
* Free hosts **don't keep files between restarts**, so days added in the sidebar exist only in your browser session. To make
  them permanent, download the updated CSV from the *Data* tab and commit it to GitHub.
* Free apps **go to sleep** when nobody uses them. The first visit after that takes about a minute to wake up.
* Holidays: add known Poya and public holiday dates in the sidebar, or permanently in `model_config.json → future_holidays`.

## Use the forecaster without the dashboard
```python
import forecaster as fc
df = fc.read_prices("data/broiler_prices.csv")
out = fc.run_forecast(df, days=7)
print(out["daily"]); print(out["weekly"])
```
