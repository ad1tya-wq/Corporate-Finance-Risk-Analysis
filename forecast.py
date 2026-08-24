import os
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

from db import fetch_transactions_df, get_fresh_cached_forecast, get_max_transaction_date, log_forecast

CACHE_TTL = timedelta(hours=6)
PLOT_PATH = os.path.join("static", "forecast_plot.png")


def _read_plot_bytes():
    if os.path.exists(PLOT_PATH):
        with open(PLOT_PATH, "rb") as f:
            return f.read()
    return None


def run_forecast(force_refresh: bool = False) -> dict:
    """
    Fetches transactions, trains Prophet, predicts 90 days out, and classifies
    the trend. Refits only when the latest transaction date has moved past
    what's cached in forecast_log, or the cache is older than CACHE_TTL —
    a chat turn that doesn't need a fresh number reuses the last run instead
    of pulling MySQL and refitting Prophet every time.
    """
    max_txn_date = get_max_transaction_date()
    if max_txn_date is None:
        return {
            "trend": "ERROR",
            "current_burn": 0.0,
            "predicted_burn": 0.0,
            "plot_path": None,
            "plot_bytes": None,
            "from_cache": False,
            "message": "No transaction data found.",
        }

    if not force_refresh:
        cached = get_fresh_cached_forecast(max_txn_date, CACHE_TTL)
        if cached:
            return {
                "trend": cached["trend"],
                "current_burn": float(cached["current_burn"]),
                "predicted_burn": float(cached["predicted_burn"]),
                "plot_path": cached["plot_path"],
                "plot_bytes": _read_plot_bytes(),
                "from_cache": True,
            }

    print("Oracle: Fetching financial data...")
    df = fetch_transactions_df()
    if df.empty:
        return {
            "trend": "ERROR",
            "current_burn": 0.0,
            "predicted_burn": 0.0,
            "plot_path": None,
            "plot_bytes": None,
            "from_cache": False,
            "message": "No data found in database.",
        }

    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = df["total_spend"]
    df = df[["ds", "y"]]

    print(f"Oracle: Training model on {len(df)} days of data...")
    # changepoint_prior_scale=0.5 makes it SENSITIVE to recent changes (like a crash)
    m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.5)
    m.fit(df)

    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    current_burn = df["y"].tail(30).mean()
    predicted_burn = forecast["yhat"].tail(30).mean()

    trend = "STABLE"
    if predicted_burn > current_burn * 1.5:
        trend = "CRITICAL SPIKE"
    elif predicted_burn > current_burn * 1.1:
        trend = "INCREASING (RISK)"

    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(10, 6))
    m.plot(forecast, ax=plt.gca())
    plt.title("Financial Burn Rate Forecast (Next 90 Days)")
    plt.xlabel("Date")
    plt.ylabel("Daily Spend ($)")
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Oracle: Forecast generated. Trend: {trend}")

    monthly_current = current_burn * 30
    monthly_predicted = predicted_burn * 30

    log_forecast(max_txn_date, trend, monthly_current, monthly_predicted, PLOT_PATH)

    return {
        "trend": trend,
        "current_burn": monthly_current,
        "predicted_burn": monthly_predicted,
        "plot_path": PLOT_PATH,
        "plot_bytes": _read_plot_bytes(),
        "from_cache": False,
    }


def format_forecast_report(result: dict) -> str:
    if result["trend"] == "ERROR":
        return f"DATA REPORT: {result.get('message', 'Forecast unavailable.')}"

    lines = [
        "DATA REPORT:",
        f"- Status: {result['trend']}",
        f"- Current Monthly Burn: ${result['current_burn']:,.2f}",
        f"- Projected Monthly Burn (90 days): ${result['predicted_burn']:,.2f}",
    ]
    if result["trend"] != "STABLE":
        lines.append("")
        lines.append(
            "SYSTEM ALERT: The projected burn exceeds the safe limit. "
            "Immediate cost-saving measures are required per company policy."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        result = run_forecast()
        print("\n--- FORECAST REPORT ---")
        print(format_forecast_report(result))
    except Exception as e:
        print(f"Error running forecast: {e}")
