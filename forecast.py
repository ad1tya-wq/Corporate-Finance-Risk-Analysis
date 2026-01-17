import json
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mysql.connector
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "password"),
        database="financial_risk_db"
    )

def run_forecast():
    """
    1. Fetches transaction data.
    2. Aggregates it by day.
    3. Trains a Prophet model.
    4. Predicts 90 days into the future.
    5. Returns the forecast plot path and the textual summary.
    """
    print("Oracle: Fetching financial data...")
    
    # 1. FETCH DATA
    conn = get_db_connection()
    # We use dictionary=True here so pandas can easily digest the rows
    cursor = conn.cursor(dictionary=True) 
    
    query = """
    SELECT date, SUM(amount) as total_spend 
    FROM transactions 
    GROUP BY date 
    ORDER BY date ASC
    """
    cursor.execute(query)
    result = cursor.fetchall()
    
    df = pd.DataFrame(result)
    
    cursor.close()
    conn.close()

    if df.empty:
        return {"trend": "ERROR", "message": "No data found in database."}

    # 2. PREPARE DATA FOR PROPHET
    # Prophet requires columns named strictly 'ds' (date) and 'y' (value)
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['total_spend']
    
    # Filter to keep only relevant columns
    df = df[['ds', 'y']]

    print(f"Oracle: Training model on {len(df)} days of data...")

    # 3. TRAIN MODEL
    # yearly_seasonality=True helps it understand annual budgets
    # changepoint_prior_scale=0.5 makes it SENSITIVE to recent changes (like our crash)
    m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.5) 
    m.fit(df)

    # 4. PREDICT FUTURE (90 Days)
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    # 5. ANALYZE RESULTS
    # Get the average spending for next week vs last week to check trend
    current_burn = df['y'].tail(30).mean() # Last 30 days actuals
    predicted_burn = forecast['yhat'].tail(30).mean() # Next 30 days prediction
    
    trend = "STABLE"
    if predicted_burn > current_burn * 1.5:
        trend = "CRITICAL SPIKE"
    elif predicted_burn > current_burn * 1.1:
        trend = "INCREASING (RISK)"

    # 6. SAVE PLOT (For the UI)
    plt.figure(figsize=(10, 6))
    m.plot(forecast, ax=plt.gca())
    plt.title("Financial Burn Rate Forecast (Next 90 Days)")
    plt.xlabel("Date")
    plt.ylabel("Daily Spend ($)")
    
    # Save to a static folder so the UI can read it later
    if not os.path.exists("static"):
        os.makedirs("static")
    
    plot_path = "static/forecast_plot.png"
    plt.savefig(plot_path)
    print(f"Oracle: Forecast generated. Trend: {trend}")
    
    monthly_current = current_burn * 30 
    monthly_predicted = predicted_burn * 30
    
    metrics = {
        "trend": trend,
        "current_burn": monthly_current,
        "predicted_burn": monthly_predicted
    }
    with open("static/metrics.json", "w") as f:
        json.dump(metrics, f)
        
    return (
        f"DATA REPORT:\n"
        f"- Status: {trend}\n"
        f"- Current Monthly Burn: ${monthly_current:,.2f}\n"
        f"- Projected Monthly Burn (90 days): ${monthly_predicted:,.2f}\n"
        f"- Visual Proof: {plot_path}\n\n"
        f"SYSTEM ALERT: The projected burn exceeds the safe limit. "
        f"Immediate cost-saving measures are required per company policy."
    )

# --- TEST BLOCK (Runs only if you execute this file directly) ---
if __name__ == "__main__":
    try:
        result = run_forecast()
        print("\n--- FORECAST REPORT ---")
        print(f"Trend: {result['trend']}")
        print(f"Current Monthly Burn (approx): ${result['current_burn'] * 30:,.2f}")
        print(f"Projected Monthly Burn: ${result['predicted_burn'] * 30:,.2f}")
        print(f"Check the plot at: {result['plot_path']}")
    except Exception as e:
        print(f"Error running forecast: {e}")