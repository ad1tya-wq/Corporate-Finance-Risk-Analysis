import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = "financial_risk_db"

# Pooled engine shared by every caller in the process instead of a raw
# mysql.connector connection opened and closed per query.
engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    pool_size=5,
    pool_recycle=3600,
    pool_pre_ping=True,
)

_FORECAST_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS forecast_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_at DATETIME NOT NULL,
    max_txn_date DATE NOT NULL,
    trend VARCHAR(30) NOT NULL,
    current_burn DECIMAL(14, 2) NOT NULL,
    predicted_burn DECIMAL(14, 2) NOT NULL,
    plot_path VARCHAR(255) NOT NULL
)
"""


def ensure_forecast_log_table():
    with engine.begin() as conn:
        conn.execute(text(_FORECAST_LOG_TABLE_SQL))


def get_max_transaction_date():
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) AS max_date FROM transactions")).mappings().first()
        return row["max_date"] if row else None


def fetch_transactions_df() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT date, SUM(amount) as total_spend FROM transactions GROUP BY date ORDER BY date ASC"),
            conn,
        )


def get_fresh_cached_forecast(max_txn_date, ttl: timedelta):
    """Returns the most recent forecast_log row if it still covers max_txn_date
    and hasn't gone stale past ttl, else None (caller should refit)."""
    ensure_forecast_log_table()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM forecast_log ORDER BY id DESC LIMIT 1")).mappings().first()
    if not row:
        return None
    if row["max_txn_date"] != max_txn_date:
        return None
    if datetime.now() - row["run_at"] > ttl:
        return None
    return dict(row)


def log_forecast(max_txn_date, trend: str, current_burn: float, predicted_burn: float, plot_path: str):
    """Inserts a new forecast run. This table doubles as the durable audit
    trail — 'Reset System Memory' in the UI never touches it."""
    ensure_forecast_log_table()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO forecast_log (run_at, max_txn_date, trend, current_burn, predicted_burn, plot_path) "
                "VALUES (:run_at, :max_txn_date, :trend, :current_burn, :predicted_burn, :plot_path)"
            ),
            {
                "run_at": datetime.now(),
                "max_txn_date": max_txn_date,
                "trend": trend,
                "current_burn": current_burn,
                "predicted_burn": predicted_burn,
                "plot_path": plot_path,
            },
        )
