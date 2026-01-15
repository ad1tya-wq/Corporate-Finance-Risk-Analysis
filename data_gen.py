import mysql.connector
from faker import Faker
import random
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


load_dotenv()

# --- CONFIGURATION ---
db_config = {
    "host": os.getenv("DB_HOST"),      
    "user": os.getenv("DB_USER"),      
    "password": os.getenv("DB_PASSWORD"), 
    "database": "financial_risk_db"
}

fake = Faker()

# --- 1. SETUP DATABASE ---
def init_db():
    conn = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"]
    )
    cursor = conn.cursor()
    
    # Create DB if not exists
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
    cursor.execute(f"USE {db_config['database']}")
    
    # Drop old tables to start fresh
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS budgets")
    cursor.execute("DROP TABLE IF EXISTS departments")
    
    # Create Tables
    cursor.execute("""
    CREATE TABLE departments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        cost_center_code VARCHAR(10)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE budgets (
        dept_id INT,
        month VARCHAR(7), -- Format 'YYYY-MM'
        monthly_cap DECIMAL(10, 2),
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE transactions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE,
        amount DECIMAL(10, 2),
        category VARCHAR(50),
        description VARCHAR(255),
        dept_id INT,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    )
    """)
    
    print("Database and Tables Created Successfully.")
    return conn

# --- 2. GENERATE DATA ---
def generate_data(conn):
    cursor = conn.cursor()
    
    # A. Create Departments
    depts = ["Sales", "Engineering", "Marketing", "HR", "Operations"]
    dept_ids = []
    
    print("Seeding Departments...")
    for d in depts:
        cursor.execute("INSERT INTO departments (name, cost_center_code) VALUES (%s, %s)", (d, f"CC-{random.randint(100,999)}"))
        dept_ids.append(cursor.lastrowid)
    
    # B. Generate Transactions (Past 365 Days)
    print("Generating Transactions (This might take a moment)...")
    start_date = datetime.now() - timedelta(days=365)
    
    records = []
    
    for i in range(365):
        current_date = start_date + timedelta(days=i)
        month_str = current_date.strftime("%Y-%m")
        
        # Simulate "Normal" operations
        for dept_id in dept_ids:
            # Random daily expense
            daily_amount = random.uniform(500, 2000) 
            
            # === THE TWIST: MAKE COSTS EXPLODE IN THE LAST 60 DAYS ===
            # This ensures your ML model predicts a crash.
            days_from_now = (datetime.now() - current_date).days
            if days_from_now < 60 and dept_id == 1: # Sales Dept going rogue
                daily_amount *= 3.5 # Massive spike in spending
                desc = "Urgent Business Class Travel (Unapproved)"
                category = "Travel"
            else:
                desc = fake.bs()
                category = random.choice(["Software License", "Office Supplies", "Server Costs", "Payroll"])

            records.append((current_date, round(daily_amount, 2), category, desc, dept_id))

    # Bulk Insert
    sql = "INSERT INTO transactions (date, amount, category, description, dept_id) VALUES (%s, %s, %s, %s, %s)"
    cursor.executemany(sql, records)
    
    conn.commit()
    print(f"Inserted {len(records)} transactions.")
    conn.close()

# --- 3. GENERATE POLICY TEXT ---
def generate_policy_text():
    text = """
    CORPORATE FINANCIAL CONTROLS HANDBOOK (2025 EDITION)
    ====================================================

    SECTION 4: EMERGENCY COST CONTROL PROTOCOLS
    
    4.1. TRIGGER CONDITIONS
    These protocols are automatically triggered when:
    (a) Projected cash flow for the upcoming quarter is negative.
    (b) Any single department exceeds its quarterly budget by > 15%.

    4.2. TRAVEL RESTRICTIONS (The "Stop-Bleeding" Clause)
    In the event of a projected deficit:
    - All non-client-facing travel is immediately SUSPENDED.
    - All existing Business Class tickets must be downgraded to Economy.
    - Manager approval is required for any travel expense over $500.
    
    4.3. HIRING FREEZE
    If the projected runway drops below 6 months, all open roles are frozen immediately.
    """
    
    with open("Corporate_Policy_2025.txt", "w") as f:
        f.write(text)
    print("Policy Document 'Corporate_Policy_2025.txt' generated.")
    print("-> PLEASE OPEN THIS FILE AND 'PRINT TO PDF' TO USE IT WITH DOCLING.")

if __name__ == "__main__":
    conn = init_db()
    generate_data(conn)
    generate_policy_text()