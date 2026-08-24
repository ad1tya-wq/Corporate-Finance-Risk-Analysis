import os
import random
from datetime import datetime, timedelta

import mysql.connector
from dotenv import load_dotenv
from faker import Faker
from fpdf import FPDF
from fpdf.enums import XPos, YPos

load_dotenv()

# --- CONFIGURATION ---
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "financial_risk_db",
}

DOCS_DIR = os.path.join("data", "docs")

fake = Faker()


# --- 1. SETUP DATABASE ---
def init_db():
    conn = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
    cursor.execute(f"USE {db_config['database']}")

    # Drop old tables to start fresh. forecast_log is intentionally NOT
    # dropped here even on a reseed -- it's the durable audit trail and
    # should survive a "start over with new fake data" reset. If it doesn't
    # exist yet, CREATE TABLE IF NOT EXISTS below makes it.
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS budgets")
    cursor.execute("DROP TABLE IF EXISTS departments")

    cursor.execute(
        """
        CREATE TABLE departments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            cost_center_code VARCHAR(10)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE budgets (
            dept_id INT,
            month VARCHAR(7), -- Format 'YYYY-MM'
            monthly_cap DECIMAL(10, 2),
            FOREIGN KEY (dept_id) REFERENCES departments(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            amount DECIMAL(10, 2),
            category VARCHAR(50),
            description VARCHAR(255),
            dept_id INT,
            FOREIGN KEY (dept_id) REFERENCES departments(id)
        )
        """
    )

    cursor.execute(
        """
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
    )

    print("Database and Tables Created Successfully.")
    return conn


# --- 2. GENERATE DATA ---
def generate_data(conn):
    cursor = conn.cursor()

    depts = ["Sales", "Engineering", "Marketing", "HR", "Operations"]
    dept_ids = []

    print("Seeding Departments...")
    for d in depts:
        cursor.execute(
            "INSERT INTO departments (name, cost_center_code) VALUES (%s, %s)",
            (d, f"CC-{random.randint(100,999)}"),
        )
        dept_ids.append(cursor.lastrowid)

    print("Generating Transactions (This might take a moment)...")
    start_date = datetime.now() - timedelta(days=365)

    records = []

    for i in range(365):
        current_date = start_date + timedelta(days=i)

        for dept_id in dept_ids:
            daily_amount = random.uniform(500, 2000)

            # === THE TWIST: MAKE COSTS EXPLODE IN THE LAST 60 DAYS ===
            # Ensures the forecast model predicts a crash.
            days_from_now = (datetime.now() - current_date).days
            if days_from_now < 60 and dept_id == 1:  # Sales Dept going rogue
                daily_amount *= 3.5
                desc = "Urgent Business Class Travel (Unapproved)"
                category = "Travel"
            else:
                desc = fake.bs()
                category = random.choice(["Software License", "Office Supplies", "Server Costs", "Payroll"])

            records.append((current_date, round(daily_amount, 2), category, desc, dept_id))

    sql = "INSERT INTO transactions (date, amount, category, description, dept_id) VALUES (%s, %s, %s, %s, %s)"
    cursor.executemany(sql, records)

    conn.commit()
    print(f"Inserted {len(records)} transactions.")
    conn.close()


# --- 3. GENERATE POLICY DOCUMENTS ---
# Several distinct synthetic policy documents (not just one), so the RAG
# retrieval + rerank steps actually have something to discriminate between.
def generate_policy_documents() -> dict:
    return {
        "emergency_cost_control_protocols": """
CORPORATE FINANCIAL CONTROLS HANDBOOK (2025 EDITION)
====================================================

SECTION 4: EMERGENCY COST CONTROL PROTOCOLS

4.1. TRIGGER CONDITIONS
These protocols are automatically triggered when:
(a) Projected cash flow for the upcoming quarter is negative.
(b) Any single department exceeds its quarterly budget by more than 15%.

4.2. TRAVEL RESTRICTIONS (The "Stop-Bleeding" Clause)
In the event of a projected deficit:
- All non-client-facing travel is immediately SUSPENDED.
- All existing Business Class tickets must be downgraded to Economy.
- Manager approval is required for any travel expense over $500.

4.3. HIRING FREEZE
If the projected runway drops below 6 months, all open roles are frozen
immediately, and no new requisitions may be opened without CFO sign-off.
""",
        "travel_expense_policy": """
TRAVEL AND EXPENSE POLICY
==========================

SECTION 1: BOOKING CLASS
Employees below Director level must book Economy class for flights under 6
hours. Business Class requires VP approval regardless of flight duration.

SECTION 2: DAILY MEAL ALLOWANCE
Domestic travel: $75/day. International travel: $110/day. Alcohol is never
a reimbursable expense under any circumstance.

SECTION 3: APPROVAL THRESHOLDS
Any single travel expense over $500 requires manager approval before
booking, not after. Expenses submitted more than 30 days after travel will
not be reimbursed.

SECTION 4: EMERGENCY SUSPENSION
During an active cost-control trigger (see Emergency Cost Control
Protocols, Section 4), all discretionary and non-client-facing travel is
suspended company-wide until the CFO lifts the restriction in writing.
""",
        "procurement_policy": """
PROCUREMENT AND VENDOR ONBOARDING POLICY
==========================================

SECTION 1: COMPETITIVE BIDDING
Any purchase over $10,000 requires at least three competing quotes before
a purchase order is issued. Sole-source purchases require VP Finance
sign-off.

SECTION 2: SOFTWARE LICENSE APPROVAL
New SaaS subscriptions over $2,000 annually must be reviewed by IT Security
before purchase. Auto-renewing contracts must be flagged 60 days before
renewal.

SECTION 3: VENDOR PAYMENT TERMS
Standard payment terms are Net 30. Net 15 or shorter requires Finance
Director approval and is reserved for vendors offering an early-payment
discount of at least 2%.

SECTION 4: EMERGENCY SPENDING FREEZE
When an emergency cost-control trigger is active, all new purchase orders
over $1,000 are held for CFO review, and non-essential software renewals
are deferred until the freeze is lifted.
""",
        "hiring_headcount_policy": """
HIRING AND HEADCOUNT POLICY
=============================

SECTION 1: REQUISITION APPROVAL
Every open role requires a signed requisition from the hiring manager and
the department VP before it can be posted externally.

SECTION 2: CONTRACTOR CONVERSIONS
Contractors may not be converted to full-time employees before completing
a minimum 90-day engagement, except with CHRO exception approval.

SECTION 3: HIRING FREEZE CONDITIONS
A company-wide hiring freeze is triggered automatically when projected
cash runway drops below 6 months. During a freeze:
- All open, unfilled requisitions are frozen immediately.
- Backfills for critical client-facing roles require CFO exception approval.
- Internal transfers are still permitted; external hiring is not.

SECTION 4: SEVERANCE
Standard severance is two weeks of base pay per year of tenure, capped at
26 weeks, administered per local employment law.
""",
        "vendor_payment_terms": """
VENDOR PAYMENT AND ACCOUNTS PAYABLE POLICY
=============================================

SECTION 1: PAYMENT RUN SCHEDULE
Accounts Payable runs payment batches every Tuesday and Friday. Invoices
received after 5pm Monday roll to the Friday run.

SECTION 2: LATE PAYMENT ESCALATION
Invoices unpaid 10 days past terms are escalated to the AP Manager.
Invoices unpaid 30 days past terms are escalated to the CFO directly.

SECTION 3: EMERGENCY PAYMENT HOLDS
During an active emergency cost-control trigger, all vendor payments over
$5,000 that are not tied to a signed contract with a penalty clause are
held for manual CFO review before release.

SECTION 4: DISPUTED INVOICES
Disputed invoices are held from the payment run until Procurement confirms
resolution in writing; the 30-day payment clock pauses during a dispute.
""",
    }


def _write_pdf(text: str, path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.strip("\n").split("\n"):
        if line.strip():
            pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln(6)
    pdf.output(path)


def generate_policy_pdfs():
    """Writes every synthetic policy document straight to PDF (no manual
    'open the .txt and print to PDF' step) so the whole synthetic corpus can
    be regenerated end-to-end by running this script."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    documents = generate_policy_documents()
    for slug, text in documents.items():
        pdf_path = os.path.join(DOCS_DIR, f"{slug}.pdf")
        _write_pdf(text, pdf_path)
        print(f"Wrote {pdf_path}")
    print(
        f"\nGenerated {len(documents)} policy PDF(s) in {DOCS_DIR}. "
        "Next: run policy_process.py to convert them to Markdown, then rag/ingest.py to index them."
    )


if __name__ == "__main__":
    conn = init_db()
    generate_data(conn)
    generate_policy_pdfs()
