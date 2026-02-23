"""
Seed expense policies and sample expense data.
Run once: python seed_expense_data.py
"""

from app.database import SessionLocal
from app.models import ExpensePolicy, Expense, ExpenseStatus
from datetime import date, datetime

def seed():
    db = SessionLocal()
    try:
        # --- Expense Policies ---
        policies = [
            {"category": "meals",           "max_amount": 50.0,  "approval_threshold": 25.0,  "requires_receipt": True,  "description": "Meals and food expenses. Auto-approve up to $25, escalate to $50, reject above."},
            {"category": "travel",          "max_amount": 200.0, "approval_threshold": 100.0, "requires_receipt": True,  "description": "Travel expenses (flights, hotels, transport). Auto-approve up to $100."},
            {"category": "equipment",       "max_amount": 500.0, "approval_threshold": 200.0, "requires_receipt": True,  "description": "Office equipment and hardware. Auto-approve up to $200."},
            {"category": "entertainment",   "max_amount": 100.0, "approval_threshold": 50.0,  "requires_receipt": True,  "description": "Client entertainment and team events. Auto-approve up to $50."},
            {"category": "office_supplies", "max_amount": 100.0, "approval_threshold": 50.0,  "requires_receipt": False, "description": "Stationery, printing, and general office supplies. Auto-approve up to $50."},
            {"category": "other",           "max_amount": 150.0, "approval_threshold": 50.0,  "requires_receipt": True,  "description": "Miscellaneous expenses. Auto-approve up to $50."},
        ]

        seeded = 0
        for p in policies:
            exists = db.query(ExpensePolicy).filter(ExpensePolicy.category == p["category"]).first()
            if not exists:
                db.add(ExpensePolicy(**p))
                seeded += 1
                print(f"[OK] Seeded policy: {p['category']}")
            else:
                print(f"[SKIP] Policy already exists: {p['category']}")

        db.commit()
        print(f"\nPolicies: {seeded} seeded, {len(policies) - seeded} already existed")

        # --- Sample Approved Expenses for John_Employee (user_id=2 typically) ---
        # Find employee user - try user_id 2 first, skip if not found
        from app.models import Employee
        employee = db.query(Employee).first()
        if not employee:
            print("[SKIP] No employees found - skipping sample expenses")
            return

        user_id = employee.user_id
        print(f"\nSeeding sample expenses for user_id={user_id}...")

        sample_expenses = [
            {
                "user_id": user_id, "amount": 18.50, "category": "meals",
                "vendor": "Pret A Manger", "date": date(2026, 2, 10),
                "description": "Team lunch", "status": ExpenseStatus.APPROVED.value,
                "llm_decision": "AUTO_APPROVE",
                "llm_reasoning": "Amount $18.50 is within the $25 auto-approve threshold for meals. No duplicates detected. OCR confidence 0.91.",
                "submitted_at": datetime(2026, 2, 10, 12, 0), "reviewed_at": datetime(2026, 2, 10, 12, 0),
            },
            {
                "user_id": user_id, "amount": 85.00, "category": "travel",
                "vendor": "National Rail", "date": date(2026, 2, 14),
                "description": "Train to client site", "status": ExpenseStatus.APPROVED.value,
                "llm_decision": "AUTO_APPROVE",
                "llm_reasoning": "Amount $85.00 is within the $100 auto-approve threshold for travel. Receipt OCR confidence 0.88.",
                "submitted_at": datetime(2026, 2, 14, 9, 0), "reviewed_at": datetime(2026, 2, 14, 9, 0),
            },
            {
                "user_id": user_id, "amount": 45.00, "category": "office_supplies",
                "vendor": "Staples", "date": date(2026, 2, 17),
                "description": "Notebook and pens", "status": ExpenseStatus.APPROVED.value,
                "llm_decision": "AUTO_APPROVE",
                "llm_reasoning": "Amount $45.00 is within the $50 auto-approve threshold for office_supplies. No receipt required.",
                "submitted_at": datetime(2026, 2, 17, 10, 0), "reviewed_at": datetime(2026, 2, 17, 10, 0),
            },
            {
                "user_id": user_id, "amount": 150.00, "category": "travel",
                "vendor": "Premier Inn", "date": date(2026, 2, 20),
                "description": "Hotel for conference", "status": ExpenseStatus.SUBMITTED.value,
                "llm_decision": "ESCALATE",
                "llm_reasoning": "Amount $150.00 exceeds the $100 auto-approve threshold for travel. Requires manager approval.",
                "submitted_at": datetime(2026, 2, 20, 8, 0),
            },
        ]

        exp_seeded = 0
        for e in sample_expenses:
            db.add(Expense(**e))
            exp_seeded += 1
        db.commit()
        print(f"[OK] Seeded {exp_seeded} sample expenses (3 approved, 1 pending)")

    finally:
        db.close()

if __name__ == "__main__":
    seed()
    print("\n[DONE] Expense seed complete.")
