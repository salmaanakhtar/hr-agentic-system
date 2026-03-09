"""
Seed payroll data: add salary/tax to existing employees, create one completed
past pay cycle (February 2026) with approved payslips.

Run once: python seed_payroll_data.py
Idempotent: skips records that already exist or are already set.
"""

from datetime import date, datetime

from app.database import SessionLocal
from app.models import (
    Employee,
    PayCycle,
    PayCycleStatus,
    Payslip,
    PayslipStatus,
    User,
)


# ---------------------------------------------------------------------------
# Salary configuration — applied to employees that have no base_salary set.
# Keyed by first_name for demo predictability; anything else gets DEFAULT.
# ---------------------------------------------------------------------------
_SALARY_BY_NAME = {
    "John":  {"base_salary": 48_000.0, "tax_rate": 0.20, "pay_frequency": "monthly", "department": "Engineering"},
    "Sarah": {"base_salary": 55_000.0, "tax_rate": 0.22, "pay_frequency": "monthly", "department": "Product"},
    "Mark":  {"base_salary": 42_000.0, "tax_rate": 0.20, "pay_frequency": "monthly", "department": "Operations"},
}
_DEFAULT_SALARY = {"base_salary": 40_000.0, "tax_rate": 0.20, "pay_frequency": "monthly"}

# Pay period: February 2026 (20 working days Mon–Fri)
_PERIOD_START = date(2026, 2, 1)
_PERIOD_END   = date(2026, 2, 28)
_WORKING_DAYS = 20


def _monthly_gross(base_salary: float) -> float:
    return round(base_salary / 12, 2)


def _calc_payslip(base_salary: float, tax_rate: float, leave_days_taken: float = 0.0, unpaid_leave_days: float = 0.0) -> dict:
    """
    Calculate payslip figures.
    gross            = base_salary / 12  (paid leave does NOT reduce gross)
    deductions_leave = unpaid_leave_days * daily_rate  (days exceeding balance)
    deductions_tax   = (gross - deductions_leave) * tax_rate
    net_pay          = gross - deductions_leave - deductions_tax
    leave_days_taken = total approved leave in period (informational)
    """
    gross = _monthly_gross(base_salary)
    daily_rate = gross / _WORKING_DAYS
    deductions_leave = round(unpaid_leave_days * daily_rate, 2)
    taxable = round(gross - deductions_leave, 2)
    deductions_tax = round(taxable * tax_rate, 2)
    net_pay = round(gross - deductions_leave - deductions_tax, 2)
    return {
        "gross_pay":        gross,
        "deductions_leave": deductions_leave,
        "deductions_tax":   deductions_tax,
        "net_pay":          net_pay,
        "days_worked":      float(_WORKING_DAYS - leave_days_taken),
        "leave_days_taken": leave_days_taken,
    }


def seed():
    db = SessionLocal()
    try:
        # -----------------------------------------------------------------------
        # 1. Update employee salary/tax data
        # -----------------------------------------------------------------------
        employees = db.query(Employee).all()
        if not employees:
            print("[ERROR] No employees found — sign up at least one employee first")
            return

        print(f"Found {len(employees)} employee(s). Setting salary data...\n")
        for emp in employees:
            if emp.base_salary:
                print(f"[SKIP] {emp.first_name} {emp.last_name} already has salary: £{emp.base_salary:,.0f}")
                continue

            cfg = _SALARY_BY_NAME.get(emp.first_name, _DEFAULT_SALARY)
            emp.base_salary    = cfg["base_salary"]
            emp.tax_rate       = cfg["tax_rate"]
            emp.pay_frequency  = cfg.get("pay_frequency", "monthly")
            if cfg.get("department") and not emp.department:
                emp.department = cfg["department"]

            print(
                f"[OK]   {emp.first_name} {emp.last_name}: "
                f"salary=£{emp.base_salary:,.0f}, tax={emp.tax_rate:.0%}"
            )

        db.commit()
        for emp in employees:
            db.refresh(emp)

        # -----------------------------------------------------------------------
        # 2. Create the completed pay cycle (Feb 2026) if it doesn't exist
        # -----------------------------------------------------------------------
        existing_cycle = db.query(PayCycle).filter(
            PayCycle.period_start == _PERIOD_START,
            PayCycle.period_end   == _PERIOD_END,
        ).first()

        if existing_cycle:
            print(f"\n[SKIP] Pay cycle Feb 2026 already exists (id={existing_cycle.id})")
            cycle = existing_cycle
        else:
            # Use first available user (HR or admin) as run_by
            runner = db.query(User).first()
            cycle = PayCycle(
                period_start = _PERIOD_START,
                period_end   = _PERIOD_END,
                status       = PayCycleStatus.COMPLETED.value,
                run_by       = runner.id if runner else None,
                run_at       = datetime(2026, 3, 1, 9, 0, 0),
            )
            db.add(cycle)
            db.flush()
            db.refresh(cycle)
            db.commit()
            print(f"\n[OK]   Created pay cycle Feb 2026 (id={cycle.id})")

        # -----------------------------------------------------------------------
        # 3. Create approved payslips for each employee
        # -----------------------------------------------------------------------
        print("\nSeeding payslips...\n")
        payslips_created = 0

        # John took 2 approved leave days in Feb (within balance — no unpaid deduction).
        # Others had clean months.
        leave_taken_overrides  = {"John": 2.0}   # total leave days taken (informational)
        unpaid_leave_overrides = {}               # leave days exceeding balance (all zero for seed)

        for emp in employees:
            existing_slip = db.query(Payslip).filter(
                Payslip.employee_id  == emp.id,
                Payslip.pay_cycle_id == cycle.id,
            ).first()
            if existing_slip:
                print(f"[SKIP] Payslip already exists for {emp.first_name} {emp.last_name} (id={existing_slip.id})")
                continue

            leave_days_taken  = leave_taken_overrides.get(emp.first_name, 0.0)
            unpaid_leave_days = unpaid_leave_overrides.get(emp.first_name, 0.0)
            figures = _calc_payslip(emp.base_salary, emp.tax_rate, leave_days_taken, unpaid_leave_days)

            reasoning = (
                f"{emp.first_name} {emp.last_name} — February 2026 payroll. "
                f"Annual salary £{emp.base_salary:,.0f}, monthly gross £{figures['gross_pay']:,.2f}. "
            )
            if leave_days_taken > 0:
                reasoning += (
                    f"{leave_days_taken:.0f} approved leave day(s) taken within balance — no unpaid deduction. "
                )
            else:
                reasoning += "No leave taken in period. "
            reasoning += (
                f"Tax: £{figures['deductions_tax']:,.2f} ({emp.tax_rate:.0%} flat rate). "
                f"Net pay: £{figures['net_pay']:,.2f}. All figures verified — no anomalies."
            )

            approver = db.query(User).first()
            payslip = Payslip(
                employee_id      = emp.id,
                pay_cycle_id     = cycle.id,
                gross_pay        = figures["gross_pay"],
                deductions_leave = figures["deductions_leave"],
                deductions_tax   = figures["deductions_tax"],
                net_pay          = figures["net_pay"],
                days_worked      = figures["days_worked"],
                leave_days_taken = figures["leave_days_taken"],
                llm_decision     = "APPROVE",
                llm_reasoning    = reasoning,
                status           = PayslipStatus.APPROVED.value,
                approved_by      = approver.id if approver else None,
                approved_at      = datetime(2026, 3, 1, 9, 30, 0),
            )
            db.add(payslip)
            db.flush()
            db.refresh(payslip)
            payslips_created += 1
            print(
                f"[OK]   Payslip for {emp.first_name} {emp.last_name}: "
                f"gross=£{figures['gross_pay']:,.2f}, "
                f"tax=£{figures['deductions_tax']:,.2f}, "
                f"net=£{figures['net_pay']:,.2f} [APPROVE]"
            )

        db.commit()
        print(f"\nPayslips: {payslips_created} seeded")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
    print("\n[DONE] Payroll seed complete.")
