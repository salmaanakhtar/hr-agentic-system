"""
Payroll API Endpoints

Provides REST API for running pay cycles (triggers PayrollAgent per employee),
viewing pay cycles and payslips, approving payslips, and payroll reports.
"""

import logging
from datetime import datetime, date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

from app.database import get_async_db
from app.models import (
    User,
    Employee,
    PayCycle,
    PayCycleStatus,
    Payslip,
    PayslipStatus,
    PayrollDecision,
)
from agents.payroll_agent import PayrollAgent
from agents.schemas import PayCycleCreate, PayrollRunInput, PayslipApprove
from app.main import get_current_user

router = APIRouter(prefix="/api/payroll", tags=["payroll"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_role(user: User, *roles: str):
    if user.role.name not in roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Required role(s): {', '.join(roles)}"
        )


def _payslip_to_dict(payslip: Payslip, employee_name: str = None) -> dict:
    return {
        "id": payslip.id,
        "employee_id": payslip.employee_id,
        "employee_name": employee_name,
        "pay_cycle_id": payslip.pay_cycle_id,
        "gross_pay": payslip.gross_pay,
        "deductions_leave": payslip.deductions_leave,
        "deductions_tax": payslip.deductions_tax,
        "net_pay": payslip.net_pay,
        "days_worked": payslip.days_worked,
        "leave_days_taken": payslip.leave_days_taken,
        "llm_decision": payslip.llm_decision,
        "llm_reasoning": payslip.llm_reasoning,
        "status": payslip.status,
        "approved_by": payslip.approved_by,
        "approved_at": payslip.approved_at.isoformat() if payslip.approved_at else None,
        "created_at": payslip.created_at.isoformat() if payslip.created_at else None,
        "updated_at": payslip.updated_at.isoformat() if payslip.updated_at else None,
    }


def _cycle_to_dict(cycle: PayCycle, payslip_count: int = 0) -> dict:
    return {
        "id": cycle.id,
        "period_start": cycle.period_start.isoformat() if cycle.period_start else None,
        "period_end": cycle.period_end.isoformat() if cycle.period_end else None,
        "status": cycle.status,
        "run_by": cycle.run_by,
        "run_at": cycle.run_at.isoformat() if cycle.run_at else None,
        "payslip_count": payslip_count,
        "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
        "updated_at": cycle.updated_at.isoformat() if cycle.updated_at else None,
    }


async def _get_employee_name(db: AsyncSession, employee_id: int) -> str:
    result = await db.execute(
        select(Employee).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if emp:
        return f"{emp.first_name} {emp.last_name}"
    return "Unknown"


# ---------------------------------------------------------------------------
# 6.3.1 — POST /run-cycle  (HR/Admin)
# ---------------------------------------------------------------------------

@router.post("/run-cycle")
async def run_pay_cycle(
    body: PayCycleCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Create a new pay cycle and run the PayrollAgent for every active employee
    who has a base_salary set.

    The agent calculates gross pay, unpaid-leave deductions, flat-rate tax,
    and saves a DRAFT payslip per employee with an APPROVE/HOLD/FLAG decision.

    Returns the new pay cycle record and per-employee results.
    """
    _require_role(current_user, "hr", "admin")

    # Validate dates
    try:
        period_start = date.fromisoformat(body.period_start)
        period_end = date.fromisoformat(body.period_end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    if period_start >= period_end:
        raise HTTPException(status_code=400, detail="period_start must be before period_end")

    # Create the PayCycle record
    cycle = PayCycle(
        period_start=period_start,
        period_end=period_end,
        status=PayCycleStatus.RUNNING.value,
        run_by=current_user.id,
        run_at=datetime.utcnow(),
    )
    db.add(cycle)
    await db.flush()
    await db.refresh(cycle)
    cycle_id = cycle.id
    await db.commit()

    logger.info(
        f"Pay cycle {cycle_id} created by user {current_user.id}: "
        f"{body.period_start} to {body.period_end}"
    )

    # Fetch all employees with a salary set
    emp_result = await db.execute(
        select(Employee).where(Employee.base_salary.isnot(None))
    )
    employees = emp_result.scalars().all()

    if not employees:
        # No employees to process — mark failed
        result_cycle = await db.execute(select(PayCycle).where(PayCycle.id == cycle_id))
        db_cycle = result_cycle.scalar_one_or_none()
        if db_cycle:
            db_cycle.status = PayCycleStatus.FAILED.value
            await db.commit()
        raise HTTPException(
            status_code=422,
            detail="No employees with salary data found. Pay cycle marked as failed."
        )

    # Run PayrollAgent per employee
    agent = PayrollAgent()
    results = []
    failed_count = 0

    for emp in employees:
        try:
            agent_input = PayrollRunInput(
                employee_id=emp.id,
                pay_cycle_id=cycle_id,
                period_start=body.period_start,
                period_end=body.period_end,
                run_by_user_id=current_user.id,
            )
            output = await agent.execute(agent_input)
            results.append({
                "employee_id": emp.id,
                "employee_name": f"{emp.first_name} {emp.last_name}",
                "success": output.success,
                "decision": output.decision,
                "payslip_id": output.payslip_id,
                "net_pay": output.net_pay,
                "message": output.message,
            })
            if not output.success:
                failed_count += 1
            logger.info(
                f"Payroll complete for employee {emp.id}: {output.decision}, "
                f"net_pay={output.net_pay}"
            )
        except Exception as e:
            logger.error(f"Payroll failed for employee {emp.id}: {e}", exc_info=True)
            failed_count += 1
            results.append({
                "employee_id": emp.id,
                "employee_name": f"{emp.first_name} {emp.last_name}",
                "success": False,
                "decision": "HOLD",
                "payslip_id": None,
                "net_pay": None,
                "message": f"Error: {str(e)}",
            })

    # Update cycle status
    final_status = (
        PayCycleStatus.FAILED.value
        if failed_count == len(employees)
        else PayCycleStatus.COMPLETED.value
    )

    result_cycle = await db.execute(select(PayCycle).where(PayCycle.id == cycle_id))
    db_cycle = result_cycle.scalar_one_or_none()
    if db_cycle:
        db_cycle.status = final_status
        await db.commit()

    return {
        "pay_cycle_id": cycle_id,
        "period_start": body.period_start,
        "period_end": body.period_end,
        "status": final_status,
        "total_employees": len(employees),
        "failed_count": failed_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 6.3.2 — GET /cycles  (HR/Admin)
# ---------------------------------------------------------------------------

@router.get("/cycles")
async def list_pay_cycles(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List all pay cycles with payslip counts. HR/Admin only."""
    _require_role(current_user, "hr", "admin")

    filters = []
    if status:
        filters.append(PayCycle.status == status)

    query = select(PayCycle)
    if filters:
        query = query.where(and_(*filters))

    query = query.order_by(PayCycle.period_start.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    cycles = result.scalars().all()

    output = []
    for cycle in cycles:
        count_result = await db.execute(
            select(func.count(Payslip.id)).where(Payslip.pay_cycle_id == cycle.id)
        )
        payslip_count = count_result.scalar() or 0
        output.append(_cycle_to_dict(cycle, payslip_count))

    return {
        "pay_cycles": output,
        "total": len(output),
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# 6.3.3 — GET /cycles/{id}  (HR/Admin)
# ---------------------------------------------------------------------------

@router.get("/cycles/{cycle_id}")
async def get_pay_cycle(
    cycle_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a single pay cycle with all its payslips. HR/Admin only."""
    _require_role(current_user, "hr", "admin")

    result = await db.execute(select(PayCycle).where(PayCycle.id == cycle_id))
    cycle = result.scalar_one_or_none()

    if not cycle:
        raise HTTPException(status_code=404, detail="Pay cycle not found")

    payslip_result = await db.execute(
        select(Payslip)
        .where(Payslip.pay_cycle_id == cycle_id)
        .order_by(Payslip.employee_id)
    )
    payslips = payslip_result.scalars().all()

    payslip_dicts = []
    for ps in payslips:
        emp_name = await _get_employee_name(db, ps.employee_id)
        payslip_dicts.append(_payslip_to_dict(ps, emp_name))

    cycle_dict = _cycle_to_dict(cycle, len(payslips))
    cycle_dict["payslips"] = payslip_dicts

    return cycle_dict


# ---------------------------------------------------------------------------
# 6.3.4 — GET /payslips  (own or all for HR/Admin)
# ---------------------------------------------------------------------------

@router.get("/payslips")
async def list_payslips(
    cycle_id: Optional[int] = Query(None, description="Filter by pay cycle"),
    employee_id: Optional[int] = Query(None, description="Filter by employee (HR/Admin only)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    List payslips.
    - Employees see only their own payslips (matched via employees.user_id).
    - HR/Admin can see all and filter by employee_id.
    """
    filters = []

    if current_user.role.name == "employee":
        # Resolve employee record for current user
        emp_result = await db.execute(
            select(Employee).where(Employee.user_id == current_user.id)
        )
        emp = emp_result.scalar_one_or_none()
        if not emp:
            return {"payslips": [], "total": 0, "limit": limit, "offset": offset}
        filters.append(Payslip.employee_id == emp.id)
    elif employee_id is not None:
        _require_role(current_user, "hr", "admin", "manager")
        filters.append(Payslip.employee_id == employee_id)

    if cycle_id is not None:
        filters.append(Payslip.pay_cycle_id == cycle_id)
    if status:
        filters.append(Payslip.status == status)

    query = select(Payslip)
    if filters:
        query = query.where(and_(*filters))

    query = query.order_by(Payslip.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    payslips = result.scalars().all()

    output = []
    for ps in payslips:
        emp_name = await _get_employee_name(db, ps.employee_id)
        output.append(_payslip_to_dict(ps, emp_name))

    return {
        "payslips": output,
        "total": len(output),
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# 6.3.5 — GET /payslips/{id}
# ---------------------------------------------------------------------------

@router.get("/payslips/{payslip_id}")
async def get_payslip(
    payslip_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get a single payslip with full LLM reasoning.
    Employees can view their own; HR/Admin can view any.
    """
    result = await db.execute(select(Payslip).where(Payslip.id == payslip_id))
    payslip = result.scalar_one_or_none()

    if not payslip:
        raise HTTPException(status_code=404, detail="Payslip not found")

    # Employees may only see their own
    if current_user.role.name == "employee":
        emp_result = await db.execute(
            select(Employee).where(Employee.user_id == current_user.id)
        )
        emp = emp_result.scalar_one_or_none()
        if not emp or emp.id != payslip.employee_id:
            raise HTTPException(status_code=403, detail="Access denied")

    emp_name = await _get_employee_name(db, payslip.employee_id)

    # Attach pay cycle period for context
    cycle_result = await db.execute(
        select(PayCycle).where(PayCycle.id == payslip.pay_cycle_id)
    )
    cycle = cycle_result.scalar_one_or_none()

    data = _payslip_to_dict(payslip, emp_name)
    if cycle:
        data["period_start"] = cycle.period_start.isoformat() if cycle.period_start else None
        data["period_end"] = cycle.period_end.isoformat() if cycle.period_end else None

    return data


# ---------------------------------------------------------------------------
# 6.3.6 — PUT /payslips/{id}/approve  (HR/Admin)
# ---------------------------------------------------------------------------

@router.put("/payslips/{payslip_id}/approve")
async def approve_payslip(
    payslip_id: int,
    body: PayslipApprove = PayslipApprove(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Approve a DRAFT payslip. Sets status to APPROVED.
    HR/Admin only.
    """
    _require_role(current_user, "hr", "admin")

    result = await db.execute(select(Payslip).where(Payslip.id == payslip_id))
    payslip = result.scalar_one_or_none()

    if not payslip:
        raise HTTPException(status_code=404, detail="Payslip not found")

    if payslip.status != PayslipStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve payslip with status '{payslip.status}'. Only DRAFT payslips can be approved."
        )

    payslip_id_cached = payslip.id
    payslip.status = PayslipStatus.APPROVED.value
    payslip.approved_by = current_user.id
    payslip.approved_at = datetime.utcnow()
    payslip.updated_at = datetime.utcnow()
    await db.commit()

    logger.info(f"User {current_user.id} approved payslip {payslip_id_cached}")

    return {
        "message": "Payslip approved successfully",
        "payslip_id": payslip_id_cached,
        "status": PayslipStatus.APPROVED.value,
        "approved_by": current_user.id,
        "notes": body.notes,
    }


# ---------------------------------------------------------------------------
# 6.3.7 — GET /reports  (HR/Admin)
# ---------------------------------------------------------------------------

@router.get("/reports")
async def get_payroll_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Payroll summary statistics. HR/Admin only."""
    _require_role(current_user, "hr", "admin")

    # Total cycles by status
    for_status = [PayCycleStatus.PENDING, PayCycleStatus.RUNNING, PayCycleStatus.COMPLETED, PayCycleStatus.FAILED]
    cycle_counts = {}
    for s in for_status:
        r = await db.execute(
            select(func.count(PayCycle.id)).where(PayCycle.status == s.value)
        )
        cycle_counts[s.value] = r.scalar() or 0

    # Total payslips by status
    for_ps = [PayslipStatus.DRAFT, PayslipStatus.APPROVED, PayslipStatus.PAID]
    payslip_counts = {}
    for s in for_ps:
        r = await db.execute(
            select(func.count(Payslip.id)).where(Payslip.status == s.value)
        )
        payslip_counts[s.value] = r.scalar() or 0

    # LLM decision breakdown
    decision_counts = {}
    for d in [PayrollDecision.APPROVE, PayrollDecision.HOLD, PayrollDecision.FLAG]:
        r = await db.execute(
            select(func.count(Payslip.id)).where(Payslip.llm_decision == d.value)
        )
        decision_counts[d.value] = r.scalar() or 0

    # Total and avg net pay across approved payslips
    total_net_result = await db.execute(
        select(func.sum(Payslip.net_pay), func.avg(Payslip.net_pay), func.count(Payslip.id))
        .where(Payslip.status == PayslipStatus.APPROVED.value)
    )
    row = total_net_result.one()
    total_net = round(row[0] or 0.0, 2)
    avg_net = round(row[1] or 0.0, 2)
    approved_count = row[2] or 0

    # Most recent completed cycle
    recent_result = await db.execute(
        select(PayCycle)
        .where(PayCycle.status == PayCycleStatus.COMPLETED.value)
        .order_by(PayCycle.period_end.desc())
        .limit(1)
    )
    recent_cycle = recent_result.scalar_one_or_none()

    return {
        "summary": {
            "total_cycles": sum(cycle_counts.values()),
            "cycles_by_status": cycle_counts,
            "total_payslips": sum(payslip_counts.values()),
            "payslips_by_status": payslip_counts,
        },
        "llm_decisions": decision_counts,
        "financials": {
            "total_net_pay_approved": total_net,
            "avg_net_pay_approved": avg_net,
            "approved_payslip_count": approved_count,
        },
        "most_recent_cycle": _cycle_to_dict(recent_cycle) if recent_cycle else None,
    }
