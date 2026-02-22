"""
Expense Management API Endpoints

Provides REST API for expense claim submission (with receipt upload),
approval workflows, policy management, and usage reports.
"""

import os
import uuid
import logging
from datetime import datetime, date
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.database import get_async_db
from app.models import (
    User,
    Employee,
    Manager,
    Expense,
    ExpensePolicy,
    ExpenseStatus,
)
from agents.expense_agent import ExpenseAgent
from agents.schemas import ExpenseSubmitInput, ExpenseApprovalOutput
from app.main import get_current_user

router = APIRouter(prefix="/api/expenses", tags=["expenses"])
logger = logging.getLogger(__name__)

# Resolve uploads directory relative to this file's location
_UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "uploads",
    "receipts",
)
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_role(user: User, *roles: str):
    if user.role.name not in roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Required role(s): {', '.join(roles)}"
        )


async def _get_employee_name(db: AsyncSession, user_id: int) -> str:
    result = await db.execute(
        select(Employee).where(Employee.user_id == user_id)
    )
    emp = result.scalar_one_or_none()
    if emp:
        return f"{emp.first_name} {emp.last_name}"
    return "Unknown"


def _expense_to_dict(expense: Expense, employee_name: str = None) -> dict:
    return {
        "id": expense.id,
        "user_id": expense.user_id,
        "employee_name": employee_name,
        "amount": expense.amount,
        "category": expense.category,
        "vendor": expense.vendor,
        "date": expense.date.isoformat() if expense.date else None,
        "description": expense.description,
        "receipt_filename": expense.receipt_filename,
        "receipt_url": expense.receipt_url,
        "ocr_confidence": expense.ocr_confidence,
        "ocr_extracted": expense.ocr_extracted,
        "status": expense.status,
        "llm_decision": expense.llm_decision,
        "llm_reasoning": expense.llm_reasoning,
        "submitted_at": expense.submitted_at.isoformat() if expense.submitted_at else None,
        "reviewed_at": expense.reviewed_at.isoformat() if expense.reviewed_at else None,
        "rejection_reason": expense.rejection_reason,
        "created_at": expense.created_at.isoformat() if expense.created_at else None,
    }


# ---------------------------------------------------------------------------
# 4.4.1: Employee Endpoints
# ---------------------------------------------------------------------------

@router.post("/submit")
async def submit_expense(
    # Form fields
    amount: float = Form(..., description="Expense amount in dollars"),
    category: str = Form(..., description="Category: meals, travel, equipment, entertainment, office_supplies, other"),
    vendor: Optional[str] = Form(None, description="Vendor or merchant name"),
    date: str = Form(..., description="Expense date (YYYY-MM-DD)"),
    description: Optional[str] = Form(None, description="Brief description of the expense"),
    # Optional receipt file
    receipt: Optional[UploadFile] = File(None, description="Receipt image (jpg/png/pdf, max 5MB)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Submit a new expense claim with optional receipt upload.

    The LLM agent will:
    1. Run OCR on the receipt (if uploaded) and compare with submitted data
    2. Check amount against policy limits for the category
    3. Detect duplicate submissions within 7 days
    4. Check monthly spend totals
    5. Auto-approve, escalate to manager, or reject based on findings

    Returns the agent decision with full LLM reasoning and OCR results.
    """
    # Validate category
    valid_categories = ["meals", "travel", "equipment", "entertainment", "office_supplies", "other"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        )

    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Handle receipt file upload
    receipt_filename = None
    receipt_path = None

    if receipt and receipt.filename:
        ext = os.path.splitext(receipt.filename)[1].lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{ext}'. Allowed: jpg, jpeg, png, pdf"
            )

        # Read and check size
        file_bytes = await receipt.read()
        if len(file_bytes) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(file_bytes) // 1024}KB). Maximum is 5MB."
            )

        # Save with UUID filename to prevent collisions
        receipt_filename = f"{uuid.uuid4().hex}{ext}"
        receipt_path = os.path.join(_UPLOADS_DIR, receipt_filename)
        os.makedirs(_UPLOADS_DIR, exist_ok=True)

        async with aiofiles.open(receipt_path, "wb") as f:
            await f.write(file_bytes)

        logger.info(f"Saved receipt: {receipt_filename} ({len(file_bytes)} bytes)")

    # Build agent input (user_id always from JWT token, never from form)
    input_data = ExpenseSubmitInput(
        user_id=current_user.id,
        amount=amount,
        category=category,
        vendor=vendor,
        date=date,
        description=description,
        receipt_filename=receipt_filename,
        receipt_path=receipt_path,
    )

    logger.info(
        f"User {current_user.id} submitting {category} expense: "
        f"${amount:.2f} at {vendor or 'unknown'} on {date}"
    )

    agent = ExpenseAgent()
    result = await agent.execute(input_data)

    return result


@router.get("/requests")
async def get_expense_requests(
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    start_date: Optional[str] = Query(None, description="Expenses on or after this date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Expenses on or before this date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get expense history for the current employee."""
    filters = [Expense.user_id == current_user.id]

    if status:
        filters.append(Expense.status == status)
    if category:
        filters.append(Expense.category == category)
    if start_date:
        try:
            filters.append(Expense.date >= datetime.strptime(start_date, "%Y-%m-%d").date())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    if end_date:
        try:
            filters.append(Expense.date <= datetime.strptime(end_date, "%Y-%m-%d").date())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")

    result = await db.execute(
        select(Expense)
        .where(and_(*filters))
        .order_by(Expense.submitted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    expenses = result.scalars().all()

    return {
        "expenses": [_expense_to_dict(e) for e in expenses],
        "total": len(expenses),
        "limit": limit,
        "offset": offset,
    }


@router.get("/requests/{expense_id}")
async def get_expense_by_id(
    expense_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a single expense with full OCR data and LLM reasoning."""
    result = await db.execute(
        select(Expense).where(Expense.id == expense_id)
    )
    expense = result.scalar_one_or_none()

    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    # Employees can only view their own expenses; managers/hr can view any
    if current_user.role.name == "employee" and expense.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    employee_name = await _get_employee_name(db, expense.user_id)
    return _expense_to_dict(expense, employee_name)


@router.put("/requests/{expense_id}/cancel")
async def cancel_expense(
    expense_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Cancel a submitted (pending) expense. Only the submitting employee can cancel."""
    result = await db.execute(
        select(Expense).where(
            and_(Expense.id == expense_id, Expense.user_id == current_user.id)
        )
    )
    expense = result.scalar_one_or_none()

    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    if expense.status != ExpenseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel expense with status '{expense.status}'. Only submitted expenses can be cancelled."
        )

    expense.status = ExpenseStatus.CANCELLED.value
    expense.updated_at = datetime.utcnow()
    await db.commit()

    return {"message": "Expense cancelled successfully", "expense_id": expense_id}


# ---------------------------------------------------------------------------
# 4.4.2: Manager Endpoints
# ---------------------------------------------------------------------------

@router.get("/pending-approvals")
async def get_pending_expense_approvals(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get all expenses pending manager review.
    Accessible by manager and hr roles.
    """
    _require_role(current_user, "manager", "hr", "admin")

    result = await db.execute(
        select(Expense)
        .where(Expense.status == ExpenseStatus.SUBMITTED.value)
        .order_by(Expense.submitted_at.asc())
    )
    expenses = result.scalars().all()

    # Attach employee names via JOIN
    output = []
    for expense in expenses:
        employee_name = await _get_employee_name(db, expense.user_id)
        output.append(_expense_to_dict(expense, employee_name))

    return {"pending_approvals": output, "total": len(output)}


@router.post("/requests/{expense_id}/approve")
async def approve_expense(
    expense_id: int,
    comments: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Approve a pending expense claim. Manager/HR only."""
    _require_role(current_user, "manager", "hr", "admin")

    result = await db.execute(
        select(Expense).where(Expense.id == expense_id)
    )
    expense = result.scalar_one_or_none()

    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    if expense.status != ExpenseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve expense with status '{expense.status}'"
        )

    # Cache values before commit
    expense_id_cached = expense.id
    expense.status = ExpenseStatus.APPROVED.value
    expense.reviewed_at = datetime.utcnow()
    expense.reviewed_by = current_user.id
    await db.commit()

    logger.info(f"Manager {current_user.id} approved expense {expense_id_cached}")

    return ExpenseApprovalOutput(
        success=True,
        message="Expense approved successfully",
        decision="approved",
        expense_id=expense_id_cached,
        comments=comments,
    )


@router.post("/requests/{expense_id}/reject")
async def reject_expense(
    expense_id: int,
    reason: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Reject a pending expense claim with a required reason. Manager/HR only."""
    _require_role(current_user, "manager", "hr", "admin")

    if not reason or not reason.strip():
        raise HTTPException(status_code=400, detail="Rejection reason is required")

    result = await db.execute(
        select(Expense).where(Expense.id == expense_id)
    )
    expense = result.scalar_one_or_none()

    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    if expense.status != ExpenseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject expense with status '{expense.status}'"
        )

    expense_id_cached = expense.id
    expense.status = ExpenseStatus.REJECTED.value
    expense.reviewed_at = datetime.utcnow()
    expense.reviewed_by = current_user.id
    expense.rejection_reason = reason.strip()
    await db.commit()

    logger.info(f"Manager {current_user.id} rejected expense {expense_id_cached}: {reason}")

    return ExpenseApprovalOutput(
        success=True,
        message="Expense rejected",
        decision="rejected",
        expense_id=expense_id_cached,
        rejection_reason=reason.strip(),
        next_steps=["Employee has been notified", "Expense marked as rejected"],
    )


# ---------------------------------------------------------------------------
# 4.4.3: HR / Admin Endpoints
# ---------------------------------------------------------------------------

@router.get("/policies")
async def get_expense_policies(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get all expense policies. Accessible by all authenticated users."""
    result = await db.execute(
        select(ExpensePolicy).order_by(ExpensePolicy.category)
    )
    policies = result.scalars().all()

    return {
        "policies": [
            {
                "id": p.id,
                "category": p.category,
                "max_amount": p.max_amount,
                "approval_threshold": p.approval_threshold,
                "requires_receipt": p.requires_receipt,
                "description": p.description,
            }
            for p in policies
        ]
    }


@router.put("/policies/{policy_id}")
async def update_expense_policy(
    policy_id: int,
    max_amount: Optional[float] = None,
    approval_threshold: Optional[float] = None,
    requires_receipt: Optional[bool] = None,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Update an expense policy limit. HR/Admin only."""
    _require_role(current_user, "hr", "admin")

    result = await db.execute(
        select(ExpensePolicy).where(ExpensePolicy.id == policy_id)
    )
    policy = result.scalar_one_or_none()

    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    if max_amount is not None:
        if approval_threshold is not None and approval_threshold >= max_amount:
            raise HTTPException(
                status_code=400,
                detail="approval_threshold must be less than max_amount"
            )
        policy.max_amount = max_amount
    if approval_threshold is not None:
        policy.approval_threshold = approval_threshold
    if requires_receipt is not None:
        policy.requires_receipt = requires_receipt
    if description is not None:
        policy.description = description

    policy.updated_at = datetime.utcnow()
    await db.commit()

    return {
        "message": "Policy updated successfully",
        "policy": {
            "id": policy.id,
            "category": policy.category,
            "max_amount": policy.max_amount,
            "approval_threshold": policy.approval_threshold,
            "requires_receipt": policy.requires_receipt,
        },
    }


@router.get("/reports")
async def get_expense_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get expense usage statistics and reports. Manager/HR/Admin only."""
    _require_role(current_user, "manager", "hr", "admin")

    # Total spend by category (approved only)
    category_result = await db.execute(
        select(Expense.category, func.sum(Expense.amount), func.count(Expense.id))
        .where(Expense.status == ExpenseStatus.APPROVED.value)
        .group_by(Expense.category)
    )
    category_stats = [
        {"category": row[0], "total_approved": round(row[1], 2), "count": row[2]}
        for row in category_result.all()
    ]

    # Approval rate
    total_result = await db.execute(select(func.count(Expense.id)))
    total = total_result.scalar() or 0

    approved_result = await db.execute(
        select(func.count(Expense.id)).where(Expense.status == ExpenseStatus.APPROVED.value)
    )
    approved = approved_result.scalar() or 0

    pending_result = await db.execute(
        select(func.count(Expense.id)).where(Expense.status == ExpenseStatus.SUBMITTED.value)
    )
    pending = pending_result.scalar() or 0

    rejected_result = await db.execute(
        select(func.count(Expense.id)).where(Expense.status == ExpenseStatus.REJECTED.value)
    )
    rejected = rejected_result.scalar() or 0

    # Auto-approved by LLM vs escalated
    auto_approved_result = await db.execute(
        select(func.count(Expense.id)).where(Expense.llm_decision == "AUTO_APPROVE")
    )
    auto_approved = auto_approved_result.scalar() or 0

    escalated_result = await db.execute(
        select(func.count(Expense.id)).where(Expense.llm_decision == "ESCALATE")
    )
    escalated = escalated_result.scalar() or 0

    return {
        "summary": {
            "total_claims": total,
            "approved": approved,
            "pending": pending,
            "rejected": rejected,
            "approval_rate": round((approved / total * 100) if total > 0 else 0, 1),
        },
        "llm_decisions": {
            "auto_approved": auto_approved,
            "escalated_to_manager": escalated,
            "auto_approval_rate": round((auto_approved / total * 100) if total > 0 else 0, 1),
        },
        "by_category": category_stats,
    }
