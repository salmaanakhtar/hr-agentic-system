"""
Leave Management API Endpoints

Provides REST API for leave request submission, approval workflows,
balance management, and team calendar views.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, extract
from typing import List, Optional
from datetime import datetime, date
import logging

from app.database import get_async_db
from app.models import (
    User,
    LeaveBalance,
    LeaveRequest,
    PriorityPeriod,
    LeaveRequestStatus,
    Employee,
    Manager
)
from agents.leave_agent import LeaveAgent
from agents.schemas import LeaveRequestInput, LeaveValidationOutput
from agents.state_manager import state_manager
from app.main import get_current_user

router = APIRouter(prefix="/api/leave", tags=["leave"])
logger = logging.getLogger(__name__)


# ============================================================================
# 3.4.1: Leave Request Endpoints
# ============================================================================

@router.post("/request", response_model=LeaveValidationOutput)
async def submit_leave_request(
    request_data: LeaveRequestInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Submit a new leave request for validation and processing.

    The LLM agent will:
    1. Check leave balance
    2. Detect conflicts with existing leave and blackout periods
    3. Validate business rules
    4. Auto-approve or escalate based on company policies

    Returns the validation result with LLM reasoning.
    """
    # Override user_id from JWT token for security
    request_data.user_id = current_user.id

    logger.info(f"User {current_user.id} submitting leave request: {request_data.leave_type} from {request_data.start_date} to {request_data.end_date}")

    # Execute Leave Agent
    agent = LeaveAgent()
    result = await agent.execute(request_data)

    if not result.success:
        logger.warning(f"Leave request failed validation: {result.reasoning}")

    return result


@router.get("/requests")
async def get_leave_requests(
    status: Optional[str] = Query(None, description="Filter by status (draft, submitted, approved, rejected, cancelled)"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Filter requests after this date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter requests before this date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get leave requests with optional filtering.

    Employees see only their own requests.
    Managers see their team's requests.
    HR/Admin see all requests.
    """
    query = select(LeaveRequest)

    # Role-based access control
    if current_user.role.name == "employee":
        # Employees see only their own requests
        query = query.where(LeaveRequest.user_id == current_user.id)
    elif current_user.role.name == "manager":
        # Managers see their team's requests
        # TODO: Add department/team filtering once org structure is in place
        pass
    # HR and Admin see all requests (no additional filter)

    # Apply filters
    if status:
        try:
            status_enum = LeaveRequestStatus[status.upper()].value
            query = query.where(LeaveRequest.status == status_enum)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if user_id:
        # Non-admin users can only query their own data
        if current_user.role.name not in ["hr", "admin"] and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Cannot access other users' leave requests")
        query = query.where(LeaveRequest.user_id == user_id)

    if start_date:
        query = query.where(LeaveRequest.start_date >= datetime.strptime(start_date, "%Y-%m-%d").date())

    if end_date:
        query = query.where(LeaveRequest.end_date <= datetime.strptime(end_date, "%Y-%m-%d").date())

    # Order by most recent first
    query = query.order_by(LeaveRequest.created_at.desc())

    # Pagination
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    requests = result.scalars().all()

    return {
        "total": len(requests),
        "limit": limit,
        "offset": offset,
        "requests": [
            {
                "id": req.id,
                "user_id": req.user_id,
                "leave_type": req.leave_type,
                "start_date": req.start_date.isoformat(),
                "end_date": req.end_date.isoformat(),
                "days_requested": req.days_requested,
                "status": req.status,
                "reason": req.reason,
                "submitted_at": req.submitted_at.isoformat() if req.submitted_at else None,
                "approved_at": req.approved_at.isoformat() if req.approved_at else None,
                "approved_by": req.approved_by,
                "rejected_at": req.rejected_at.isoformat() if req.rejected_at else None,
                "rejected_by": req.rejected_by,
                "rejection_reason": req.rejection_reason,
                "created_at": req.created_at.isoformat()
            }
            for req in requests
        ]
    }


@router.get("/balance")
async def get_leave_balance(
    user_id: Optional[int] = Query(None, description="User ID (defaults to current user)"),
    year: Optional[int] = Query(None, description="Year (defaults to current year)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get leave balance for a user.

    Returns balance by leave type (vacation, sick leave, personal, etc.)
    """
    # Default to current user
    target_user_id = user_id if user_id else current_user.id

    # Access control
    if current_user.role.name not in ["hr", "admin"] and target_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot access other users' leave balance")

    # Default to current year
    target_year = year if year else date.today().year

    query = select(LeaveBalance).where(
        and_(
            LeaveBalance.user_id == target_user_id,
            LeaveBalance.year == target_year
        )
    )

    result = await db.execute(query)
    balances = result.scalars().all()

    if not balances:
        # Return empty balance if not found
        return {
            "user_id": target_user_id,
            "year": target_year,
            "balances": []
        }

    return {
        "user_id": target_user_id,
        "year": target_year,
        "balances": [
            {
                "leave_type": balance.leave_type,
                "total_days": balance.total_days,
                "used_days": balance.used_days,
                "carried_forward": balance.carried_forward,
                "remaining_days": balance.remaining_days
            }
            for balance in balances
        ]
    }


@router.put("/request/{request_id}/cancel")
async def cancel_leave_request(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Cancel a pending leave request.

    Only the request owner can cancel, and only if status is DRAFT or SUBMITTED.
    """
    # Get the request
    result = await db.execute(
        select(LeaveRequest).where(LeaveRequest.id == request_id)
    )
    leave_request = result.scalar_one_or_none()

    if not leave_request:
        raise HTTPException(status_code=404, detail="Leave request not found")

    # Check ownership
    if leave_request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot cancel other users' requests")

    # Check if cancellable
    if leave_request.status not in [LeaveRequestStatus.DRAFT.value, LeaveRequestStatus.SUBMITTED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel request with status: {leave_request.status}"
        )

    # Cancel the request
    leave_request.status = LeaveRequestStatus.CANCELLED.value
    leave_request.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"User {current_user.id} cancelled leave request {request_id}")

    return {
        "success": True,
        "message": "Leave request cancelled successfully",
        "request_id": request_id
    }


# ============================================================================
# 3.4.2: Manager Approval Endpoints
# ============================================================================

@router.get("/pending-approvals")
async def get_pending_approvals(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get all pending leave requests requiring approval from current user.

    Only managers and HR can access this endpoint.
    """
    if current_user.role.name not in ["manager", "hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only managers and HR can view pending approvals")

    # Get pending leave requests
    # TODO: Filter by department/team once org structure is in place
    query = select(LeaveRequest).where(
        LeaveRequest.status == LeaveRequestStatus.SUBMITTED.value
    ).order_by(LeaveRequest.submitted_at.asc())

    result = await db.execute(query)
    requests = result.scalars().all()

    return {
        "total": len(requests),
        "pending_approvals": [
            {
                "id": req.id,
                "user_id": req.user_id,
                "leave_type": req.leave_type,
                "start_date": req.start_date.isoformat(),
                "end_date": req.end_date.isoformat(),
                "days_requested": req.days_requested,
                "reason": req.reason,
                "submitted_at": req.submitted_at.isoformat() if req.submitted_at else None
            }
            for req in requests
        ]
    }


@router.post("/approve/{request_id}")
async def approve_or_reject_leave(
    request_id: int,
    decision: str = Query(..., description="Decision: 'approved' or 'rejected'"),
    comments: Optional[str] = Query(None, description="Approval/rejection comments"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Approve or reject a leave request.

    Only managers and HR can approve/reject requests.
    Updates the leave request status and balance accordingly.
    """
    if current_user.role.name not in ["manager", "hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only managers and HR can approve/reject requests")

    if decision not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Decision must be 'approved' or 'rejected'")

    # Get the request
    result = await db.execute(
        select(LeaveRequest).where(LeaveRequest.id == request_id)
    )
    leave_request = result.scalar_one_or_none()

    if not leave_request:
        raise HTTPException(status_code=404, detail="Leave request not found")

    # Check if request is in submitted status
    if leave_request.status != LeaveRequestStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve/reject request with status: {leave_request.status}"
        )

    if decision == "approved":
        # Approve the request
        leave_request.status = LeaveRequestStatus.APPROVED.value
        leave_request.approved_at = datetime.utcnow()
        leave_request.approved_by = current_user.id

        # Deduct from leave balance
        current_year = date.today().year
        balance_result = await db.execute(
            select(LeaveBalance).where(
                and_(
                    LeaveBalance.user_id == leave_request.user_id,
                    LeaveBalance.leave_type == leave_request.leave_type,
                    LeaveBalance.year == current_year
                )
            )
        )
        balance = balance_result.scalar_one_or_none()

        if balance:
            balance.used_days += leave_request.days_requested
            balance.updated_at = datetime.utcnow()

        logger.info(f"Manager {current_user.id} approved leave request {request_id}")

    else:  # rejected
        leave_request.status = LeaveRequestStatus.REJECTED.value
        leave_request.rejected_at = datetime.utcnow()
        leave_request.rejected_by = current_user.id
        leave_request.rejection_reason = comments or "Rejected by manager"

        logger.info(f"Manager {current_user.id} rejected leave request {request_id}")

    leave_request.updated_at = datetime.utcnow()
    await db.commit()

    return {
        "success": True,
        "message": f"Leave request {decision} successfully",
        "request_id": request_id,
        "status": leave_request.status
    }


@router.get("/team-calendar")
async def get_team_calendar(
    department: Optional[str] = Query(None, description="Filter by department"),
    month: Optional[str] = Query(None, description="Month in YYYY-MM format"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get team calendar showing approved leave and priority periods.

    Returns all approved leave for the team plus blackout/priority periods.
    """
    if current_user.role.name not in ["manager", "hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only managers and HR can view team calendar")

    # Parse month or default to current month
    if month:
        try:
            target_date = datetime.strptime(month, "%Y-%m")
            start_of_month = target_date.date().replace(day=1)
            # Get last day of month
            if target_date.month == 12:
                end_of_month = target_date.date().replace(year=target_date.year + 1, month=1, day=1)
            else:
                end_of_month = target_date.date().replace(month=target_date.month + 1, day=1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    else:
        today = date.today()
        start_of_month = today.replace(day=1)
        if today.month == 12:
            end_of_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            end_of_month = today.replace(month=today.month + 1, day=1)

    # Get approved leave requests for the month
    leave_query = select(LeaveRequest).where(
        and_(
            LeaveRequest.status == LeaveRequestStatus.APPROVED.value,
            or_(
                and_(LeaveRequest.start_date <= start_of_month, LeaveRequest.end_date >= start_of_month),
                and_(LeaveRequest.start_date <= end_of_month, LeaveRequest.end_date >= end_of_month),
                and_(LeaveRequest.start_date >= start_of_month, LeaveRequest.end_date <= end_of_month)
            )
        )
    )

    # TODO: Add department filter once org structure is in place

    leave_result = await db.execute(leave_query)
    leave_requests = leave_result.scalars().all()

    # Get priority periods for the month
    priority_query = select(PriorityPeriod).where(
        or_(
            and_(PriorityPeriod.start_date <= start_of_month, PriorityPeriod.end_date >= start_of_month),
            and_(PriorityPeriod.start_date <= end_of_month, PriorityPeriod.end_date >= end_of_month),
            and_(PriorityPeriod.start_date >= start_of_month, PriorityPeriod.end_date <= end_of_month)
        )
    )

    priority_result = await db.execute(priority_query)
    priority_periods = priority_result.scalars().all()

    return {
        "month": month or f"{start_of_month.year}-{start_of_month.month:02d}",
        "leave_requests": [
            {
                "id": req.id,
                "user_id": req.user_id,
                "leave_type": req.leave_type,
                "start_date": req.start_date.isoformat(),
                "end_date": req.end_date.isoformat(),
                "days_requested": req.days_requested
            }
            for req in leave_requests
        ],
        "priority_periods": [
            {
                "id": period.id,
                "name": period.name,
                "start_date": period.start_date.isoformat(),
                "end_date": period.end_date.isoformat(),
                "is_blackout": period.is_blackout,
                "description": period.description
            }
            for period in priority_periods
        ]
    }


# ============================================================================
# 3.4.3: Admin Endpoints
# ============================================================================

@router.post("/balance/adjust")
async def adjust_leave_balance(
    user_id: int,
    leave_type: str,
    adjustment: float = Query(..., description="Amount to adjust (positive or negative)"),
    reason: str = Query(..., description="Reason for adjustment"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Manually adjust a user's leave balance.

    Only HR and Admin can adjust balances.
    """
    if current_user.role.name not in ["hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only HR and Admin can adjust leave balances")

    current_year = date.today().year

    # Get or create balance
    result = await db.execute(
        select(LeaveBalance).where(
            and_(
                LeaveBalance.user_id == user_id,
                LeaveBalance.leave_type == leave_type,
                LeaveBalance.year == current_year
            )
        )
    )
    balance = result.scalar_one_or_none()

    if not balance:
        # Create new balance
        balance = LeaveBalance(
            user_id=user_id,
            leave_type=leave_type,
            total_days=20.0 + adjustment,  # Default + adjustment
            used_days=0.0,
            carried_forward=0.0,
            year=current_year
        )
        db.add(balance)
    else:
        # Adjust existing balance
        balance.total_days += adjustment
        balance.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Admin {current_user.id} adjusted leave balance for user {user_id}: {adjustment} days ({reason})")

    return {
        "success": True,
        "message": f"Leave balance adjusted by {adjustment} days",
        "user_id": user_id,
        "leave_type": leave_type,
        "new_total": balance.total_days,
        "remaining": balance.remaining_days
    }


@router.get("/reports")
async def get_leave_reports(
    start_date: Optional[str] = Query(None, description="Report start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Report end date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Generate leave usage statistics and reports.

    Only HR and Admin can access reports.
    """
    if current_user.role.name not in ["hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only HR and Admin can access reports")

    # Default to current year if no dates provided
    if not start_date:
        start_date = f"{date.today().year}-01-01"
    if not end_date:
        end_date = f"{date.today().year}-12-31"

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Total leave requests by status
    status_query = select(
        LeaveRequest.status,
        func.count(LeaveRequest.id).label("count")
    ).where(
        and_(
            LeaveRequest.created_at >= start,
            LeaveRequest.created_at <= end
        )
    ).group_by(LeaveRequest.status)

    status_result = await db.execute(status_query)
    status_counts = {row.status: row.count for row in status_result}

    # Total days requested by leave type
    type_query = select(
        LeaveRequest.leave_type,
        func.sum(LeaveRequest.days_requested).label("total_days")
    ).where(
        and_(
            LeaveRequest.created_at >= start,
            LeaveRequest.created_at <= end,
            LeaveRequest.status == LeaveRequestStatus.APPROVED.value
        )
    ).group_by(LeaveRequest.leave_type)

    type_result = await db.execute(type_query)
    type_totals = {row.leave_type: float(row.total_days) for row in type_result}

    # Average approval time
    avg_time_query = select(
        func.avg(
            func.extract('epoch', LeaveRequest.approved_at - LeaveRequest.submitted_at)
        ).label("avg_seconds")
    ).where(
        and_(
            LeaveRequest.submitted_at.isnot(None),
            LeaveRequest.approved_at.isnot(None),
            LeaveRequest.created_at >= start,
            LeaveRequest.created_at <= end
        )
    )

    avg_time_result = await db.execute(avg_time_query)
    avg_seconds = avg_time_result.scalar()
    avg_approval_hours = round(avg_seconds / 3600, 2) if avg_seconds else 0

    return {
        "period": {
            "start_date": start_date,
            "end_date": end_date
        },
        "status_breakdown": status_counts,
        "leave_type_totals": type_totals,
        "average_approval_time_hours": avg_approval_hours,
        "total_requests": sum(status_counts.values())
    }
