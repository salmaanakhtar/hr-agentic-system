from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON, Float, Date
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from typing import Dict, Any
from enum import Enum

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role_id = Column(Integer, ForeignKey("roles.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    role = relationship("Role")

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    department = Column(String, nullable=True)

    user = relationship("User")

class Manager(Base):
    __tablename__ = "managers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    department = Column(String, nullable=True)

    user = relationship("User")

class HR(Base):
    __tablename__ = "hrs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)

    user = relationship("User")


class LeaveType(str, Enum):
    VACATION = "vacation"
    SICK_LEAVE = "sick_leave"
    PERSONAL = "personal"
    MATERNITY = "maternity"
    PATERNITY = "paternity"
    BEREAVEMENT = "bereavement"


class LeaveRequestStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class LeaveBalance(Base):
    __tablename__ = "leave_balances"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("employees.user_id"), index=True)
    leave_type = Column(String, index=True)  # Using string for flexibility
    total_days = Column(Float, default=0.0)
    used_days = Column(Float, default=0.0)
    carried_forward = Column(Float, default=0.0)
    year = Column(Integer, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee")

    @property
    def remaining_days(self) -> float:
        return self.total_days + self.carried_forward - self.used_days


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("employees.user_id"), index=True)
    leave_type = Column(String, index=True)
    start_date = Column(Date, index=True)
    end_date = Column(Date, index=True)
    days_requested = Column(Float)
    status = Column(String, default=LeaveRequestStatus.DRAFT.value, index=True)
    reason = Column(Text, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejected_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee")
    approver = relationship("User", foreign_keys=[approved_by])
    rejector = relationship("User", foreign_keys=[rejected_by])

    @property
    def is_pending(self) -> bool:
        return self.status == LeaveRequestStatus.SUBMITTED.value

    @property
    def is_approved(self) -> bool:
        return self.status == LeaveRequestStatus.APPROVED.value

    @property
    def is_rejected(self) -> bool:
        return self.status == LeaveRequestStatus.REJECTED.value


class PriorityPeriod(Base):
    __tablename__ = "priority_periods"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    start_date = Column(Date, index=True)
    end_date = Column(Date, index=True)
    description = Column(Text, nullable=True)
    is_blackout = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def duration_days(self) -> int:
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return 0
    __tablename__ = "approval_states"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)


class WorkflowState(Base):
    __tablename__ = "workflow_states"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String, unique=True, index=True)
    workflow_type = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    completed_at = Column(DateTime, nullable=True)

    data = Column(JSON, default=dict)
    workflow_metadata = Column(JSON, default=dict)
    steps = Column(JSON, default=list)
    current_step = Column(String, nullable=True)
    next_steps = Column(JSON, default=list)

    version = Column(Integer, default=1)
    checksum = Column(String, nullable=True)
    is_valid = Column(Boolean, default=True)

    ttl_hours = Column(Integer, default=24)
    cleanup_scheduled = Column(Boolean, default=False)

    user = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "user_id": self.user_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "data": self.data or {},
            "metadata": self.workflow_metadata or {},
            "steps": self.steps or [],
            "current_step": self.current_step,
            "next_steps": self.next_steps or [],
            "version": self.version,
            "checksum": self.checksum,
            "is_valid": self.is_valid,
            "ttl_hours": self.ttl_hours,
            "cleanup_scheduled": self.cleanup_scheduled
        }