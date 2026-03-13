from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON, Float, Date
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from typing import Dict, Any
from enum import Enum
from pgvector.sqlalchemy import Vector

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
    base_salary = Column(Float, nullable=True)        # Annual gross salary
    pay_frequency = Column(String, default="monthly") # monthly | biweekly
    tax_rate = Column(Float, default=0.20)            # Flat tax rate (e.g. 0.20 = 20%)

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
    llm_decision = Column(String, nullable=True)  # AUTO_APPROVE, ESCALATE, REJECT
    llm_reasoning = Column(Text, nullable=True)  # AI reasoning explanation
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


class ExpenseCategory(str, Enum):
    MEALS = "meals"
    TRAVEL = "travel"
    EQUIPMENT = "equipment"
    ENTERTAINMENT = "entertainment"
    OFFICE_SUPPLIES = "office_supplies"
    OTHER = "other"


class ExpenseStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Expense(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("employees.user_id"), index=True)
    amount = Column(Float, nullable=False)
    category = Column(String, index=True)
    vendor = Column(String, nullable=True)
    date = Column(Date, index=True)
    description = Column(Text, nullable=True)
    receipt_filename = Column(String, nullable=True)
    receipt_path = Column(String, nullable=True)
    ocr_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)  # 0.0-1.0 overall confidence
    ocr_extracted = Column(JSON, nullable=True)     # per-field extracted data + confidence
    status = Column(String, default=ExpenseStatus.DRAFT.value, index=True)
    llm_decision = Column(String, nullable=True)    # AUTO_APPROVE, ESCALATE, REJECT
    llm_reasoning = Column(Text, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee")
    reviewer = relationship("User", foreign_keys=[reviewed_by])

    @property
    def is_pending(self) -> bool:
        return self.status == ExpenseStatus.SUBMITTED.value

    @property
    def is_approved(self) -> bool:
        return self.status == ExpenseStatus.APPROVED.value

    @property
    def receipt_url(self) -> str:
        if self.receipt_filename:
            return f"/uploads/receipts/{self.receipt_filename}"
        return None


class ExpensePolicy(Base):
    __tablename__ = "expense_policies"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, unique=True, index=True)
    max_amount = Column(Float, nullable=False)           # Hard reject above this
    approval_threshold = Column(Float, nullable=False)   # Auto-approve below this
    requires_receipt = Column(Boolean, default=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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


# ---------------------------------------------------------------------------
# Hiring
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"
    ON_HOLD = "on_hold"


class ApplicationStatus(str, Enum):
    APPLIED = "applied"
    SHORTLISTED = "shortlisted"
    INTERVIEWING = "interviewing"
    OFFERED = "offered"
    REJECTED = "rejected"
    PASSED = "passed"


class HiringDecision(str, Enum):
    SHORTLIST = "SHORTLIST"
    REVIEW = "REVIEW"
    PASS = "PASS"


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    department = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=False)
    requirements = Column(Text, nullable=False)       # plain text list of required skills/experience
    required_skills = Column(JSON, default=list)      # structured list for matching
    experience_years = Column(Integer, default=0)
    employment_type = Column(String, default="full_time")  # full_time, part_time, contract
    location = Column(String, nullable=True)
    salary_min = Column(Float, nullable=True)
    salary_max = Column(Float, nullable=True)
    status = Column(String, default=JobStatus.OPEN.value, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    embedding = Column(Vector(1536), nullable=True)   # text-embedding-3-small of title+description+requirements
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    creator = relationship("User", foreign_keys=[created_by])
    applications = relationship("JobApplication", back_populates="job", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)
    cv_filename = Column(String, nullable=True)
    cv_path = Column(String, nullable=True)
    cv_text = Column(Text, nullable=True)             # full extracted text from CV
    cv_embedding = Column(Vector(1536), nullable=True) # text-embedding-3-small of full CV text
    skills = Column(JSON, default=list)               # extracted skills list
    experience_years = Column(Integer, nullable=True)
    education = Column(JSON, default=list)            # extracted education entries
    current_title = Column(String, nullable=True)
    linkedin_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    applications = relationship("JobApplication", back_populates="candidate", cascade="all, delete-orphan")

    @property
    def cv_url(self) -> str:
        if self.cv_filename:
            return f"/uploads/cvs/{self.cv_filename}"
        return None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class JobApplication(Base):
    __tablename__ = "job_applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("job_postings.id"), nullable=False, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False, index=True)
    status = Column(String, default=ApplicationStatus.APPLIED.value, index=True)
    similarity_score = Column(Float, nullable=True)   # cosine similarity 0.0-1.0
    skill_coverage = Column(Float, nullable=True)     # fraction of required skills matched
    rank = Column(Integer, nullable=True)             # rank among all applicants for this job (1 = best)
    llm_decision = Column(String, nullable=True)      # SHORTLIST, REVIEW, PASS
    llm_reasoning = Column(Text, nullable=True)
    interview_date = Column(DateTime, nullable=True)
    interview_notes = Column(Text, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    applied_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    job = relationship("JobPosting", back_populates="applications")
    candidate = relationship("Candidate", back_populates="applications")
    reviewer = relationship("User", foreign_keys=[reviewed_by])


# ---------------------------------------------------------------------------
# Payroll
# ---------------------------------------------------------------------------

class PayCycleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PayslipStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    PAID = "paid"


class PayrollDecision(str, Enum):
    APPROVE = "APPROVE"
    HOLD = "HOLD"
    FLAG = "FLAG"


class PayCycle(Base):
    __tablename__ = "pay_cycles"

    id = Column(Integer, primary_key=True, index=True)
    period_start = Column(Date, nullable=False, index=True)
    period_end = Column(Date, nullable=False, index=True)
    status = Column(String, default=PayCycleStatus.PENDING.value, index=True)
    run_by = Column(Integer, ForeignKey("users.id"), nullable=True)   # HR/admin who triggered
    run_at = Column(DateTime, nullable=True)                           # When the run was triggered
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    runner = relationship("User", foreign_keys=[run_by])
    payslips = relationship("Payslip", back_populates="pay_cycle", cascade="all, delete-orphan")


class Payslip(Base):
    __tablename__ = "payslips"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    pay_cycle_id = Column(Integer, ForeignKey("pay_cycles.id"), nullable=False, index=True)
    gross_pay = Column(Float, nullable=False)
    deductions_leave = Column(Float, default=0.0)   # Unpaid leave deductions
    deductions_tax = Column(Float, default=0.0)     # Tax withheld
    net_pay = Column(Float, nullable=False)
    days_worked = Column(Float, nullable=False)      # Working days in period minus leave
    leave_days_taken = Column(Float, default=0.0)   # Total leave days in period
    llm_decision = Column(String, nullable=True)    # APPROVE, HOLD, FLAG
    llm_reasoning = Column(Text, nullable=True)
    status = Column(String, default=PayslipStatus.DRAFT.value, index=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", foreign_keys=[employee_id])
    pay_cycle = relationship("PayCycle", back_populates="payslips")
    approver = relationship("User", foreign_keys=[approved_by])


# ---------------------------------------------------------------------------
# Policy Compliance (Phase 7)
# ---------------------------------------------------------------------------

class PolicyCategory(str, Enum):
    GENERAL = "general"
    LEAVE = "leave"
    EXPENSE = "expense"
    HIRING = "hiring"
    PAYROLL = "payroll"
    CODE_OF_CONDUCT = "code_of_conduct"
    SAFETY = "safety"
    OTHER = "other"


class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    category    = Column(String, nullable=False)          # PolicyCategory value
    filename    = Column(String, nullable=False)          # UUID filename on disk
    file_path   = Column(String, nullable=False)          # absolute path
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    chunks   = relationship("PolicyChunk", back_populates="document",
                            cascade="all, delete-orphan")
    uploader = relationship("User", foreign_keys=[uploaded_by])


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    id          = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("policy_documents.id", ondelete="CASCADE"),
                         nullable=False)
    content     = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)         # order within document
    embedding   = Column(Vector(1536), nullable=True)     # OpenAI text-embedding-3-small
    token_count = Column(Integer, nullable=True)

    document = relationship("PolicyDocument", back_populates="chunks")