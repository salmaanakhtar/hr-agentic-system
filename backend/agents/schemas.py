from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"

class BasicAgentInput(BaseModel):
    user_id: int
    workflow_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class BasicAgentOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowAgentInput(BaseModel):
    user_id: int
    workflow_id: Optional[str] = None
    step_data: Dict[str, Any] = Field(default_factory=dict)
    previous_steps: List[Dict[str, Any]] = Field(default_factory=list)


class QueryAgentInput(BaseModel):
    user_id: int
    query: str
    context: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None


class QueryAgentOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)



class DecisionAgentOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)



class MultiStepAgentOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    current_step: str
    next_steps: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    workflow_data: Dict[str, Any] = Field(default_factory=dict)


class LeaveRequestInput(BaseModel):
    user_id: int
    leave_type: str
    start_date: str
    end_date: str
    reason: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)


class ExpenseSubmitInput(BaseModel):
    user_id: int
    amount: float
    category: str  # meals, travel, equipment, entertainment, office_supplies, other
    vendor: Optional[str] = None
    date: str       # ISO date string YYYY-MM-DD
    description: Optional[str] = None
    receipt_filename: Optional[str] = None  # set by API after file saved
    receipt_path: Optional[str] = None      # set by API after file saved


# Keep old name as alias for backwards compatibility with orchestrator schemas
ExpenseClaimInput = ExpenseSubmitInput


class OCRExtractionResult(BaseModel):
    raw_text: str = ""
    vendor: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    vendor_confidence: float = 0.0   # 0.0-1.0
    amount_confidence: float = 0.0
    date_confidence: float = 0.0
    overall_confidence: float = 0.0


class HiringRequestInput(BaseModel):
    user_id: int
    position: str
    department: str
    requirements: List[str] = Field(default_factory=list)
    priority: str = "normal"


class PayrollQueryInput(BaseModel):
    user_id: int
    query_type: str
    period: Optional[str] = None


class LeaveDecisionOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: ApprovalState
    approved_days: Optional[int] = None
    comments: Optional[str] = None


class LeaveValidationOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    is_valid: bool
    conflicts: List[str] = Field(default_factory=list)
    auto_approval_eligible: bool
    required_approvals: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class LeaveApprovalOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: ApprovalState
    approved_days: Optional[float] = None
    rejection_reason: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)
    policy_references: List[str] = Field(default_factory=list)


class ExpenseValidationOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: str   # AUTO_APPROVE, ESCALATE, REJECT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    expense_id: Optional[int] = None
    validated_amount: Optional[float] = None
    ocr_result: Optional[OCRExtractionResult] = None
    ocr_match: bool = True              # True if OCR data matches submitted data
    policy_violations: List[str] = Field(default_factory=list)
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ExpenseApprovalOutput(BaseModel):
    success: bool
    message: str
    decision: str   # approved, rejected
    expense_id: int
    comments: Optional[str] = None
    rejection_reason: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)


class HiringApplicationInput(BaseModel):
    application_id: int
    job_id: int
    candidate_id: int
    submitted_by: int  # user_id of the HR/manager who triggered evaluation


class HiringApplicationOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: str   # SHORTLIST, REVIEW, PASS
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    application_id: int
    similarity_score: Optional[float] = None
    skill_coverage: Optional[float] = None
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CandidateRankingOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    top_candidate: Optional[Dict[str, Any]] = None
    ranking_criteria: List[str] = Field(default_factory=list)


class CVParseResult(BaseModel):
    raw_text: str = ""
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = None
    education: List[Dict[str, Any]] = Field(default_factory=list)
    current_title: Optional[str] = None
    parse_confidence: float = 0.0   # 0.0-1.0, fraction of structured fields extracted


# ---------------------------------------------------------------------------
# Hiring API Request/Response Schemas (Phase 5.4)
# ---------------------------------------------------------------------------

class JobPostingCreate(BaseModel):
    title: str
    department: str
    description: str
    requirements: str
    required_skills: List[str] = Field(default_factory=list)
    experience_years: int = 0
    employment_type: str = "full_time"
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None


class JobPostingUpdate(BaseModel):
    title: Optional[str] = None
    department: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    required_skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    status: Optional[str] = None


class ApplicationCreate(BaseModel):
    job_id: int
    candidate_id: int


class ApplicationStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


class InterviewScheduleCreate(BaseModel):
    interview_date: str   # ISO 8601 datetime string
    interview_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Payroll Schemas (Phase 6.2)
# ---------------------------------------------------------------------------

class PayrollRunInput(BaseModel):
    employee_id: int
    pay_cycle_id: int
    period_start: str   # ISO date YYYY-MM-DD
    period_end: str     # ISO date YYYY-MM-DD
    run_by_user_id: int


class PayrollRunOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    decision: str       # APPROVE, HOLD, FLAG
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    payslip_id: Optional[int] = None
    employee_id: int
    gross_pay: Optional[float] = None
    deductions_leave: Optional[float] = None
    deductions_tax: Optional[float] = None
    net_pay: Optional[float] = None
    days_worked: Optional[float] = None
    leave_days_taken: Optional[float] = None
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Payroll API Request Schemas (Phase 6.3)
# ---------------------------------------------------------------------------

class PayCycleCreate(BaseModel):
    period_start: str   # ISO date YYYY-MM-DD
    period_end: str     # ISO date YYYY-MM-DD


class PayslipApprove(BaseModel):
    notes: Optional[str] = None