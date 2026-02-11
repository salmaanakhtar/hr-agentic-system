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


class ExpenseClaimInput(BaseModel):
    user_id: int
    amount: float
    category: str
    description: str
    date: str
    receipts: List[str] = Field(default_factory=list)


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
    decision: ApprovalState
    validated_amount: Optional[float] = None
    policy_violations: List[str] = Field(default_factory=list)


class CandidateRankingOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    top_candidate: Optional[Dict[str, Any]] = None
    ranking_criteria: List[str] = Field(default_factory=list)