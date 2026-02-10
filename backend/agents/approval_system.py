from typing import Any, Dict, List, Optional, Set
import logging
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass

from .schemas import ApprovalState
from .state import WorkflowState, WorkflowStatus
from .state_manager import state_manager


class EscalationLevel(str, Enum):

    NONE = "none"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    HR = "hr"
    EXECUTIVE = "executive"


class EscalationTrigger(str, Enum):

    HIGH_VALUE = "high_value"
    POLICY_VIOLATION = "policy_violation"
    UNUSUAL_PATTERN = "unusual_pattern"
    CONFLICT_OF_INTEREST = "conflict_of_interest"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    EXCEPTION_CASE = "exception_case"
    TIMEOUT = "timeout"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class ApprovalRequest:

    request_id: str
    workflow_id: str
    agent_name: str
    request_type: str
    requester_id: int
    approver_id: Optional[int]
    escalation_level: EscalationLevel
    status: ApprovalState
    priority: str
    title: str
    description: str
    data: Dict[str, Any]
    triggers: List[EscalationTrigger]
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime]
    decision: Optional[ApprovalState]
    decision_reason: Optional[str]
    decided_at: Optional[datetime]
    decided_by: Optional[int]
    metadata: Dict[str, Any]


@dataclass
class EscalationRule:

    rule_id: str
    name: str
    description: str
    trigger_conditions: List[EscalationTrigger]
    escalation_level: EscalationLevel
    priority_threshold: Optional[str] = None
    value_threshold: Optional[float] = None
    time_limit_hours: Optional[int] = None
    auto_escalate: bool = False
    notify_stakeholders: bool = True


class ApprovalQueueManager:

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._approval_requests: Dict[str, ApprovalRequest] = {}
        self._escalation_rules: List[EscalationRule] = []

        self._initialize_default_rules()

    def _initialize_default_rules(self):

        self._escalation_rules = [
            EscalationRule(
                rule_id="high_value_expense",
                name="High Value Expense",
                description="Expenses over $500 require manager approval",
                trigger_conditions=[EscalationTrigger.HIGH_VALUE],
                escalation_level=EscalationLevel.MANAGER,
                value_threshold=500.0,
                time_limit_hours=24,
                auto_escalate=True
            ),
            EscalationRule(
                rule_id="policy_violation",
                name="Policy Violation",
                description="Policy violations require HR review",
                trigger_conditions=[EscalationTrigger.POLICY_VIOLATION],
                escalation_level=EscalationLevel.HR,
                time_limit_hours=48,
                auto_escalate=True
            ),
            EscalationRule(
                rule_id="regulatory_requirement",
                name="Regulatory Requirement",
                description="Regulatory requirements need executive approval",
                trigger_conditions=[EscalationTrigger.REGULATORY_REQUIREMENT],
                escalation_level=EscalationLevel.EXECUTIVE,
                time_limit_hours=72,
                auto_escalate=False
            ),
            EscalationRule(
                rule_id="timeout_escalation",
                name="Timeout Escalation",
                description="Pending approvals escalate after timeout",
                trigger_conditions=[EscalationTrigger.TIMEOUT],
                escalation_level=EscalationLevel.MANAGER,
                time_limit_hours=24,
                auto_escalate=True
            )
        ]

    async def create_approval_request(self,
                                    workflow_id: str,
                                    agent_name: str,
                                    request_type: str,
                                    requester_id: int,
                                    title: str,
                                    description: str,
                                    data: Dict[str, Any],
                                    triggers: List[EscalationTrigger],
                                    priority: str = "normal") -> ApprovalRequest:
        import uuid

        request_id = f"apr_{uuid.uuid4().hex[:8]}"

        escalation_level = self._determine_escalation_level(triggers, data, priority)


        due_date = self._calculate_due_date(escalation_level, priority)

        approval_request = ApprovalRequest(
            request_id=request_id,
            workflow_id=workflow_id,
            agent_name=agent_name,
            request_type=request_type,
            requester_id=requester_id,
            approver_id=None,
            escalation_level=escalation_level,
            status=ApprovalState.PENDING,
            priority=priority,
            title=title,
            description=description,
            data=data,
            triggers=triggers,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            due_date=due_date,
            decision=None,
            decision_reason=None,
            decided_at=None,
            decided_by=None,
            metadata={
                "created_by_agent": agent_name,
                "escalation_rules_applied": [rule.rule_id for rule in self._escalation_rules
                                           if any(t in rule.trigger_conditions for t in triggers)]
            }
        )
        self._approval_requests[request_id] = approval_request

  
        await self._assign_approver(approval_request)

        await self._update_workflow_state(workflow_id, approval_request)

        self.logger.info(f"Created approval request {request_id} for workflow {workflow_id}")
        return approval_request

    def _determine_escalation_level(self, triggers: List[EscalationTrigger],
                                  data: Dict[str, Any], priority: str) -> EscalationLevel:

        level = EscalationLevel.NONE


        for rule in self._escalation_rules:
            if any(trigger in rule.trigger_conditions for trigger in triggers):

                if rule.value_threshold and "amount" in data:
                    try:
                        amount = float(data["amount"])
                        if amount >= rule.value_threshold:
                            level = max(level, rule.escalation_level, key=lambda x: ["none", "supervisor", "manager", "hr", "executive"].index(x.value))
                    except (ValueError, TypeError):
                        pass

                elif rule.priority_threshold and priority == rule.priority_threshold:
                    level = max(level, rule.escalation_level, key=lambda x: ["none", "supervisor", "manager", "hr", "executive"].index(x.value))

                else:
                    
                    level = max(level, rule.escalation_level, key=lambda x: ["none", "supervisor", "manager", "hr", "executive"].index(x.value))

        return level

    def _calculate_due_date(self, escalation_level: EscalationLevel, priority: str) -> Optional[datetime]:
        base_hours = {
            EscalationLevel.NONE: 24,
            EscalationLevel.SUPERVISOR: 12,
            EscalationLevel.MANAGER: 24,
            EscalationLevel.HR: 48,
            EscalationLevel.EXECUTIVE: 72
        }

        hours = base_hours.get(escalation_level, 24)


        if priority == "urgent":
            hours = max(1, hours // 2)
        elif priority == "low":
            hours *= 2

        return datetime.now() + timedelta(hours=hours)

    async def _assign_approver(self, approval_request: ApprovalRequest) -> None:


        escalation_assignments = {
            EscalationLevel.SUPERVISOR: 2,
            EscalationLevel.MANAGER: 3,
            EscalationLevel.HR: 4,
            EscalationLevel.EXECUTIVE: 5,
        }

        if approval_request.escalation_level in escalation_assignments:
            approval_request.approver_id = escalation_assignments[approval_request.escalation_level]

 
        approval_request.updated_at = datetime.now()

    async def _update_workflow_state(self, workflow_id: str, approval_request: ApprovalRequest) -> None:
        
        try:
            workflow_state = await state_manager.get_workflow_state(workflow_id)
            if workflow_state:
                workflow_state.set("approval_request", {
                    "request_id": approval_request.request_id,
                    "status": approval_request.status.value,
                    "escalation_level": approval_request.escalation_level.value,
                    "approver_id": approval_request.approver_id,
                    "due_date": approval_request.due_date.isoformat() if approval_request.due_date else None
                })

                workflow_state.set("requires_approval", True)
                workflow_state.status = WorkflowStatus.PAUSED

                await state_manager.update_workflow_state(workflow_state)

        except ValueError:
            self.logger.warning(f"Could not find workflow {workflow_id} to update with approval info")

    async def approve_request(self, request_id: str, approver_id: int,
                            decision: ApprovalState, reason: str = "") -> bool:
        
        if request_id not in self._approval_requests:
            self.logger.error(f"Approval request {request_id} not found")
            return False

        approval_request = self._approval_requests[request_id]


        approval_request.status = decision
        approval_request.decision = decision
        approval_request.decision_reason = reason
        approval_request.decided_at = datetime.now()
        approval_request.decided_by = approver_id
        approval_request.updated_at = datetime.now()

        await self._update_workflow_after_decision(approval_request)

        self.logger.info(f"Approval request {request_id} {decision.value} by user {approver_id}")
        return True

    async def _update_workflow_after_decision(self, approval_request: ApprovalRequest) -> None:
        
        try:
            workflow_state = await state_manager.get_workflow_state(approval_request.workflow_id)

            if workflow_state:

                workflow_state.set("approval_decision", {
                    "decision": approval_request.decision.value,
                    "reason": approval_request.decision_reason,
                    "decided_by": approval_request.decided_by,
                    "decided_at": approval_request.decided_at.isoformat()
                })


                if approval_request.decision == ApprovalState.APPROVED:
                    workflow_state.status = WorkflowStatus.RUNNING
                    workflow_state.set("approval_granted", True)
                elif approval_request.decision == ApprovalState.REJECTED:
                    workflow_state.status = WorkflowStatus.FAILED
                    workflow_state.set("approval_denied", True)
                    workflow_state.set("failure_reason", approval_request.decision_reason)
                else:

                    workflow_state.status = WorkflowStatus.PAUSED

                await state_manager.update_workflow_state(workflow_state)

        except ValueError:
            self.logger.error(f"Could not find workflow {approval_request.workflow_id} for approval update")

    async def check_timeouts(self) -> List[str]:
        
        now = datetime.now()
        escalated = []

        for request_id, request in self._approval_requests.items():
            if (request.status == ApprovalState.PENDING and
                request.due_date and now > request.due_date):

  
                await self._escalate_request(request_id, EscalationTrigger.TIMEOUT)
                escalated.append(request_id)

        return escalated

    async def _escalate_request(self, request_id: str, trigger: EscalationTrigger) -> None:
        
        if request_id not in self._approval_requests:
            return

        request = self._approval_requests[request_id]


        if trigger not in request.triggers:
            request.triggers.append(trigger)

   
        new_level = self._determine_escalation_level(request.triggers, request.data, request.priority)

        if new_level != request.escalation_level:
            old_level = request.escalation_level
            request.escalation_level = new_level
            request.updated_at = datetime.now()

            await self._assign_approver(request)

            self.logger.info(f"Escalated request {request_id} from {old_level.value} to {new_level.value}")

    def get_pending_approvals(self, user_id: Optional[int] = None) -> List[ApprovalRequest]:
        
        pending = [
            req for req in self._approval_requests.values()
            if req.status == ApprovalState.PENDING
        ]

        if user_id:
            pending = [req for req in pending if req.approver_id == user_id]

        return pending

    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
       
        return self._approval_requests.get(request_id)

    def get_approval_stats(self) -> Dict[str, Any]:
        
        total = len(self._approval_requests)
        pending = len([r for r in self._approval_requests.values() if r.status == ApprovalState.PENDING])
        approved = len([r for r in self._approval_requests.values() if r.status == ApprovalState.APPROVED])
        rejected = len([r for r in self._approval_requests.values() if r.status == ApprovalState.REJECTED])
        escalated = len([r for r in self._approval_requests.values() if r.status == ApprovalState.ESCALATED])

        return {
            "total_requests": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "escalated": escalated,
            "approval_rate": (approved / total * 100) if total > 0 else 0
        }



approval_manager = ApprovalQueueManager()