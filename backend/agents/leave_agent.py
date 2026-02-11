"""
Leave Request Agent - LangChain Powered

Validates and processes employee leave requests using GPT-4o-mini for autonomous
decision-making. The LLM analyzes leave balance, conflicts, and business rules
to make approval decisions with natural language reasoning.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import logging
import os
import json

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.base import Agent
from agents.schemas import (
    LeaveRequestInput,
    LeaveValidationOutput,
    ApprovalState
)
from agents.approval_system import approval_manager, EscalationTrigger
from agents.reasoning_system import reasoning_manager, ReasoningStep, ReasoningType, ConfidenceLevel
from agents.state_manager import state_manager
from app.database import AsyncSessionLocal
from app.models import (
    LeaveBalance,
    LeaveRequest,
    PriorityPeriod,
    LeaveRequestStatus,
    LeaveType,
    Employee
)


# Tool Input Schemas
class LeaveBalanceCheckInput(BaseModel):
    """Input schema for leave balance checking tool."""
    user_id: int = Field(description="Employee user ID")
    leave_type: str = Field(description="Type of leave (vacation, sick_leave, personal, maternity, paternity, bereavement)")
    days_requested: float = Field(description="Number of days being requested")


class ConflictDetectionInput(BaseModel):
    """Input schema for conflict detection tool."""
    user_id: int = Field(description="Employee user ID")
    start_date: str = Field(description="Leave start date in YYYY-MM-DD format")
    end_date: str = Field(description="Leave end date in YYYY-MM-DD format")


class BusinessRulesInput(BaseModel):
    """Input schema for business rules validation tool."""
    start_date: str = Field(description="Leave start date in YYYY-MM-DD format")
    end_date: str = Field(description="Leave end date in YYYY-MM-DD format")
    days_requested: int = Field(description="Number of days being requested")


class LeaveAgent(Agent[LeaveRequestInput, LeaveValidationOutput]):
    """
    LangChain-powered agent for leave request validation and approval.

    Uses GPT-4o-mini to analyze leave requests and make autonomous decisions
    about approval, escalation, or rejection. The LLM has access to tools for
    checking balance, detecting conflicts, and validating business rules.
    """

    # Business rules constants
    AUTO_APPROVAL_MAX_DAYS = 3
    MIN_NOTICE_DAYS = 3
    MAX_CONSECUTIVE_DAYS = 14

    def __init__(self):
        super().__init__(
            name="leave_agent",
            description="LangChain-powered agent that validates and processes employee leave requests using GPT-4o-mini for autonomous decision-making"
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # Deterministic for business logic
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define tools that the LLM agent can use
        self.tools = [
            StructuredTool.from_function(
                func=self._tool_check_leave_balance,
                name="check_leave_balance",
                description="Check employee's remaining leave balance for a specific leave type. Returns total days, used days, and remaining days.",
                args_schema=LeaveBalanceCheckInput,
                return_direct=False
            ),
            StructuredTool.from_function(
                func=self._tool_detect_conflicts,
                name="detect_conflicts",
                description="Detect conflicts with existing approved leave requests and company blackout/priority periods. Returns list of conflicts and warnings.",
                args_schema=ConflictDetectionInput,
                return_direct=False
            ),
            StructuredTool.from_function(
                func=self._tool_validate_business_rules,
                name="validate_business_rules",
                description="Validate leave request against company business rules including minimum notice period (3 days) and maximum consecutive days (14 days). Returns violations and warnings.",
                args_schema=BusinessRulesInput,
                return_direct=False
            )
        ]

        # Create agent prompt with clear instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Leave Request Validation Agent for an HR management system.

Your role is to analyze employee leave requests and make autonomous approval decisions based on company policies.

TOOLS AVAILABLE:
1. check_leave_balance - Verify employee has sufficient leave days remaining
2. detect_conflicts - Find overlapping leave requests or blackout periods
3. validate_business_rules - Check notice period and consecutive day limits

DECISION PROCESS:
1. Use ALL available tools to gather complete information
2. Analyze the data considering company policies
3. Make a clear decision: auto-approve, escalate to manager, or reject
4. Provide detailed reasoning for your decision

AUTO-APPROVAL CRITERIA (all must be met):
- Request is 3 days or less
- Employee has sufficient leave balance remaining
- No conflicts with existing approved leave
- No conflicts with company blackout periods
- Meets minimum 3-day advance notice requirement
- Does not exceed 14 consecutive days

ESCALATION TO MANAGER:
- Request is 4-14 days (requires manager approval)
- Has warnings but no hard violations (e.g., priority period overlap)
- Short notice (less than 3 days advance) but otherwise valid

REJECTION:
- Insufficient leave balance (hard block)
- Conflicts with blackout period (hard block)
- Overlaps with existing approved leave (hard block)

OUTPUT FORMAT (use this exact structure):
DECISION: [AUTO_APPROVE | ESCALATE | REJECT]
REASONING: [2-3 sentences explaining your analysis and decision]
CONFIDENCE: [HIGH | MEDIUM | LOW]
FACTORS: [comma-separated list of key factors you considered]
RECOMMENDATIONS: [specific recommendations for manager if escalating, or next steps]

Be thorough, use all tools, and provide clear reasoning for every decision."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create LangChain agent
        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            return_intermediate_steps=True
        )

        self.logger.info("Leave Agent initialized with LangChain + GPT-4o-mini")

    async def execute(self, input_data: LeaveRequestInput) -> LeaveValidationOutput:
        """
        Execute leave request validation using LangChain agent.

        The LLM will autonomously decide whether to auto-approve, escalate,
        or reject the leave request based on balance, conflicts, and business rules.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(f"Processing leave request for user {input_data.user_id}, workflow {workflow_id}")

        try:
            # Parse dates
            start_date = datetime.strptime(input_data.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(input_data.end_date, "%Y-%m-%d").date()
            days_requested = (end_date - start_date).days + 1

            # Basic date validation (before LLM)
            if start_date < date.today():
                return self._create_validation_output(
                    is_valid=False,
                    conflicts=["Cannot request leave for past dates"],
                    auto_approval_eligible=False,
                    reasoning="Leave request rejected: cannot request leave for dates in the past",
                    workflow_id=workflow_id
                )

            if end_date < start_date:
                return self._create_validation_output(
                    is_valid=False,
                    conflicts=["End date must be on or after start date"],
                    auto_approval_eligible=False,
                    reasoning="Leave request rejected: invalid date range",
                    workflow_id=workflow_id
                )

            # Prepare input for LLM agent
            llm_input = f"""
Analyze the following leave request and make an approval decision:

EMPLOYEE ID: {input_data.user_id}
LEAVE TYPE: {input_data.leave_type}
START DATE: {input_data.start_date}
END DATE: {input_data.end_date}
DAYS REQUESTED: {days_requested}
REASON: {input_data.reason or 'Not provided'}

Use the available tools to:
1. Check the employee's leave balance for {input_data.leave_type}
2. Detect any conflicts with existing leave or blackout periods
3. Validate against business rules (notice period, consecutive days)

Then make your decision following the decision criteria in your instructions.
"""

            # Execute LangChain agent
            self.logger.info(f"Invoking LLM agent for leave request validation")
            result = await self.agent_executor.ainvoke({"input": llm_input})

            # Parse LLM decision
            llm_output = result['output']
            intermediate_steps = result.get('intermediate_steps', [])

            self.logger.info(f"LLM decision: {llm_output}")

            # Record LLM reasoning trace
            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered leave request decision",
                    input_data={
                        "user_id": input_data.user_id,
                        "leave_type": input_data.leave_type,
                        "days_requested": days_requested,
                        "start_date": input_data.start_date,
                        "end_date": input_data.end_date
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps)
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=self._parse_factors(llm_output),
                    alternatives_considered=["auto_approve", "escalate", "reject"],
                    agent_name=self.name,
                    workflow_id=workflow_id
                )
            )

            # Parse decision from LLM output
            decision_info = self._parse_llm_decision(llm_output)

            # Create leave request in database
            async with AsyncSessionLocal() as db:
                leave_request = await self._create_leave_request(
                    db,
                    input_data,
                    start_date,
                    end_date,
                    days_requested,
                    decision_info['decision'] == 'AUTO_APPROVE'
                )

                # Handle decision
                if decision_info['decision'] == 'AUTO_APPROVE':
                    # Auto-approve and update balance
                    balance_id = await self._get_balance_id(db, input_data.user_id, input_data.leave_type)
                    await self._auto_approve_leave(db, leave_request, balance_id)
                    await db.commit()

                    return self._create_validation_output(
                        is_valid=True,
                        conflicts=[],
                        auto_approval_eligible=True,
                        reasoning=decision_info['reasoning'],
                        validation_warnings=[],
                        required_approvals=[],
                        recommended_actions=[decision_info['recommendations']],
                        workflow_id=workflow_id,
                        leave_request_id=leave_request.id
                    )

                elif decision_info['decision'] == 'ESCALATE':
                    # Create approval request for manager
                    await self._create_approval_request(
                        leave_request,
                        input_data,
                        days_requested,
                        decision_info.get('conflicts', []),
                        workflow_id,
                        decision_info['reasoning']
                    )
                    await db.commit()

                    return self._create_validation_output(
                        is_valid=True,
                        conflicts=decision_info.get('conflicts', []),
                        auto_approval_eligible=False,
                        reasoning=decision_info['reasoning'],
                        validation_warnings=decision_info.get('warnings', []),
                        required_approvals=["manager"],
                        recommended_actions=[decision_info['recommendations']],
                        workflow_id=workflow_id,
                        leave_request_id=leave_request.id
                    )

                else:  # REJECT
                    leave_request.status = LeaveRequestStatus.REJECTED.value
                    leave_request.rejected_at = datetime.utcnow()
                    leave_request.rejection_reason = decision_info['reasoning']
                    await db.commit()

                    return self._create_validation_output(
                        is_valid=False,
                        conflicts=decision_info.get('conflicts', []),
                        auto_approval_eligible=False,
                        reasoning=decision_info['reasoning'],
                        validation_warnings=[],
                        required_approvals=[],
                        recommended_actions=[decision_info['recommendations']],
                        workflow_id=workflow_id,
                        leave_request_id=leave_request.id
                    )

        except Exception as e:
            self.logger.error(f"Error processing leave request: {e}", exc_info=True)

            # Record error in reasoning
            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="Leave request processing failed",
                    input_data=input_data.dict(),
                    output_data={"error": str(e)},
                    confidence=ConfidenceLevel.VERY_LOW,
                    reasoning=f"System error: {str(e)}",
                    factors=["system_error"],
                    alternatives_considered=[],
                    agent_name=self.name,
                    workflow_id=workflow_id
                )
            )

            return self._create_validation_output(
                is_valid=False,
                conflicts=[f"System error: {str(e)}"],
                auto_approval_eligible=False,
                reasoning=f"Failed to process leave request due to system error: {str(e)}",
                workflow_id=workflow_id
            )

    # ========================================================================
    # LangChain Tool Functions (called by LLM)
    # ========================================================================

    def _tool_check_leave_balance(self, user_id: int, leave_type: str, days_requested: float) -> str:
        """
        Tool function for LLM to check employee leave balance.
        Must be synchronous for LangChain tools.
        """
        import asyncio
        return asyncio.run(self._check_leave_balance_async(user_id, leave_type, days_requested))

    async def _check_leave_balance_async(self, user_id: int, leave_type: str, days_requested: float) -> str:
        """Async implementation of balance check."""
        current_year = date.today().year

        async with AsyncSessionLocal() as db:
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
                # Create default balance
                balance = LeaveBalance(
                    user_id=user_id,
                    leave_type=leave_type,
                    total_days=20.0,
                    used_days=0.0,
                    carried_forward=0.0,
                    year=current_year
                )
                db.add(balance)
                await db.flush()

            remaining_days = balance.remaining_days
            sufficient = remaining_days >= days_requested

            return json.dumps({
                "total_days": balance.total_days,
                "used_days": balance.used_days,
                "carried_forward": balance.carried_forward,
                "remaining_days": remaining_days,
                "days_requested": days_requested,
                "sufficient": sufficient,
                "message": f"Employee has {remaining_days} days remaining (out of {balance.total_days + balance.carried_forward} total). Request is for {days_requested} days. {'Sufficient balance.' if sufficient else 'INSUFFICIENT BALANCE - this is a blocking issue.'}"
            })

    def _tool_detect_conflicts(self, user_id: int, start_date: str, end_date: str) -> str:
        """
        Tool function for LLM to detect leave conflicts.
        Must be synchronous for LangChain tools.
        """
        import asyncio
        return asyncio.run(self._detect_conflicts_async(user_id, start_date, end_date))

    async def _detect_conflicts_async(self, user_id: int, start_date: str, end_date: str) -> str:
        """Async implementation of conflict detection."""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        conflicts = []
        warnings = []

        async with AsyncSessionLocal() as db:
            # Check overlapping leave
            overlap_result = await db.execute(
                select(LeaveRequest).where(
                    and_(
                        LeaveRequest.user_id == user_id,
                        LeaveRequest.status.in_([
                            LeaveRequestStatus.SUBMITTED.value,
                            LeaveRequestStatus.APPROVED.value
                        ]),
                        or_(
                            and_(LeaveRequest.start_date <= start, LeaveRequest.end_date >= start),
                            and_(LeaveRequest.start_date <= end, LeaveRequest.end_date >= end),
                            and_(LeaveRequest.start_date >= start, LeaveRequest.end_date <= end)
                        )
                    )
                )
            )
            overlapping_leaves = overlap_result.scalars().all()

            for leave in overlapping_leaves:
                conflicts.append(f"Overlaps with existing {leave.leave_type} leave from {leave.start_date} to {leave.end_date}")

            # Check priority/blackout periods
            priority_result = await db.execute(
                select(PriorityPeriod).where(
                    or_(
                        and_(PriorityPeriod.start_date <= start, PriorityPeriod.end_date >= start),
                        and_(PriorityPeriod.start_date <= end, PriorityPeriod.end_date >= end),
                        and_(PriorityPeriod.start_date >= start, PriorityPeriod.end_date <= end)
                    )
                )
            )
            priority_periods = priority_result.scalars().all()

            for period in priority_periods:
                if period.is_blackout:
                    conflicts.append(f"BLACKOUT PERIOD: '{period.name}' ({period.start_date} to {period.end_date}) - leave cannot be approved during blackout")
                else:
                    warnings.append(f"Priority period: '{period.name}' ({period.start_date} to {period.end_date}) - may require additional scrutiny")

        return json.dumps({
            "conflicts": conflicts,
            "warnings": warnings,
            "has_blocking_conflicts": len(conflicts) > 0,
            "message": f"Found {len(conflicts)} blocking conflicts and {len(warnings)} warnings. " +
                      (f"CONFLICTS: {', '.join(conflicts)}" if conflicts else "No blocking conflicts.") +
                      (f" WARNINGS: {', '.join(warnings)}" if warnings else "")
        })

    def _tool_validate_business_rules(self, start_date: str, end_date: str, days_requested: int) -> str:
        """
        Tool function for LLM to validate business rules.
        Synchronous (no DB access needed).
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        violations = []
        warnings = []

        # Check notice period
        days_until_leave = (start - date.today()).days
        if days_until_leave < 0:
            violations.append("Cannot request leave for past dates")
        elif days_until_leave < self.MIN_NOTICE_DAYS:
            warnings.append(f"Short notice: only {days_until_leave} days advance notice (recommended: {self.MIN_NOTICE_DAYS} days)")

        # Check consecutive days
        if days_requested > self.MAX_CONSECUTIVE_DAYS:
            warnings.append(f"Long leave period: {days_requested} days exceeds recommended maximum of {self.MAX_CONSECUTIVE_DAYS} days")

        return json.dumps({
            "violations": violations,
            "warnings": warnings,
            "days_until_leave": days_until_leave,
            "meets_notice_requirement": days_until_leave >= self.MIN_NOTICE_DAYS,
            "within_consecutive_limit": days_requested <= self.MAX_CONSECUTIVE_DAYS,
            "message": f"Business rules check: {len(violations)} violations, {len(warnings)} warnings. " +
                      f"Notice period: {days_until_leave} days. Consecutive days: {days_requested}."
        })

    # ========================================================================
    # Helper Functions
    # ========================================================================

    def _parse_llm_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse structured decision from LLM output."""
        decision_info = {
            'decision': 'ESCALATE',  # Default to escalate if parsing fails
            'reasoning': llm_output,
            'recommendations': 'Review the request manually',
            'conflicts': [],
            'warnings': []
        }

        # Parse decision
        if 'DECISION:' in llm_output:
            decision_line = [line for line in llm_output.split('\n') if 'DECISION:' in line][0]
            if 'AUTO_APPROVE' in decision_line:
                decision_info['decision'] = 'AUTO_APPROVE'
            elif 'REJECT' in decision_line:
                decision_info['decision'] = 'REJECT'
            elif 'ESCALATE' in decision_line:
                decision_info['decision'] = 'ESCALATE'

        # Parse reasoning
        if 'REASONING:' in llm_output:
            reasoning_lines = []
            capture = False
            for line in llm_output.split('\n'):
                if 'REASONING:' in line:
                    capture = True
                    reasoning_lines.append(line.replace('REASONING:', '').strip())
                elif capture and line.strip() and not any(keyword in line for keyword in ['CONFIDENCE:', 'FACTORS:', 'RECOMMENDATIONS:', 'DECISION:']):
                    reasoning_lines.append(line.strip())
                elif capture and any(keyword in line for keyword in ['CONFIDENCE:', 'FACTORS:', 'RECOMMENDATIONS:']):
                    break
            if reasoning_lines:
                decision_info['reasoning'] = ' '.join(reasoning_lines)

        # Parse recommendations
        if 'RECOMMENDATIONS:' in llm_output:
            rec_lines = []
            capture = False
            for line in llm_output.split('\n'):
                if 'RECOMMENDATIONS:' in line:
                    capture = True
                    rec_lines.append(line.replace('RECOMMENDATIONS:', '').strip())
                elif capture and line.strip() and not any(keyword in line for keyword in ['CONFIDENCE:', 'FACTORS:', 'REASONING:', 'DECISION:']):
                    rec_lines.append(line.strip())
                elif capture and any(keyword in line for keyword in ['CONFIDENCE:', 'FACTORS:', 'REASONING:']):
                    break
            if rec_lines:
                decision_info['recommendations'] = ' '.join(rec_lines)

        return decision_info

    def _parse_confidence(self, llm_output: str) -> ConfidenceLevel:
        """Parse confidence level from LLM output."""
        if 'CONFIDENCE: HIGH' in llm_output:
            return ConfidenceLevel.HIGH
        elif 'CONFIDENCE: MEDIUM' in llm_output:
            return ConfidenceLevel.MEDIUM
        elif 'CONFIDENCE: LOW' in llm_output:
            return ConfidenceLevel.LOW
        elif 'CONFIDENCE: VERY_HIGH' in llm_output:
            return ConfidenceLevel.VERY_HIGH
        else:
            return ConfidenceLevel.MEDIUM  # Default

    def _parse_factors(self, llm_output: str) -> List[str]:
        """Parse factors from LLM output."""
        if 'FACTORS:' in llm_output:
            factors_line = [line for line in llm_output.split('\n') if 'FACTORS:' in line]
            if factors_line:
                factors_text = factors_line[0].replace('FACTORS:', '').strip()
                return [f.strip() for f in factors_text.split(',')]
        return ["leave_balance", "conflicts", "business_rules"]  # Default

    async def _get_balance_id(self, db: AsyncSession, user_id: int, leave_type: str) -> int:
        """Get balance ID for updating after approval."""
        current_year = date.today().year
        result = await db.execute(
            select(LeaveBalance).where(
                and_(
                    LeaveBalance.user_id == user_id,
                    LeaveBalance.leave_type == leave_type,
                    LeaveBalance.year == current_year
                )
            )
        )
        balance = result.scalar_one()
        return balance.id

    async def _create_leave_request(
        self,
        db: AsyncSession,
        input_data: LeaveRequestInput,
        start_date: date,
        end_date: date,
        days_requested: float,
        auto_approved: bool
    ) -> LeaveRequest:
        """Create leave request record in database."""
        leave_request = LeaveRequest(
            user_id=input_data.user_id,
            leave_type=input_data.leave_type,
            start_date=start_date,
            end_date=end_date,
            days_requested=days_requested,
            status=LeaveRequestStatus.APPROVED.value if auto_approved else LeaveRequestStatus.SUBMITTED.value,
            reason=input_data.reason,
            submitted_at=datetime.utcnow(),
            approved_at=datetime.utcnow() if auto_approved else None,
            approved_by=None  # System auto-approval
        )

        db.add(leave_request)
        await db.flush()
        return leave_request

    async def _auto_approve_leave(
        self,
        db: AsyncSession,
        leave_request: LeaveRequest,
        balance_id: int
    ) -> None:
        """Auto-approve leave and update balance."""
        leave_request.status = LeaveRequestStatus.APPROVED.value
        leave_request.approved_at = datetime.utcnow()

        # Update balance
        balance_result = await db.execute(
            select(LeaveBalance).where(LeaveBalance.id == balance_id)
        )
        balance = balance_result.scalar_one()
        balance.used_days += leave_request.days_requested

        self.logger.info(f"Auto-approved leave request {leave_request.id} for user {leave_request.user_id}")

    async def _create_approval_request(
        self,
        leave_request: LeaveRequest,
        input_data: LeaveRequestInput,
        days_requested: int,
        conflicts: List[str],
        workflow_id: str,
        llm_reasoning: str
    ) -> Any:
        """Create approval request for manager review."""
        triggers = []

        if conflicts:
            triggers.append(EscalationTrigger.POLICY_VIOLATION)
        if days_requested > 7:
            triggers.append(EscalationTrigger.HIGH_VALUE)

        approval_request = await approval_manager.create_approval_request(
            workflow_id=workflow_id,
            agent_name=self.name,
            request_type="leave_request",
            requester_id=input_data.user_id,
            title=f"Leave Request - {input_data.leave_type.replace('_', ' ').title()} ({days_requested} days)",
            description=f"Employee requests {days_requested} days of {input_data.leave_type} from {input_data.start_date} to {input_data.end_date}.\n\nReason: {input_data.reason or 'Not provided'}\n\nLLM Analysis:\n{llm_reasoning}",
            data={
                "leave_request_id": leave_request.id,
                "user_id": input_data.user_id,
                "leave_type": input_data.leave_type,
                "start_date": input_data.start_date,
                "end_date": input_data.end_date,
                "days_requested": days_requested,
                "reason": input_data.reason,
                "conflicts": conflicts,
                "llm_reasoning": llm_reasoning
            },
            triggers=triggers,
            priority="high" if days_requested > 5 else "normal"
        )

        self.logger.info(f"Created approval request {approval_request.request_id} for leave request {leave_request.id}")
        return approval_request

    def _create_validation_output(
        self,
        is_valid: bool,
        conflicts: List[str],
        auto_approval_eligible: bool,
        reasoning: str,
        validation_warnings: Optional[List[str]] = None,
        required_approvals: Optional[List[str]] = None,
        recommended_actions: Optional[List[str]] = None,
        workflow_id: Optional[str] = None,
        leave_request_id: Optional[int] = None
    ) -> LeaveValidationOutput:
        """Create standardized validation output."""
        return LeaveValidationOutput(
            success=is_valid,
            message="Leave request validated successfully" if is_valid else "Leave request validation failed",
            reasoning=reasoning,
            is_valid=is_valid,
            conflicts=conflicts,
            auto_approval_eligible=auto_approval_eligible,
            required_approvals=required_approvals or [],
            validation_warnings=validation_warnings or [],
            recommended_actions=recommended_actions or []
        )

    def get_input_schema(self) -> type[LeaveRequestInput]:
        return LeaveRequestInput

    def get_output_schema(self) -> type[LeaveValidationOutput]:
        return LeaveValidationOutput
