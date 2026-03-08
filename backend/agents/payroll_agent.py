"""
Payroll Agent - LangChain Powered

Calculates monthly payslips for employees using:
- Employee salary and tax rate from employee record
- Approved leave data from leave requests
- Unpaid leave deduction calculation (leave beyond balance)
- Flat-rate tax calculation
- APPROVE / HOLD / FLAG decisions with reasoning traces

Decision criteria:
  APPROVE : All data present, net_pay > 0, no anomalies
  HOLD    : Missing salary data or critical calculation error
  FLAG    : Net pay variance > 20% vs last period, or net_pay <= 0
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from sqlalchemy import select, and_
import uuid
import logging
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel as BaseModelV1, Field

from agents.base import Agent
from agents.schemas import PayrollRunInput, PayrollRunOutput
from agents.reasoning_system import reasoning_manager, ReasoningStep, ReasoningType, ConfidenceLevel
from app.database import AsyncSessionLocal
from app.models import (
    Employee, LeaveRequest, LeaveBalance, LeaveRequestStatus,
    Payslip, PayslipStatus,
)


# ---------------------------------------------------------------------------
# Tool Input Schemas (Pydantic V1 required for LangChain 0.1.x)
# ---------------------------------------------------------------------------

class GetEmployeePayrollInfoInput(BaseModelV1):
    employee_id: int = Field(description="ID of the employee to retrieve payroll information for")


class GetApprovedLeaveInput(BaseModelV1):
    employee_id: int = Field(description="ID of the employee whose approved leave to query")
    period_start: str = Field(description="Pay period start date in YYYY-MM-DD format")
    period_end: str = Field(description="Pay period end date in YYYY-MM-DD format")


class ApplyLeaveDeductionsInput(BaseModelV1):
    employee_id: int = Field(description="ID of the employee")
    monthly_gross: float = Field(description="Full monthly gross pay (base_salary / 12) before any deductions")
    period_start: str = Field(description="Pay period start date in YYYY-MM-DD format")
    period_end: str = Field(description="Pay period end date in YYYY-MM-DD format")


class CalculateTaxInput(BaseModelV1):
    employee_id: int = Field(description="ID of the employee")
    taxable_amount: float = Field(description="Taxable pay amount — gross_pay minus unpaid leave deductions")


class GeneratePayslipInput(BaseModelV1):
    employee_id: int = Field(description="ID of the employee")
    pay_cycle_id: int = Field(description="ID of the pay cycle this payslip belongs to")
    gross_pay: float = Field(description="Full monthly gross pay before deductions")
    deductions_leave: float = Field(description="Unpaid leave deduction amount in currency units")
    deductions_tax: float = Field(description="Tax deduction amount in currency units")
    days_worked: float = Field(description="Working days in the period minus total leave days taken")
    leave_days_taken: float = Field(description="Total approved leave days falling within the period")


# ---------------------------------------------------------------------------
# Payroll Agent
# ---------------------------------------------------------------------------

class PayrollAgent(Agent[PayrollRunInput, PayrollRunOutput]):
    """
    LangChain-powered agent for calculating employee payslips.

    Uses GPT-4o-mini to orchestrate pay calculations: gross pay, leave
    deductions (unpaid portion only), flat-rate tax, and APPROVE/HOLD/FLAG
    decisions with full reasoning traces.
    """

    def __init__(self):
        super().__init__(
            name="payroll_agent",
            description=(
                "LangChain-powered agent that calculates monthly payslips using "
                "salary data, approved leave records, and flat-rate tax, producing "
                "APPROVE/HOLD/FLAG decisions with reasoning traces"
            )
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.tools = [
            StructuredTool.from_function(
                coroutine=self._tool_get_employee_payroll_info,
                name="get_employee_payroll_info",
                description=(
                    "Retrieve the employee's salary and payroll configuration: "
                    "base_salary (annual), pay_frequency, and flat tax_rate. "
                    "Returns employee name, department, monthly_gross = base_salary / 12. "
                    "ALWAYS call this first — if salary is missing, use HOLD decision."
                ),
                args_schema=GetEmployeePayrollInfoInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_get_approved_leave,
                name="get_approved_leave",
                description=(
                    "Query all approved leave requests overlapping the pay period for this employee. "
                    "Returns total leave days, breakdown by leave type, remaining leave balances, "
                    "and computed days_worked (working_days - leave_days). "
                    "Call this after get_employee_payroll_info."
                ),
                args_schema=GetApprovedLeaveInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_apply_leave_deductions,
                name="apply_leave_deductions",
                description=(
                    "Calculate the unpaid leave deduction amount. "
                    "Only leave days that exceed the employee's remaining balance are unpaid. "
                    "Returns unpaid_days and deduction_amount in currency units. "
                    "Call this after get_approved_leave."
                ),
                args_schema=ApplyLeaveDeductionsInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_calculate_tax,
                name="calculate_tax",
                description=(
                    "Calculate the flat-rate tax deduction using the employee's stored tax_rate. "
                    "taxable_amount = gross_pay - deductions_leave. "
                    "Returns tax_amount and net_after_tax. "
                    "Call this after apply_leave_deductions."
                ),
                args_schema=CalculateTaxInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_generate_payslip,
                name="generate_payslip",
                description=(
                    "Assemble and save the final payslip to the database. "
                    "Provide all computed values: gross_pay, deductions_leave, deductions_tax, "
                    "days_worked, leave_days_taken. "
                    "Returns payslip_id, net_pay, and any variance vs the last pay period. "
                    "Call this last — a FLAG warning will appear if variance exceeds 20% or net_pay <= 0."
                ),
                args_schema=GeneratePayslipInput,
                return_direct=False,
            ),
        ]

        self.system_prompt = """You are a Payroll Agent for an HR management system.

Your role is to calculate the monthly payslip for an employee using their salary
data, approved leave records, and flat-rate tax configuration.

TOOLS AVAILABLE:
1. get_employee_payroll_info  - Fetch salary, tax rate, and employee details
2. get_approved_leave         - Query approved leave taken within the pay period
3. apply_leave_deductions     - Compute unpaid leave deductions (leave beyond balance only)
4. calculate_tax              - Apply flat tax rate to taxable pay
5. generate_payslip           - Assemble and save the final payslip to the database

CALCULATION PROCESS:
1. Call get_employee_payroll_info to confirm salary data and get monthly_gross (base_salary / 12)
2. Call get_approved_leave to find all approved leave within the period
3. Call apply_leave_deductions with monthly_gross to compute unpaid deduction amount
4. Call calculate_tax with taxable_amount = gross_pay - deductions_leave
5. Call generate_payslip with all computed values to save the payslip

KEY RULES:
- gross_pay = base_salary / 12  (full monthly amount, paid leave does not reduce this)
- deductions_leave = unpaid portion only (days exceeding remaining balance * daily_rate)
- deductions_tax = taxable_amount * tax_rate (flat rate)
- net_pay = gross_pay - deductions_leave - deductions_tax
- days_worked = working_days_in_period - total_leave_days (informational field)

DECISION CRITERIA:
- APPROVE : All data present, net_pay > 0, variance within 20% of last period
- HOLD    : Missing or zero base_salary, employee not found, calculation error
- FLAG    : net_pay <= 0, or variance > 20% vs last pay period

OUTPUT FORMAT (use this exact structure):
DECISION: [APPROVE | HOLD | FLAG]
REASONING: [2-3 sentences explaining the payroll calculation and any notable observations]
CONFIDENCE: [HIGH | MEDIUM | LOW]
FACTORS: [comma-separated list, e.g. salary_verified, leave_deducted, tax_applied]
RECOMMENDATIONS: [any follow-up actions required, e.g. manager review, data correction]

Be systematic, call all tools in order, and document the complete calculation."""

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.logger.info("Payroll Agent initialized with LangChain + GPT-4o-mini")

    # -------------------------------------------------------------------------
    # Agent Loop
    # -------------------------------------------------------------------------

    async def _run_agent_loop(self, user_input: str, max_iterations: int = 7) -> tuple[str, list]:
        """Run the LangChain tool-calling agent loop."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        intermediate_steps = []

        for _ in range(max_iterations):
            response = await self.llm_with_tools.ainvoke(messages)

            if not response.tool_calls:
                return response.content, intermediate_steps

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                self.logger.info(f"LLM calling tool: {tool_name} with args: {tool_args}")

                try:
                    tool = self.tool_map[tool_name]
                    tool_result = await tool.ainvoke(tool_args)
                    intermediate_steps.append((tool_name, tool_args, tool_result))
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call["id"],
                    })
                except Exception as e:
                    self.logger.error(f"Tool execution error ({tool_name}): {e}")
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}),
                        "tool_call_id": tool_call["id"],
                    })

        return "ERROR: Max iterations reached without final decision", intermediate_steps

    # -------------------------------------------------------------------------
    # Main Execute
    # -------------------------------------------------------------------------

    async def execute(self, input_data: PayrollRunInput) -> PayrollRunOutput:
        """
        Execute payroll calculation for a single employee.

        The LLM autonomously calls tools to fetch salary data, query leave,
        compute deductions, apply tax, and save the payslip before making
        an APPROVE/HOLD/FLAG decision.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(
            f"Running payroll for employee {input_data.employee_id}, "
            f"pay_cycle {input_data.pay_cycle_id}, "
            f"period {input_data.period_start} to {input_data.period_end}, "
            f"workflow {workflow_id}"
        )

        try:
            llm_input = f"""Calculate the payslip for the following employee:

EMPLOYEE ID: {input_data.employee_id}
PAY CYCLE ID: {input_data.pay_cycle_id}
PERIOD START: {input_data.period_start}
PERIOD END: {input_data.period_end}

Please call the tools in this exact order:
1. get_employee_payroll_info with employee_id={input_data.employee_id}
2. get_approved_leave with employee_id={input_data.employee_id}, period_start="{input_data.period_start}", period_end="{input_data.period_end}"
3. apply_leave_deductions with employee_id={input_data.employee_id}, monthly_gross=(base_salary/12 from step 1), period_start="{input_data.period_start}", period_end="{input_data.period_end}"
4. calculate_tax with employee_id={input_data.employee_id}, taxable_amount=(gross_pay - deductions_leave)
5. generate_payslip with employee_id={input_data.employee_id}, pay_cycle_id={input_data.pay_cycle_id}, and all computed values from steps 1-4

Then make your APPROVE/HOLD/FLAG decision.
"""

            self.logger.info("Invoking LLM agent for payroll calculation")
            llm_output, intermediate_steps = await self._run_agent_loop(llm_input)
            self.logger.info(f"LLM payroll output: {llm_output[:200]}...")

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered payroll calculation",
                    input_data={
                        "employee_id": input_data.employee_id,
                        "pay_cycle_id": input_data.pay_cycle_id,
                        "period_start": input_data.period_start,
                        "period_end": input_data.period_end,
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps),
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=self._parse_factors(llm_output),
                    alternatives_considered=["approve", "hold", "flag"],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            decision_info = self._parse_llm_decision(llm_output)
            payslip_data = self._extract_payslip_data(intermediate_steps)
            payslip_id = payslip_data.get("payslip_id")

            # Write LLM decision and reasoning back to the payslip
            if payslip_id:
                async with AsyncSessionLocal() as db:
                    result = await db.execute(
                        select(Payslip).where(Payslip.id == payslip_id)
                    )
                    payslip = result.scalar_one_or_none()
                    if payslip:
                        payslip.llm_decision = decision_info["decision"]
                        payslip.llm_reasoning = decision_info["reasoning"]
                        await db.flush()
                        await db.commit()

            return PayrollRunOutput(
                success=decision_info["decision"] == "APPROVE",
                message=self._build_message(decision_info["decision"]),
                reasoning=decision_info["reasoning"],
                decision=decision_info["decision"],
                confidence=decision_info["confidence_score"],
                payslip_id=payslip_id,
                employee_id=input_data.employee_id,
                gross_pay=payslip_data.get("gross_pay"),
                deductions_leave=payslip_data.get("deductions_leave"),
                deductions_tax=payslip_data.get("deductions_tax"),
                net_pay=payslip_data.get("net_pay"),
                days_worked=payslip_data.get("days_worked"),
                leave_days_taken=payslip_data.get("leave_days_taken"),
                factors=self._parse_factors(llm_output),
                recommendations=(
                    [decision_info["recommendations"]]
                    if decision_info.get("recommendations")
                    else []
                ),
            )

        except Exception as e:
            self.logger.error(
                f"Error running payroll for employee {input_data.employee_id}: {e}",
                exc_info=True,
            )

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="Payroll calculation failed",
                    input_data=input_data.dict(),
                    output_data={"error": str(e)},
                    confidence=ConfidenceLevel.VERY_LOW,
                    reasoning=f"System error: {str(e)}",
                    factors=["system_error"],
                    alternatives_considered=[],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            return PayrollRunOutput(
                success=False,
                message=f"Payroll calculation failed: {str(e)}",
                reasoning=f"System error during payroll run: {str(e)}",
                decision="HOLD",
                confidence=0.0,
                employee_id=input_data.employee_id,
                factors=["system_error"],
            )

    # -------------------------------------------------------------------------
    # Tool Functions (called autonomously by the LLM)
    # -------------------------------------------------------------------------

    async def _tool_get_employee_payroll_info(self, employee_id: int) -> str:
        """Fetch employee salary data and payroll configuration."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Employee).where(Employee.id == employee_id)
                )
                employee = result.scalar_one_or_none()

            if not employee:
                return json.dumps({
                    "success": False,
                    "message": (
                        f"Employee {employee_id} not found in the database. "
                        "Cannot process payroll — use HOLD decision."
                    ),
                })

            if not employee.base_salary:
                return json.dumps({
                    "success": False,
                    "employee_id": employee_id,
                    "first_name": employee.first_name,
                    "last_name": employee.last_name,
                    "message": (
                        f"Employee {employee.first_name} {employee.last_name} has no base_salary set. "
                        "Cannot calculate payroll — use HOLD decision."
                    ),
                })

            monthly_gross = round(employee.base_salary / 12, 2)
            tax_rate = employee.tax_rate or 0.20

            return json.dumps({
                "success": True,
                "employee_id": employee_id,
                "first_name": employee.first_name,
                "last_name": employee.last_name,
                "department": employee.department,
                "base_salary": employee.base_salary,
                "monthly_gross": monthly_gross,
                "pay_frequency": employee.pay_frequency or "monthly",
                "tax_rate": tax_rate,
                "message": (
                    f"Employee: {employee.first_name} {employee.last_name} ({employee.department}). "
                    f"Annual salary: {employee.base_salary:.2f}, monthly gross: {monthly_gross:.2f}, "
                    f"tax rate: {tax_rate:.0%}."
                ),
            })

        except Exception as e:
            self.logger.error(f"get_employee_payroll_info tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve payroll info: {str(e)}",
            })

    async def _tool_get_approved_leave(
        self, employee_id: int, period_start: str, period_end: str
    ) -> str:
        """Query all approved leave overlapping the pay period for this employee."""
        try:
            period_start_date = date.fromisoformat(period_start)
            period_end_date = date.fromisoformat(period_end)
            working_days = self._count_working_days(period_start_date, period_end_date)

            async with AsyncSessionLocal() as db:
                emp_result = await db.execute(
                    select(Employee).where(Employee.id == employee_id)
                )
                employee = emp_result.scalar_one_or_none()
                if not employee:
                    return json.dumps({
                        "success": False,
                        "message": f"Employee {employee_id} not found.",
                    })

                user_id = employee.user_id

                leave_result = await db.execute(
                    select(LeaveRequest).where(
                        and_(
                            LeaveRequest.user_id == user_id,
                            LeaveRequest.status == LeaveRequestStatus.APPROVED.value,
                            LeaveRequest.start_date <= period_end_date,
                            LeaveRequest.end_date >= period_start_date,
                        )
                    )
                )
                leave_requests = leave_result.scalars().all()

                period_year = period_start_date.year
                balance_result = await db.execute(
                    select(LeaveBalance).where(
                        and_(
                            LeaveBalance.user_id == user_id,
                            LeaveBalance.year == period_year,
                        )
                    )
                )
                balances = balance_result.scalars().all()
                balance_map = {b.leave_type: round(b.remaining_days, 2) for b in balances}

            # Calculate the overlap days per request that fall within this period
            leave_details = []
            total_leave_days = 0.0

            for req in leave_requests:
                overlap_start = max(req.start_date, period_start_date)
                overlap_end = min(req.end_date, period_end_date)
                overlap_days = float(self._count_working_days(overlap_start, overlap_end))
                total_leave_days += overlap_days
                leave_details.append({
                    "request_id": req.id,
                    "leave_type": req.leave_type,
                    "days_in_period": overlap_days,
                    "full_request_days": req.days_requested,
                })

            by_type: Dict[str, float] = {}
            for d in leave_details:
                by_type[d["leave_type"]] = by_type.get(d["leave_type"], 0.0) + d["days_in_period"]

            days_worked = max(0.0, working_days - total_leave_days)

            return json.dumps({
                "success": True,
                "employee_id": employee_id,
                "period_start": period_start,
                "period_end": period_end,
                "working_days_in_period": working_days,
                "total_leave_days": total_leave_days,
                "days_worked": days_worked,
                "leave_by_type": by_type,
                "leave_balance_remaining": balance_map,
                "leave_requests": leave_details,
                "message": (
                    f"{len(leave_requests)} approved leave request(s) in period. "
                    f"Total leave: {total_leave_days} day(s) out of {working_days} working days. "
                    f"Days worked: {days_worked}. "
                    f"Leave balance remaining: {balance_map}."
                ),
            })

        except Exception as e:
            self.logger.error(f"get_approved_leave tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve approved leave: {str(e)}",
            })

    async def _tool_apply_leave_deductions(
        self, employee_id: int, monthly_gross: float, period_start: str, period_end: str
    ) -> str:
        """Calculate the unpaid leave deduction for days exceeding the leave balance."""
        try:
            period_start_date = date.fromisoformat(period_start)
            period_end_date = date.fromisoformat(period_end)
            working_days = self._count_working_days(period_start_date, period_end_date)

            if working_days == 0:
                return json.dumps({
                    "success": False,
                    "message": "No working days found in the specified period.",
                })

            daily_rate = monthly_gross / working_days

            async with AsyncSessionLocal() as db:
                emp_result = await db.execute(
                    select(Employee).where(Employee.id == employee_id)
                )
                employee = emp_result.scalar_one_or_none()
                if not employee:
                    return json.dumps({
                        "success": False,
                        "message": f"Employee {employee_id} not found.",
                    })

                user_id = employee.user_id
                period_year = period_start_date.year

                leave_result = await db.execute(
                    select(LeaveRequest).where(
                        and_(
                            LeaveRequest.user_id == user_id,
                            LeaveRequest.status == LeaveRequestStatus.APPROVED.value,
                            LeaveRequest.start_date <= period_end_date,
                            LeaveRequest.end_date >= period_start_date,
                        )
                    )
                )
                leave_requests = leave_result.scalars().all()

                balance_result = await db.execute(
                    select(LeaveBalance).where(
                        and_(
                            LeaveBalance.user_id == user_id,
                            LeaveBalance.year == period_year,
                        )
                    )
                )
                balances = balance_result.scalars().all()
                balance_map = {b.leave_type: b.remaining_days for b in balances}

            # Aggregate leave days in period by type
            days_by_type: Dict[str, float] = {}
            for req in leave_requests:
                overlap_start = max(req.start_date, period_start_date)
                overlap_end = min(req.end_date, period_end_date)
                days = float(self._count_working_days(overlap_start, overlap_end))
                days_by_type[req.leave_type] = days_by_type.get(req.leave_type, 0.0) + days

            # Unpaid = days taken that exceed remaining balance per type
            unpaid_breakdown = []
            total_unpaid_days = 0.0

            for leave_type, days_taken in days_by_type.items():
                balance = balance_map.get(leave_type, 0.0)
                unpaid = max(0.0, days_taken - balance)
                if unpaid > 0:
                    unpaid_breakdown.append({
                        "leave_type": leave_type,
                        "days_taken": days_taken,
                        "balance_available": round(balance, 2),
                        "unpaid_days": round(unpaid, 2),
                    })
                    total_unpaid_days += unpaid

            deduction_amount = round(total_unpaid_days * daily_rate, 2)

            return json.dumps({
                "success": True,
                "employee_id": employee_id,
                "working_days_in_period": working_days,
                "daily_rate": round(daily_rate, 2),
                "total_unpaid_days": total_unpaid_days,
                "deduction_amount": deduction_amount,
                "unpaid_breakdown": unpaid_breakdown,
                "message": (
                    f"Unpaid leave: {total_unpaid_days} day(s). "
                    f"Daily rate: {daily_rate:.2f}. "
                    f"Leave deduction: {deduction_amount:.2f}."
                    + (
                        f" Breakdown: {unpaid_breakdown}"
                        if unpaid_breakdown
                        else " No unpaid leave — all leave is within balance."
                    )
                ),
            })

        except Exception as e:
            self.logger.error(f"apply_leave_deductions tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Leave deduction calculation failed: {str(e)}",
            })

    async def _tool_calculate_tax(self, employee_id: int, taxable_amount: float) -> str:
        """Calculate flat-rate tax on the taxable pay amount."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Employee).where(Employee.id == employee_id)
                )
                employee = result.scalar_one_or_none()

            if not employee:
                return json.dumps({
                    "success": False,
                    "message": f"Employee {employee_id} not found.",
                })

            tax_rate = employee.tax_rate or 0.20
            tax_amount = round(taxable_amount * tax_rate, 2)
            net_after_tax = round(taxable_amount - tax_amount, 2)

            return json.dumps({
                "success": True,
                "employee_id": employee_id,
                "taxable_amount": round(taxable_amount, 2),
                "tax_rate": tax_rate,
                "tax_amount": tax_amount,
                "net_after_tax": net_after_tax,
                "message": (
                    f"Tax: {taxable_amount:.2f} * {tax_rate:.0%} = {tax_amount:.2f}. "
                    f"Net after tax: {net_after_tax:.2f}."
                ),
            })

        except Exception as e:
            self.logger.error(f"calculate_tax tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Tax calculation failed: {str(e)}",
            })

    async def _tool_generate_payslip(
        self,
        employee_id: int,
        pay_cycle_id: int,
        gross_pay: float,
        deductions_leave: float,
        deductions_tax: float,
        days_worked: float,
        leave_days_taken: float,
    ) -> str:
        """Assemble all computed values and save the payslip to the database."""
        try:
            net_pay = round(gross_pay - deductions_leave - deductions_tax, 2)

            # Check variance against most recent payslip for FLAG detection
            variance_info = None
            async with AsyncSessionLocal() as db:
                last_result = await db.execute(
                    select(Payslip)
                    .where(Payslip.employee_id == employee_id)
                    .order_by(Payslip.created_at.desc())
                    .limit(1)
                )
                last_payslip = last_result.scalar_one_or_none()

            if last_payslip and last_payslip.net_pay and last_payslip.net_pay > 0:
                variance = abs(net_pay - last_payslip.net_pay) / last_payslip.net_pay
                variance_info = {
                    "last_net_pay": last_payslip.net_pay,
                    "current_net_pay": net_pay,
                    "variance_pct": round(variance * 100, 1),
                    "flag_threshold_exceeded": variance > 0.20,
                }

            # Create the payslip record (llm_decision written after the agent loop)
            async with AsyncSessionLocal() as db:
                payslip = Payslip(
                    employee_id=employee_id,
                    pay_cycle_id=pay_cycle_id,
                    gross_pay=round(gross_pay, 2),
                    deductions_leave=round(deductions_leave, 2),
                    deductions_tax=round(deductions_tax, 2),
                    net_pay=net_pay,
                    days_worked=round(days_worked, 1),
                    leave_days_taken=round(leave_days_taken, 1),
                    status=PayslipStatus.DRAFT.value,
                )
                db.add(payslip)
                await db.flush()
                await db.refresh(payslip)
                payslip_id = payslip.id
                await db.commit()

            flag_warning = ""
            if net_pay <= 0:
                flag_warning = " WARNING: Net pay is zero or negative — use FLAG decision."
            elif variance_info and variance_info["flag_threshold_exceeded"]:
                flag_warning = (
                    f" WARNING: Net pay variance {variance_info['variance_pct']:.1f}% "
                    f"vs last period ({variance_info['last_net_pay']:.2f}) exceeds 20% — "
                    "use FLAG decision."
                )

            return json.dumps({
                "success": True,
                "payslip_id": payslip_id,
                "employee_id": employee_id,
                "pay_cycle_id": pay_cycle_id,
                "gross_pay": round(gross_pay, 2),
                "deductions_leave": round(deductions_leave, 2),
                "deductions_tax": round(deductions_tax, 2),
                "net_pay": net_pay,
                "days_worked": round(days_worked, 1),
                "leave_days_taken": round(leave_days_taken, 1),
                "variance_vs_last_period": variance_info,
                "message": (
                    f"Payslip #{payslip_id} created (DRAFT). "
                    f"Gross: {gross_pay:.2f} | Leave deduction: {deductions_leave:.2f} | "
                    f"Tax: {deductions_tax:.2f} | Net: {net_pay:.2f}."
                    + flag_warning
                ),
            })

        except Exception as e:
            self.logger.error(f"generate_payslip tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Payslip generation failed: {str(e)}",
            })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _count_working_days(start: date, end: date) -> int:
        """Count Monday–Friday days inclusive between start and end."""
        if start > end:
            return 0
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon=0, Fri=4
                count += 1
            current += timedelta(days=1)
        return count

    def _parse_llm_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse structured fields from the LLM's formatted output."""
        decision_info: Dict[str, Any] = {
            "decision": "HOLD",  # Safe default if parsing fails
            "reasoning": llm_output,
            "recommendations": "Review payroll data manually",
            "confidence_score": 0.5,
        }

        _stop = ["CONFIDENCE:", "FACTORS:", "RECOMMENDATIONS:", "DECISION:"]

        # Parse DECISION
        if "DECISION:" in llm_output:
            for line in llm_output.split("\n"):
                if "DECISION:" in line:
                    if "APPROVE" in line:
                        decision_info["decision"] = "APPROVE"
                    elif "FLAG" in line:
                        decision_info["decision"] = "FLAG"
                    elif "HOLD" in line:
                        decision_info["decision"] = "HOLD"
                    break

        # Parse REASONING
        if "REASONING:" in llm_output:
            lines, capture = [], False
            for line in llm_output.split("\n"):
                if "REASONING:" in line:
                    capture = True
                    lines.append(line.replace("REASONING:", "").strip())
                elif capture and line.strip() and not any(k in line for k in _stop):
                    lines.append(line.strip())
                elif capture and any(k in line for k in _stop):
                    break
            if lines:
                decision_info["reasoning"] = " ".join(lines)

        # Parse RECOMMENDATIONS
        if "RECOMMENDATIONS:" in llm_output:
            lines, capture = [], False
            for line in llm_output.split("\n"):
                if "RECOMMENDATIONS:" in line:
                    capture = True
                    lines.append(line.replace("RECOMMENDATIONS:", "").strip())
                elif capture and line.strip() and not any(k in line for k in _stop):
                    lines.append(line.strip())
                elif capture and any(k in line for k in _stop):
                    break
            if lines:
                decision_info["recommendations"] = " ".join(lines)

        # Parse CONFIDENCE into numeric score
        for label, score in {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.items():
            if f"CONFIDENCE: {label}" in llm_output:
                decision_info["confidence_score"] = score
                break

        return decision_info

    def _extract_payslip_data(self, intermediate_steps: list) -> Dict[str, Any]:
        """Extract payslip numbers from the generate_payslip tool result."""
        for tool_name, _, result in intermediate_steps:
            if tool_name == "generate_payslip":
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                    return {
                        "payslip_id": data.get("payslip_id"),
                        "gross_pay": data.get("gross_pay"),
                        "deductions_leave": data.get("deductions_leave"),
                        "deductions_tax": data.get("deductions_tax"),
                        "net_pay": data.get("net_pay"),
                        "days_worked": data.get("days_worked"),
                        "leave_days_taken": data.get("leave_days_taken"),
                    }
                except Exception as e:
                    self.logger.warning(f"Could not parse payslip data from intermediate steps: {e}")
        return {}

    def _parse_confidence(self, llm_output: str) -> ConfidenceLevel:
        if "CONFIDENCE: HIGH" in llm_output:
            return ConfidenceLevel.HIGH
        elif "CONFIDENCE: MEDIUM" in llm_output:
            return ConfidenceLevel.MEDIUM
        elif "CONFIDENCE: LOW" in llm_output:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    def _parse_factors(self, llm_output: str) -> List[str]:
        if "FACTORS:" in llm_output:
            for line in llm_output.split("\n"):
                if "FACTORS:" in line:
                    text = line.replace("FACTORS:", "").strip()
                    return [f.strip() for f in text.split(",") if f.strip()]
        return ["salary_verified", "leave_checked", "tax_applied"]

    def _build_message(self, decision: str) -> str:
        return {
            "APPROVE": "Payslip approved — calculations verified",
            "HOLD": "Payslip on hold — missing or incomplete data",
            "FLAG": "Payslip flagged — unusual variance or anomaly detected",
        }.get(decision, "Payslip processed")

    def get_input_schema(self) -> type[PayrollRunInput]:
        return PayrollRunInput

    def get_output_schema(self) -> type[PayrollRunOutput]:
        return PayrollRunOutput


# Agent is registered via agents/register_agents.py on application startup
