"""
Expense Claim Agent - LangChain Powered

Validates and processes employee expense claims using GPT-4o-mini for autonomous
decision-making. The LLM analyzes OCR receipt data, policy limits, duplicate
checks, and monthly spend to make approval decisions with natural language reasoning.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import logging
import os
import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel as BaseModelV1, Field  # LangChain 0.1.x uses Pydantic V1

from agents.base import Agent
from agents.schemas import (
    ExpenseSubmitInput,
    ExpenseValidationOutput,
    OCRExtractionResult,
)
from agents.approval_system import approval_manager, EscalationTrigger
from agents.reasoning_system import reasoning_manager, ReasoningStep, ReasoningType, ConfidenceLevel
from app.database import AsyncSessionLocal
from app.models import Expense, ExpensePolicy, ExpenseStatus, Employee


# ---------------------------------------------------------------------------
# Tool Input Schemas (Pydantic V1 required for LangChain 0.1.x)
# ---------------------------------------------------------------------------

class ValidateReceiptOCRInput(BaseModelV1):
    expense_id: int = Field(description="ID of the expense record already saved in the database")
    receipt_path: str = Field(description="Full filesystem path to the saved receipt image")
    submitted_amount: float = Field(description="Amount the employee claimed on the form")
    submitted_vendor: str = Field(default="", description="Vendor name the employee entered on the form")
    submitted_date: str = Field(description="Date the employee entered on the form (YYYY-MM-DD)")


class CheckPolicyLimitsInput(BaseModelV1):
    category: str = Field(description="Expense category: meals, travel, equipment, entertainment, office_supplies, or other")
    amount: float = Field(description="Expense amount to validate against policy limits")


class DetectDuplicatesInput(BaseModelV1):
    user_id: int = Field(description="Employee user ID")
    vendor: str = Field(default="", description="Vendor name of the expense")
    amount: float = Field(description="Expense amount")
    expense_date: str = Field(description="Date of the expense in YYYY-MM-DD format")
    current_expense_id: int = Field(description="ID of the current expense to exclude from the duplicate search")


class CheckMonthlySpendInput(BaseModelV1):
    user_id: int = Field(description="Employee user ID")
    category: str = Field(description="Expense category to sum monthly approved totals for")


# ---------------------------------------------------------------------------
# Expense Agent
# ---------------------------------------------------------------------------

class ExpenseAgent(Agent[ExpenseSubmitInput, ExpenseValidationOutput]):
    """
    LangChain-powered agent for expense claim validation and approval.

    Uses GPT-4o-mini to analyze expense claims against OCR receipt data,
    company policy limits, and duplicate detection to make autonomous
    approval decisions with natural language reasoning.
    """

    def __init__(self):
        super().__init__(
            name="expense_agent",
            description="LangChain-powered agent that validates expense claims using OCR receipt analysis and GPT-4o-mini for autonomous decision-making"
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.tools = [
            StructuredTool.from_function(
                coroutine=self._tool_validate_receipt_ocr,
                name="validate_receipt_ocr",
                description=(
                    "Run OCR on the uploaded receipt image and compare extracted data "
                    "(amount, vendor, date) against what the employee submitted. "
                    "Returns mismatch flags and per-field confidence scores. "
                    "ALWAYS call this first when a receipt was uploaded."
                ),
                args_schema=ValidateReceiptOCRInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_check_policy_limits,
                name="check_policy_limits",
                description=(
                    "Check the submitted expense amount against company policy limits "
                    "for the given category. Returns the auto-approve threshold, hard "
                    "maximum, and a verdict: AUTO_APPROVE, ESCALATE, or REJECT."
                ),
                args_schema=CheckPolicyLimitsInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_detect_duplicates,
                name="detect_duplicates",
                description=(
                    "Check whether this expense is a duplicate of a recently submitted "
                    "expense from the same employee within a 7-day window (same vendor "
                    "and similar amount). Returns duplicate flag and details."
                ),
                args_schema=DetectDuplicatesInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_check_monthly_spend,
                name="check_monthly_spend",
                description=(
                    "Get the total approved expenses this calendar month for this employee "
                    "in this category. Returns total spent and percentage of monthly limit used."
                ),
                args_schema=CheckMonthlySpendInput,
                return_direct=False,
            ),
        ]

        self.system_prompt = """You are an Expense Claim Validation Agent for an HR management system.

Your role is to analyze employee expense claims and make autonomous approval decisions
based on receipt OCR analysis and company expense policies.

TOOLS AVAILABLE:
1. validate_receipt_ocr  - Run OCR on the receipt image and compare with submitted data
2. check_policy_limits   - Check amount against category policy limits
3. detect_duplicates     - Check for duplicate expense submissions within 7 days
4. check_monthly_spend   - Check how much the employee has spent this month in this category

DECISION PROCESS:
1. If a receipt was uploaded, ALWAYS call validate_receipt_ocr first
2. Call check_policy_limits to verify the amount is within policy
3. Call detect_duplicates to check for potential fraud
4. Call check_monthly_spend to check for excessive spending patterns
5. Analyze all results and make a clear final decision

EXPENSE POLICIES (hardcoded until Policy Agent is built in Phase 7):
- meals:           auto-approve <= $25,  escalate $25-$50,   reject > $50
- travel:          auto-approve <= $100, escalate $100-$200, reject > $200
- equipment:       auto-approve <= $200, escalate $200-$500, reject > $500
- entertainment:   auto-approve <= $50,  escalate $50-$100,  reject > $100
- office_supplies: auto-approve <= $50,  escalate $50-$100,  reject > $100
- other:           auto-approve <= $50,  escalate $50-$150,  reject > $150

AUTO-APPROVAL CRITERIA (ALL must be met):
- Amount is at or below the auto-approve threshold for this category
- OCR confidence >= 0.70 (if receipt uploaded) OR claim is low-value with no receipt
- OCR extracted amount matches submitted amount within 10% (if receipt uploaded)
- No duplicate detected within 7 days
- Monthly spend in this category remains below 80% of limit after this claim

ESCALATION TO MANAGER:
- Amount exceeds auto-approve threshold but is below hard maximum
- OCR confidence is between 0.40 and 0.70 (needs manual receipt review)
- Monthly spend would exceed 80% of category limit
- Receipt uploaded but OCR could not reliably extract key fields

REJECTION:
- Amount exceeds hard maximum for category (hard block)
- Clear duplicate detected: same vendor and similar amount within 7 days (hard block)
- OCR extracted amount differs from submitted amount by more than 20% (hard block)
- OCR confidence below 0.40 — receipt is unreadable or invalid (hard block)

OUTPUT FORMAT (use this exact structure):
DECISION: [AUTO_APPROVE | ESCALATE | REJECT]
REASONING: [2-3 sentences explaining your analysis and decision]
CONFIDENCE: [HIGH | MEDIUM | LOW]
OCR_MATCH: [TRUE | FALSE | NO_RECEIPT]
FACTORS: [comma-separated list of key factors considered]
RECOMMENDATIONS: [specific recommendations for the manager if escalating, or next steps for the employee]

Be thorough, call all relevant tools, and provide clear reasoning for every decision."""

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.logger.info("Expense Agent initialized with LangChain + GPT-4o-mini")

    # -------------------------------------------------------------------------
    # Agent Loop (identical pattern to LeaveAgent)
    # -------------------------------------------------------------------------

    async def _run_agent_loop(self, user_input: str, max_iterations: int = 6) -> tuple[str, list]:
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

    async def execute(self, input_data: ExpenseSubmitInput) -> ExpenseValidationOutput:
        """
        Execute expense claim validation using the LangChain agent.

        The LLM autonomously calls tools to gather OCR data, policy limits,
        duplicate checks, and monthly spend before making a final decision.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(
            f"Processing expense claim for user {input_data.user_id}, workflow {workflow_id}"
        )

        try:
            # Create the Expense record first so tools can reference it by ID
            async with AsyncSessionLocal() as db:
                expense = Expense(
                    user_id=input_data.user_id,
                    amount=input_data.amount,
                    category=input_data.category,
                    vendor=input_data.vendor,
                    date=datetime.strptime(input_data.date, "%Y-%m-%d").date(),
                    description=input_data.description,
                    receipt_filename=input_data.receipt_filename,
                    receipt_path=input_data.receipt_path,
                    status=ExpenseStatus.SUBMITTED.value,
                    submitted_at=datetime.utcnow(),
                )
                db.add(expense)
                await db.flush()
                await db.refresh(expense)
                expense_id = expense.id  # Cache ID before leaving async context
                await db.commit()

            has_receipt = bool(input_data.receipt_path)

            llm_input = f"""Analyze the following expense claim and make an approval decision:

EMPLOYEE ID: {input_data.user_id}
EXPENSE ID: {expense_id}
CATEGORY: {input_data.category}
SUBMITTED AMOUNT: ${input_data.amount:.2f}
VENDOR: {input_data.vendor or 'Not provided'}
DATE: {input_data.date}
DESCRIPTION: {input_data.description or 'Not provided'}
RECEIPT UPLOADED: {"Yes - file path: " + str(input_data.receipt_path) if has_receipt else "No receipt uploaded"}

Use the available tools to:
1. {"Call validate_receipt_ocr with expense_id=" + str(expense_id) + " and receipt_path=" + str(input_data.receipt_path) if has_receipt else "No receipt to validate - skip validate_receipt_ocr"}
2. Call check_policy_limits for category '{input_data.category}' with amount {input_data.amount}
3. Call detect_duplicates with user_id={input_data.user_id}, amount={input_data.amount}, expense_date='{input_data.date}', current_expense_id={expense_id}
4. Call check_monthly_spend for user_id={input_data.user_id} and category='{input_data.category}'

Then make your final decision following the criteria in your instructions.
"""

            self.logger.info("Invoking LLM agent for expense claim validation")
            llm_output, intermediate_steps = await self._run_agent_loop(llm_input)
            self.logger.info(f"LLM decision output: {llm_output[:200]}...")

            # Record reasoning trace
            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered expense claim decision",
                    input_data={
                        "user_id": input_data.user_id,
                        "expense_id": expense_id,
                        "category": input_data.category,
                        "amount": input_data.amount,
                        "has_receipt": has_receipt,
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps),
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=self._parse_factors(llm_output),
                    alternatives_considered=["auto_approve", "escalate", "reject"],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            decision_info = self._parse_llm_decision(llm_output)
            ocr_result = self._extract_ocr_from_steps(intermediate_steps)

            # Update expense record with decision and OCR data
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Expense).where(Expense.id == expense_id))
                expense = result.scalar_one()

                expense.llm_decision = decision_info["decision"]
                expense.llm_reasoning = decision_info["reasoning"]

                if ocr_result:
                    expense.ocr_text = ocr_result.raw_text
                    expense.ocr_confidence = ocr_result.overall_confidence
                    expense.ocr_extracted = {
                        "vendor": ocr_result.vendor,
                        "amount": ocr_result.amount,
                        "date": ocr_result.date,
                        "vendor_confidence": ocr_result.vendor_confidence,
                        "amount_confidence": ocr_result.amount_confidence,
                        "date_confidence": ocr_result.date_confidence,
                    }

                if decision_info["decision"] == "AUTO_APPROVE":
                    expense.status = ExpenseStatus.APPROVED.value
                    expense.reviewed_at = datetime.utcnow()
                elif decision_info["decision"] == "REJECT":
                    expense.status = ExpenseStatus.REJECTED.value
                    expense.reviewed_at = datetime.utcnow()
                    expense.rejection_reason = decision_info["reasoning"]
                # ESCALATE stays as SUBMITTED — manager reviews via pending approvals

                await db.commit()

            # Create approval queue entry for escalations
            if decision_info["decision"] == "ESCALATE":
                await self._create_approval_request(
                    expense_id=expense_id,
                    input_data=input_data,
                    workflow_id=workflow_id,
                    llm_reasoning=decision_info["reasoning"],
                    ocr_result=ocr_result,
                )

            return ExpenseValidationOutput(
                success=decision_info["decision"] != "REJECT",
                message=self._build_message(decision_info["decision"]),
                reasoning=decision_info["reasoning"],
                decision=decision_info["decision"],
                confidence=decision_info["confidence_score"],
                expense_id=expense_id,
                validated_amount=input_data.amount,
                ocr_result=ocr_result,
                ocr_match=decision_info.get("ocr_match", True),
                policy_violations=decision_info.get("violations", []),
                factors=self._parse_factors(llm_output),
                recommendations=(
                    [decision_info["recommendations"]]
                    if decision_info.get("recommendations")
                    else []
                ),
            )

        except Exception as e:
            self.logger.error(f"Error processing expense claim: {e}", exc_info=True)

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="Expense claim processing failed",
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

            return ExpenseValidationOutput(
                success=False,
                message=f"Failed to process expense claim: {str(e)}",
                reasoning=f"System error during processing: {str(e)}",
                decision="ESCALATE",  # Fail-safe to escalation, never silently auto-approve
                confidence=0.0,
                expense_id=None,
                policy_violations=[f"System error: {str(e)}"],
                factors=["system_error"],
            )

    # -------------------------------------------------------------------------
    # LangChain Tool Functions (called autonomously by the LLM)
    # -------------------------------------------------------------------------

    async def _tool_validate_receipt_ocr(
        self,
        expense_id: int,
        receipt_path: str,
        submitted_amount: float,
        submitted_vendor: str,
        submitted_date: str,
    ) -> str:
        """Run OCR on the receipt image and compare with the submitted expense fields."""
        try:
            from agents.tools.receipt_ocr import process_receipt

            # EasyOCR is synchronous — run in thread pool to avoid blocking the event loop
            ocr_result = await asyncio.to_thread(process_receipt, receipt_path)

            # Compare submitted amount vs OCR-extracted amount
            mismatches = []
            amount_diff_pct = 0.0

            if ocr_result.amount is not None:
                amount_diff_pct = (
                    abs(ocr_result.amount - submitted_amount)
                    / max(submitted_amount, 0.01)
                    * 100
                )
                if amount_diff_pct > 20:
                    mismatches.append(
                        f"Amount mismatch: submitted ${submitted_amount:.2f}, "
                        f"OCR detected ${ocr_result.amount:.2f} "
                        f"({amount_diff_pct:.1f}% difference)"
                    )

            return json.dumps({
                "ocr_vendor": ocr_result.vendor,
                "ocr_amount": ocr_result.amount,
                "ocr_date": ocr_result.date,
                "ocr_raw_text": ocr_result.raw_text[:400],  # Truncate for token efficiency
                "vendor_confidence": ocr_result.vendor_confidence,
                "amount_confidence": ocr_result.amount_confidence,
                "date_confidence": ocr_result.date_confidence,
                "overall_confidence": ocr_result.overall_confidence,
                "submitted_amount": submitted_amount,
                "submitted_vendor": submitted_vendor,
                "amount_diff_pct": round(amount_diff_pct, 1),
                "mismatches": mismatches,
                "ocr_match": len(mismatches) == 0,
                "message": (
                    f"OCR overall confidence: {ocr_result.overall_confidence:.0%}. "
                    + (
                        f"Mismatches: {'; '.join(mismatches)}"
                        if mismatches
                        else "No significant mismatches."
                    )
                    + (
                        " LOW CONFIDENCE - escalate for manual review."
                        if ocr_result.overall_confidence < 0.7
                        else ""
                    )
                ),
            })

        except Exception as e:
            self.logger.error(f"OCR tool error: {e}")
            return json.dumps({
                "error": str(e),
                "overall_confidence": 0.0,
                "ocr_match": False,
                "message": (
                    f"OCR processing failed: {str(e)}. "
                    "Treat as low confidence — escalate for manual review."
                ),
            })

    async def _tool_check_policy_limits(self, category: str, amount: float) -> str:
        """Check submitted amount against the ExpensePolicy table for this category."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ExpensePolicy).where(ExpensePolicy.category == category)
            )
            policy = result.scalar_one_or_none()

        # Fallback defaults if policies have not been seeded yet
        _defaults = {
            "meals":           (25.0,  50.0),
            "travel":          (100.0, 200.0),
            "equipment":       (200.0, 500.0),
            "entertainment":   (50.0,  100.0),
            "office_supplies": (50.0,  100.0),
            "other":           (50.0,  150.0),
        }

        if not policy:
            approval_threshold, max_amount = _defaults.get(category, (50.0, 150.0))
            requires_receipt = amount > 25.0
        else:
            approval_threshold = policy.approval_threshold
            max_amount = policy.max_amount
            requires_receipt = policy.requires_receipt

        if amount > max_amount:
            verdict = "REJECT"
            verdict_reason = (
                f"Amount ${amount:.2f} exceeds the hard maximum of ${max_amount:.2f} "
                f"for {category}. This is a blocking rejection."
            )
        elif amount > approval_threshold:
            verdict = "ESCALATE"
            verdict_reason = (
                f"Amount ${amount:.2f} exceeds the auto-approve threshold of "
                f"${approval_threshold:.2f} for {category} (hard max: ${max_amount:.2f}). "
                "Requires manager approval."
            )
        else:
            verdict = "AUTO_APPROVE"
            verdict_reason = (
                f"Amount ${amount:.2f} is within the auto-approve limit of "
                f"${approval_threshold:.2f} for {category}."
            )

        return json.dumps({
            "category": category,
            "submitted_amount": amount,
            "approval_threshold": approval_threshold,
            "max_amount": max_amount,
            "requires_receipt": requires_receipt,
            "policy_verdict": verdict,
            "message": verdict_reason,
        })

    async def _tool_detect_duplicates(
        self,
        user_id: int,
        vendor: str,
        amount: float,
        expense_date: str,
        current_expense_id: int,
    ) -> str:
        """Check for duplicate expense submissions within a 7-day window."""
        try:
            check_date = datetime.strptime(expense_date, "%Y-%m-%d").date()
        except ValueError:
            check_date = date.today()

        window_start = check_date - timedelta(days=7)
        window_end = check_date + timedelta(days=7)
        amount_lower = amount * 0.9
        amount_upper = amount * 1.1

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Expense).where(
                    and_(
                        Expense.user_id == user_id,
                        Expense.id != current_expense_id,
                        Expense.status.in_([
                            ExpenseStatus.SUBMITTED.value,
                            ExpenseStatus.APPROVED.value,
                        ]),
                        Expense.amount.between(amount_lower, amount_upper),
                        Expense.date.between(window_start, window_end),
                    )
                )
            )
            duplicates = result.scalars().all()

        if duplicates:
            dup_info = [
                f"Expense #{d.id}: ${d.amount:.2f} at {d.vendor or 'unknown vendor'} "
                f"on {d.date} (status: {d.status})"
                for d in duplicates
            ]
            return json.dumps({
                "is_duplicate": True,
                "duplicate_count": len(duplicates),
                "duplicates": dup_info,
                "message": (
                    f"DUPLICATE DETECTED: {len(duplicates)} similar expense(s) within 7 days: "
                    + "; ".join(dup_info)
                    + ". This is a blocking rejection reason."
                ),
            })

        return json.dumps({
            "is_duplicate": False,
            "duplicate_count": 0,
            "duplicates": [],
            "message": "No duplicate expenses detected within the 7-day window.",
        })

    async def _tool_check_monthly_spend(self, user_id: int, category: str) -> str:
        """Sum approved expenses this calendar month by category and compare to monthly limit."""
        month_start = date.today().replace(day=1)

        async with AsyncSessionLocal() as db:
            spend_result = await db.execute(
                select(func.sum(Expense.amount)).where(
                    and_(
                        Expense.user_id == user_id,
                        Expense.category == category,
                        Expense.status == ExpenseStatus.APPROVED.value,
                        Expense.date >= month_start,
                    )
                )
            )
            monthly_total = spend_result.scalar() or 0.0

            policy_result = await db.execute(
                select(ExpensePolicy).where(ExpensePolicy.category == category)
            )
            policy = policy_result.scalar_one_or_none()

        # Monthly cap = 3x the single-claim hard maximum (reasonable heuristic)
        monthly_limit = (policy.max_amount * 3) if policy else 300.0
        pct_used = (monthly_total / monthly_limit * 100) if monthly_limit > 0 else 0.0
        near_limit = pct_used >= 80.0

        return json.dumps({
            "category": category,
            "monthly_total_approved": round(monthly_total, 2),
            "monthly_limit": monthly_limit,
            "pct_used": round(pct_used, 1),
            "near_limit": near_limit,
            "message": (
                f"Employee has spent ${monthly_total:.2f} on {category} this month "
                f"({pct_used:.1f}% of the ${monthly_limit:.2f} monthly limit). "
                + (
                    "WARNING: Near monthly limit — escalate for manager review."
                    if near_limit
                    else "Within normal monthly spending limits."
                )
            ),
        })

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _parse_llm_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse structured fields from the LLM's formatted output."""
        decision_info = {
            "decision": "ESCALATE",  # Safe default if parsing fails
            "reasoning": llm_output,
            "recommendations": "Review the expense claim manually",
            "violations": [],
            "ocr_match": True,
            "confidence_score": 0.5,
        }

        # Parse DECISION
        if "DECISION:" in llm_output:
            decision_lines = [l for l in llm_output.split("\n") if "DECISION:" in l]
            if decision_lines:
                line = decision_lines[0]
                if "AUTO_APPROVE" in line:
                    decision_info["decision"] = "AUTO_APPROVE"
                elif "REJECT" in line:
                    decision_info["decision"] = "REJECT"
                elif "ESCALATE" in line:
                    decision_info["decision"] = "ESCALATE"

        # Parse REASONING
        _stop_keywords = ["CONFIDENCE:", "FACTORS:", "RECOMMENDATIONS:", "DECISION:", "OCR_MATCH:"]
        if "REASONING:" in llm_output:
            lines, capture = [], False
            for line in llm_output.split("\n"):
                if "REASONING:" in line:
                    capture = True
                    lines.append(line.replace("REASONING:", "").strip())
                elif capture and line.strip() and not any(k in line for k in _stop_keywords):
                    lines.append(line.strip())
                elif capture and any(k in line for k in _stop_keywords):
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
                elif capture and line.strip() and not any(k in line for k in _stop_keywords):
                    lines.append(line.strip())
                elif capture and any(k in line for k in _stop_keywords):
                    break
            if lines:
                decision_info["recommendations"] = " ".join(lines)

        # Parse OCR_MATCH
        if "OCR_MATCH: FALSE" in llm_output.upper():
            decision_info["ocr_match"] = False

        # Parse CONFIDENCE into a numeric score
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
        for label, score in confidence_map.items():
            if f"CONFIDENCE: {label}" in llm_output:
                decision_info["confidence_score"] = score
                break

        return decision_info

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
            factors_lines = [l for l in llm_output.split("\n") if "FACTORS:" in l]
            if factors_lines:
                text = factors_lines[0].replace("FACTORS:", "").strip()
                return [f.strip() for f in text.split(",") if f.strip()]
        return ["policy_limits", "ocr_validation", "duplicate_check", "monthly_spend"]

    def _extract_ocr_from_steps(self, intermediate_steps: list) -> Optional[OCRExtractionResult]:
        """Pull the OCR result out of the agent's intermediate tool steps."""
        for (tool_name, _, result) in intermediate_steps:
            if tool_name == "validate_receipt_ocr":
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                    return OCRExtractionResult(
                        raw_text=data.get("ocr_raw_text", ""),
                        vendor=data.get("ocr_vendor"),
                        amount=data.get("ocr_amount"),
                        date=data.get("ocr_date"),
                        vendor_confidence=data.get("vendor_confidence", 0.0),
                        amount_confidence=data.get("amount_confidence", 0.0),
                        date_confidence=data.get("date_confidence", 0.0),
                        overall_confidence=data.get("overall_confidence", 0.0),
                    )
                except Exception as e:
                    self.logger.warning(f"Could not parse OCR result from intermediate steps: {e}")
        return None

    def _build_message(self, decision: str) -> str:
        return {
            "AUTO_APPROVE": "Expense claim automatically approved",
            "ESCALATE": "Expense claim submitted for manager review",
            "REJECT": "Expense claim rejected",
        }.get(decision, "Expense claim processed")

    async def _create_approval_request(
        self,
        expense_id: int,
        input_data: ExpenseSubmitInput,
        workflow_id: str,
        llm_reasoning: str,
        ocr_result: Optional[OCRExtractionResult],
    ) -> None:
        """Create an entry in the approval queue for manager review."""
        triggers = []
        if input_data.amount > 100:
            triggers.append(EscalationTrigger.HIGH_VALUE)
        if ocr_result and ocr_result.overall_confidence < 0.7:
            triggers.append(EscalationTrigger.POLICY_VIOLATION)

        await approval_manager.create_approval_request(
            workflow_id=workflow_id,
            agent_name=self.name,
            request_type="expense_claim",
            requester_id=input_data.user_id,
            title=(
                f"Expense Claim - {input_data.category.replace('_', ' ').title()} "
                f"(${input_data.amount:.2f})"
            ),
            description=(
                f"Employee submitted ${input_data.amount:.2f} {input_data.category} expense "
                f"at {input_data.vendor or 'unknown vendor'} on {input_data.date}.\n\n"
                f"LLM Analysis:\n{llm_reasoning}\n\n"
                + (
                    f"OCR Confidence: {ocr_result.overall_confidence:.0%}"
                    if ocr_result
                    else "No receipt uploaded."
                )
            ),
            data={
                "expense_id": expense_id,
                "user_id": input_data.user_id,
                "category": input_data.category,
                "amount": input_data.amount,
                "vendor": input_data.vendor,
                "date": input_data.date,
                "has_receipt": bool(input_data.receipt_path),
                "ocr_confidence": ocr_result.overall_confidence if ocr_result else None,
                "llm_reasoning": llm_reasoning,
            },
            triggers=triggers,
            priority="high" if input_data.amount > 100 else "normal",
        )
        self.logger.info(f"Created approval request for expense {expense_id}")

    def get_input_schema(self) -> type[ExpenseSubmitInput]:
        return ExpenseSubmitInput

    def get_output_schema(self) -> type[ExpenseValidationOutput]:
        return ExpenseValidationOutput


# Agent is registered via agents/register_agents.py on application startup
