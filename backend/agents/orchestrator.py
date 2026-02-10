from typing import Any, Dict, List, Optional, Set, Tuple
import re
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .base import Agent
from .schemas import (
    BasicAgentInput, BasicAgentOutput,
    WorkflowAgentInput, MultiStepAgentOutput,
    ApprovalState
)
from .registry import registry
from .state import WorkflowState, WorkflowStatus, WorkflowStep
from .state_manager import state_manager
from .runner import agent_runner


class IntentType(str, Enum):
    
    LEAVE_REQUEST = "leave_request"
    EXPENSE_CLAIM = "expense_claim"
    HIRING_REQUEST = "hiring_request"
    PAYROLL_QUERY = "payroll_query"
    GENERAL_QUERY = "general_query"
    APPROVAL_REQUEST = "approval_request"
    STATUS_CHECK = "status_check"
    UNKNOWN = "unknown"


class ExecutionPriority(str, Enum):
    
    CRITICAL = "critical" 
    HIGH = "high"        
    NORMAL = "normal"     
    LOW = "low"         
    BACKGROUND = "background"  


@dataclass
class AgentExecutionPlan:
    
    intent_type: IntentType
    primary_agent: str
    supporting_agents: List[str]
    execution_order: List[str]
    dependencies: Dict[str, Set[str]]  
    priority: ExecutionPriority
    estimated_duration: int  
    requires_approval: bool
    approval_triggers: List[str]
    workflow_type: str


@dataclass
class IntentAnalysis:
    
    intent_type: IntentType
    confidence: float
    extracted_data: Dict[str, Any]
    keywords: List[str]
    context_hints: List[str]
    reasoning: str


class IntentRecognizer:
   

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")


        self.intent_patterns = {
            IntentType.LEAVE_REQUEST: {
                "keywords": ["leave", "vacation", "holiday", "time off", "pto", "sick leave", "maternity", "paternity"],
                "patterns": [
                    r"request.*leave", r"apply.*leave", r"take.*leave",
                    r"book.*holiday", r"schedule.*time.?off"
                ],
                "context": ["start_date", "end_date", "leave_type", "reason"]
            },
            IntentType.EXPENSE_CLAIM: {
                "keywords": ["expense", "reimbursement", "receipt", "claim", "cost", "spend", "budget"],
                "patterns": [
                    r"submit.*expense", r"claim.*expense", r"reimburse",
                    r"expense.*report", r"cost.*claim"
                ],
                "context": ["amount", "category", "date", "description"]
            },
            IntentType.HIRING_REQUEST: {
                "keywords": ["hire", "recruit", "job", "position", "vacancy", "candidate", "interview"],
                "patterns": [
                    r"hire.*person", r"recruit.*candidate", r"fill.*position",
                    r"create.*job", r"post.*job"
                ],
                "context": ["position", "department", "requirements", "priority"]
            },
            IntentType.PAYROLL_QUERY: {
                "keywords": ["payroll", "salary", "pay", "wage", "payslip", "compensation"],
                "patterns": [
                    r"payroll.*query", r"salary.*information", r"pay.*slip",
                    r"compensation.*details"
                ],
                "context": ["period", "query_type"]
            },
            IntentType.APPROVAL_REQUEST: {
                "keywords": ["approve", "review", "pending", "decision", "authorize"],
                "patterns": [
                    r"approve.*request", r"review.*application", r"pending.*approval",
                    r"make.*decision", r"authorize"
                ],
                "context": ["request_type", "request_id"]
            },
            IntentType.STATUS_CHECK: {
                "keywords": ["status", "check", "progress", "update", "state"],
                "patterns": [
                    r"check.*status", r"request.*status", r"workflow.*status",
                    r"application.*status"
                ],
                "context": ["request_id", "workflow_id"]
            }
        }

    def analyze_intent(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> IntentAnalysis:
        
        user_input_lower = user_input.lower()
        context = context or {}

        best_match = IntentType.UNKNOWN
        best_confidence = 0.0
        extracted_data = {}
        keywords_found = []
        reasoning_parts = []


        for intent_type, config in self.intent_patterns.items():
            confidence = 0.0
            intent_keywords = []


            for keyword in config["keywords"]:
                if keyword in user_input_lower:
                    confidence += 0.3
                    intent_keywords.append(keyword)
                    keywords_found.append(keyword)


            for pattern in config["patterns"]:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    confidence += 0.5
                    reasoning_parts.append(f"Matched pattern: {pattern}")


            if context:
                for context_key in config["context"]:
                    if context_key in context:
                        confidence += 0.2
                        extracted_data[context_key] = context[context_key]

 
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = intent_type


        extracted_data.update(self._extract_entities(user_input))

        reasoning = f"Intent recognition: {best_match.value} (confidence: {best_confidence:.2f})"
        if reasoning_parts:
            reasoning += f" - {', '.join(reasoning_parts)}"
        if keywords_found:
            reasoning += f" - Keywords: {', '.join(keywords_found)}"

        return IntentAnalysis(
            intent_type=best_match,
            confidence=min(best_confidence, 1.0), 
            extracted_data=extracted_data,
            keywords=keywords_found,
            context_hints=list(context.keys()) if context else [],
            reasoning=reasoning
        )

    def _extract_entities(self, user_input: str) -> Dict[str, Any]:
        
        entities = {}

      
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 
            r'\b\d{4}-\d{2}-\d{2}\b',  
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, user_input)
            if dates:
                entities["dates"] = dates
                break

     
        amount_patterns = [
            r'\$?\d+(?:\.\d{2})?',  
            r'\d+(?:\.\d{2})?\s*(?:dollars?|usd|gbp|eur)',  
        ]
        for pattern in amount_patterns:
            amounts = re.findall(pattern, user_input, re.IGNORECASE)
            if amounts:
                entities["amounts"] = amounts
                break

        return entities


class AgentSelector:
    

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        
        self.agent_mappings = {
            IntentType.LEAVE_REQUEST: {
                "primary": "leave_request_agent",
                "supporting": ["calendar_agent", "policy_agent"],
                "workflow_type": "leave_request_workflow"
            },
            IntentType.EXPENSE_CLAIM: {
                "primary": "expense_claim_agent",
                "supporting": ["ocr_agent", "policy_agent", "audit_agent"],
                "workflow_type": "expense_claim_workflow"
            },
            IntentType.HIRING_REQUEST: {
                "primary": "hiring_agent",
                "supporting": ["cv_parser_agent", "ranking_agent"],
                "workflow_type": "hiring_workflow"
            },
            IntentType.PAYROLL_QUERY: {
                "primary": "payroll_agent",
                "supporting": ["database_agent"],
                "workflow_type": "payroll_query_workflow"
            },
            IntentType.APPROVAL_REQUEST: {
                "primary": "approval_agent",
                "supporting": ["notification_agent"],
                "workflow_type": "approval_workflow"
            },
            IntentType.STATUS_CHECK: {
                "primary": "status_agent",
                "supporting": ["database_agent"],
                "workflow_type": "status_check_workflow"
            }
        }

    def create_execution_plan(self, intent_analysis: IntentAnalysis,
                            user_context: Dict[str, Any]) -> AgentExecutionPlan:
        
        intent_type = intent_analysis.intent_type

        if intent_type not in self.agent_mappings:
            
            return AgentExecutionPlan(
                intent_type=IntentType.GENERAL_QUERY,
                primary_agent="general_query_agent",
                supporting_agents=[],
                execution_order=["general_query_agent"],
                dependencies={},
                priority=ExecutionPriority.NORMAL,
                estimated_duration=30,
                requires_approval=False,
                approval_triggers=[],
                workflow_type="general_query_workflow"
            )

        mapping = self.agent_mappings[intent_type]

        
        execution_order, dependencies = self._determine_execution_order(
            mapping["primary"], mapping["supporting"]
        )

        
        requires_approval, approval_triggers = self._check_approval_requirements(
            intent_type, intent_analysis.extracted_data, user_context
        )

        
        priority = self._determine_priority(intent_type, intent_analysis.extracted_data)

       
        estimated_duration = self._estimate_duration(execution_order, requires_approval)

        return AgentExecutionPlan(
            intent_type=intent_type,
            primary_agent=mapping["primary"],
            supporting_agents=mapping["supporting"],
            execution_order=execution_order,
            dependencies=dependencies,
            priority=priority,
            estimated_duration=estimated_duration,
            requires_approval=requires_approval,
            approval_triggers=approval_triggers,
            workflow_type=mapping["workflow_type"]
        )

    def _determine_execution_order(self, primary: str, supporting: List[str]) -> Tuple[List[str], Dict[str, Set[str]]]:
        
        execution_order = [primary]  
        dependencies = {primary: set()}  

        
        for agent in supporting:
            execution_order.append(agent)
            dependencies[agent] = {primary}

        return execution_order, dependencies

    def _check_approval_requirements(self, intent_type: IntentType,
                                   extracted_data: Dict[str, Any],
                                   user_context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        
        requires_approval = False
        triggers = []

        if intent_type == IntentType.LEAVE_REQUEST:
            
            if "dates" in extracted_data:
                
                requires_approval = True
                triggers.append("leave_duration_check")

        elif intent_type == IntentType.EXPENSE_CLAIM:
            
            if "amounts" in extracted_data:
                try:
                    amount = float(extracted_data["amounts"][0].replace('$', ''))
                    if amount > 500:  
                        requires_approval = True
                        triggers.append("high_value_expense")
                except (ValueError, IndexError):
                    pass

        elif intent_type == IntentType.HIRING_REQUEST:
            
            requires_approval = True
            triggers.append("hiring_approval_required")

        
        user_role = user_context.get("role", "employee")
        if user_role == "manager" and intent_type in [IntentType.LEAVE_REQUEST]:
            requires_approval = False
            triggers = []

        return requires_approval, triggers

    def _determine_priority(self, intent_type: IntentType, extracted_data: Dict[str, Any]) -> ExecutionPriority:
        
        if intent_type == IntentType.HIRING_REQUEST:
            return ExecutionPriority.HIGH
        elif intent_type == IntentType.EXPENSE_CLAIM:
            
            if extracted_data.get("urgent", False):
                return ExecutionPriority.HIGH
            return ExecutionPriority.NORMAL
        else:
            return ExecutionPriority.NORMAL

    def _estimate_duration(self, execution_order: List[str], requires_approval: bool) -> int:
        
        base_duration = len(execution_order) * 30 
        if requires_approval:
            base_duration += 3600  
        return base_duration


class OrchestratorAgent(Agent[BasicAgentInput, MultiStepAgentOutput]):
    

    def __init__(self):
        super().__init__(
            name="orchestrator_agent",
            description="Central coordinator for multi-agent workflows"
        )

        self.intent_recognizer = IntentRecognizer()
        self.agent_selector = AgentSelector()

    async def execute(self, input_data: BasicAgentInput) -> MultiStepAgentOutput:
        
        try:
            
            user_query = input_data.data.get("query", "")
            if not user_query:
                return self.create_output(
                    success=False,
                    message="No query provided",
                    reasoning="Orchestrator requires a query to determine intent",
                    current_step="intent_analysis",
                    next_steps=[],
                    completed_steps=[],
                    workflow_data={}
                )

            
            intent_analysis = self.intent_recognizer.analyze_intent(
                user_query,
                input_data.data
            )

            self.logger.info(f"Intent analysis: {intent_analysis.intent_type.value} "
                           f"(confidence: {intent_analysis.confidence:.2f})")

            
            user_context = {
                "user_id": input_data.user_id,
                "role": input_data.data.get("user_role", "employee")
            }

            execution_plan = self.agent_selector.create_execution_plan(
                intent_analysis, user_context
            )

            
            workflow_state = await self._create_workflow_state(
                input_data, intent_analysis, execution_plan
            )

            
            next_steps = await self._determine_next_steps(execution_plan, workflow_state)

            return MultiStepAgentOutput(
                success=True,
                message=f"Intent recognized: {intent_analysis.intent_type.value}. "
                       f"Created workflow with {len(execution_plan.execution_order)} agents.",
                reasoning=intent_analysis.reasoning,
                current_step="orchestration",
                next_steps=next_steps,
                completed_steps=["intent_analysis"],
                workflow_data={
                    "intent_type": intent_analysis.intent_type.value,
                    "confidence": intent_analysis.confidence,
                    "execution_plan": {
                        "primary_agent": execution_plan.primary_agent,
                        "supporting_agents": execution_plan.supporting_agents,
                        "execution_order": execution_plan.execution_order,
                        "requires_approval": execution_plan.requires_approval,
                        "priority": execution_plan.priority.value,
                        "estimated_duration": execution_plan.estimated_duration
                    },
                    "extracted_data": intent_analysis.extracted_data,
                    "workflow_id": workflow_state.workflow_id if workflow_state else None
                }
            )

        except Exception as e:
            self.logger.error(f"Orchestrator execution failed: {e}", exc_info=True)
            return MultiStepAgentOutput(
                success=False,
                message=f"Orchestration failed: {str(e)}",
                reasoning="Unexpected error during orchestration",
                current_step="error",
                next_steps=[],
                completed_steps=[],
                workflow_data={"error": str(e)}
            )

    async def _create_workflow_state(self, input_data: BasicAgentInput,
                                   intent_analysis: IntentAnalysis,
                                   execution_plan: AgentExecutionPlan) -> Optional[WorkflowState]:
        
        try:
            workflow_state = await state_manager.create_workflow_state(
                workflow_type=execution_plan.workflow_type,
                user_id=input_data.user_id,
                initial_data={
                    "intent_analysis": {
                        "intent_type": intent_analysis.intent_type.value,
                        "confidence": intent_analysis.confidence,
                        "extracted_data": intent_analysis.extracted_data,
                        "keywords": intent_analysis.keywords,
                        "reasoning": intent_analysis.reasoning
                    },
                    "execution_plan": {
                        "primary_agent": execution_plan.primary_agent,
                        "supporting_agents": execution_plan.supporting_agents,
                        "execution_order": execution_plan.execution_order,
                        "dependencies": {k: list(v) for k, v in execution_plan.dependencies.items()},
                        "requires_approval": execution_plan.requires_approval,
                        "approval_triggers": execution_plan.approval_triggers,
                        "priority": execution_plan.priority.value,
                        "estimated_duration": execution_plan.estimated_duration
                    },
                    "orchestrator_metadata": {
                        "created_at": datetime.now().isoformat(),
                        "user_query": input_data.data.get("query", ""),
                        "orchestrator_version": "1.0"
                    }
                }
            )

            
            for agent_name in execution_plan.execution_order:
                step = WorkflowStep(
                    step_id=f"execute_{agent_name}",
                    step_name=f"Execute {agent_name}",
                    agent_name=agent_name,
                    status="pending",
                    dependencies=list(execution_plan.dependencies.get(agent_name, []))
                )
                workflow_state.steps.append(step)

            
            await state_manager.update_workflow_state(workflow_state)

            return workflow_state

        except Exception as e:
            self.logger.error(f"Failed to create workflow state: {e}")
            return None

    async def _determine_next_steps(self, execution_plan: AgentExecutionPlan,
                                  workflow_state: Optional[WorkflowState]) -> List[str]:
        
        if not workflow_state:
            return []

        next_steps = []

        
        for step in workflow_state.steps:
            if step.status == "pending":
                
                deps_completed = all(
                    any(s.step_id == f"execute_{dep}" and s.status == "completed"
                        for s in workflow_state.steps)
                    for dep in step.dependencies
                )

                if deps_completed:
                    next_steps.append(step.step_id)

        return next_steps

    def get_input_schema(self):
        return BasicAgentInput

    def get_output_schema(self):
        return MultiStepAgentOutput