from typing import Any, Dict, List, Optional, Union
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from .state import WorkflowState
from .state_manager import state_manager


class ReasoningType(str, Enum):
    
    INTENT_ANALYSIS = "intent_analysis"
    DATA_EXTRACTION = "data_extraction"
    RULE_EVALUATION = "rule_evaluation"
    POLICY_CHECK = "policy_check"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_MAKING = "decision_making"
    ESCALATION_CHECK = "escalation_check"
    APPROVAL_ROUTING = "approval_routing"
    OUTCOME_GENERATION = "outcome_generation"


class ConfidenceLevel(str, Enum):
    
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ReasoningStep:
    
    step_id: str
    reasoning_type: ReasoningType
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: ConfidenceLevel
    reasoning: str
    factors: List[str]
    alternatives_considered: List[str]
    timestamp: datetime
    agent_name: str
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        
        data = asdict(self)
        data["reasoning_type"] = self.reasoning_type.value
        data["confidence"] = self.confidence.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        
        data_copy = data.copy()
        data_copy["reasoning_type"] = ReasoningType(data["reasoning_type"])
        data_copy["confidence"] = ConfidenceLevel(data["confidence"])
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "metadata" not in data_copy:
            data_copy["metadata"] = {}
        return cls(**data_copy)


@dataclass
class DecisionExplanation:
    
    decision_id: str
    workflow_id: str
    agent_name: str
    final_decision: str
    confidence_score: float
    reasoning_chain: List[ReasoningStep]
    key_factors: List[str]
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    policy_references: List[str]
    human_readable_summary: str
    technical_details: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        
        data = asdict(self)
        data["reasoning_chain"] = [step.to_dict() for step in self.reasoning_chain]
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionExplanation":
        
        data_copy = data.copy()
        data_copy["reasoning_chain"] = [ReasoningStep.from_dict(step) for step in data["reasoning_chain"]]
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data_copy)


class ExplanationGenerator:
    

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_explanation(self, reasoning_chain: List[ReasoningStep],
                           final_decision: str, confidence_score: float) -> str:
        
        if not reasoning_chain:
            return f"Decision: {final_decision} (Confidence: {confidence_score:.1%})"

        explanation_parts = []
        explanation_parts.append(f"**Decision: {final_decision}**")
        explanation_parts.append(f"**Overall Confidence: {confidence_score:.1%}**")
        explanation_parts.append("")

        
        steps_by_type = {}
        for step in reasoning_chain:
            if step.reasoning_type not in steps_by_type:
                steps_by_type[step.reasoning_type] = []
            steps_by_type[step.reasoning_type].append(step)

        
        for reasoning_type, steps in steps_by_type.items():
            explanation_parts.append(f"### {reasoning_type.value.replace('_', ' ').title()}")
            for step in steps:
                explanation_parts.append(f"**{step.description}**")
                explanation_parts.append(f"Confidence: {step.confidence.value}")
                explanation_parts.append(f"Reasoning: {step.reasoning}")

                if step.factors:
                    explanation_parts.append("Key Factors:")
                    for factor in step.factors:
                        explanation_parts.append(f"- {factor}")

                if step.alternatives_considered:
                    explanation_parts.append("Alternatives Considered:")
                    for alt in step.alternatives_considered:
                        explanation_parts.append(f"- {alt}")

                explanation_parts.append("")

        return "\n".join(explanation_parts)

    def generate_summary(self, reasoning_chain: List[ReasoningStep],
                        final_decision: str, max_length: int = 200) -> str:
        
        if not reasoning_chain:
            return f"Decision: {final_decision}"

        
        key_factors = []
        confidence_levels = []

        for step in reasoning_chain:
            confidence_levels.append(step.confidence.value)
            if step.factors:
                key_factors.extend(step.factors[:2]) 

        
        avg_confidence = self._calculate_average_confidence(confidence_levels)

        
        summary = f"Decision: {final_decision}. "
        summary += f"Confidence: {avg_confidence}. "

        if key_factors:
            summary += f"Key factors: {', '.join(key_factors[:3])}"

        
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."

        return summary

    def _calculate_average_confidence(self, confidence_levels: List[str]) -> str:
        
        level_values = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9
        }

        if not confidence_levels:
            return "unknown"

        avg_value = sum(level_values.get(level, 0.5) for level in confidence_levels) / len(confidence_levels)

        
        if avg_value >= 0.8:
            return "very_high"
        elif avg_value >= 0.6:
            return "high"
        elif avg_value >= 0.4:
            return "medium"
        elif avg_value >= 0.2:
            return "low"
        else:
            return "very_low"


class ReasoningTraceManager:
    

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.explanation_generator = ExplanationGenerator()
        self._reasoning_traces: Dict[str, List[ReasoningStep]] = {}
        self._decision_explanations: Dict[str, DecisionExplanation] = {}

    async def record_reasoning_step(self, step: ReasoningStep) -> None:
        
        workflow_id = step.workflow_id or "global"

        if workflow_id not in self._reasoning_traces:
            self._reasoning_traces[workflow_id] = []

        self._reasoning_traces[workflow_id].append(step)

        
        if step.workflow_id:
            try:
                workflow_state = await state_manager.get_workflow_state(step.workflow_id)
                if workflow_state:
                    reasoning_traces = workflow_state.metadata.get("reasoning_traces", [])
                    reasoning_traces.append(step.to_dict())
                    workflow_state.metadata["reasoning_traces"] = reasoning_traces
                    await state_manager.update_workflow_state(workflow_state)
            except ValueError:
                pass  

        self.logger.debug(f"Recorded reasoning step: {step.step_id} for workflow {workflow_id}")

    async def create_decision_explanation(self,
                                        decision_id: str,
                                        workflow_id: str,
                                        agent_name: str,
                                        final_decision: str,
                                        confidence_score: float,
                                        key_factors: List[str] = None,
                                        alternative_options: List[Dict[str, Any]] = None,
                                        risk_assessment: Dict[str, Any] = None,
                                        policy_references: List[str] = None) -> DecisionExplanation:
        
        reasoning_chain = self._reasoning_traces.get(workflow_id, [])

        
        human_readable_summary = self.explanation_generator.generate_summary(
            reasoning_chain, final_decision
        )

        
        detailed_explanation = self.explanation_generator.generate_explanation(
            reasoning_chain, final_decision, confidence_score
        )

        explanation = DecisionExplanation(
            decision_id=decision_id,
            workflow_id=workflow_id,
            agent_name=agent_name,
            final_decision=final_decision,
            confidence_score=confidence_score,
            reasoning_chain=reasoning_chain,
            key_factors=key_factors or [],
            alternative_options=alternative_options or [],
            risk_assessment=risk_assessment or {},
            policy_references=policy_references or [],
            human_readable_summary=human_readable_summary,
            technical_details={
                "detailed_explanation": detailed_explanation,
                "reasoning_steps_count": len(reasoning_chain),
                "confidence_distribution": self._analyze_confidence_distribution(reasoning_chain)
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

       
        self._decision_explanations[decision_id] = explanation

        
        try:
            workflow_state = await state_manager.get_workflow_state(workflow_id)
            if workflow_state:
                workflow_state.set("decision_explanation", explanation.to_dict())
                await state_manager.update_workflow_state(workflow_state)
        except ValueError:
            pass

        self.logger.info(f"Created decision explanation: {decision_id} for workflow {workflow_id}")
        return explanation

    def _analyze_confidence_distribution(self, reasoning_chain: List[ReasoningStep]) -> Dict[str, int]:
        
        distribution = {
            "very_low": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "very_high": 0
        }

        for step in reasoning_chain:
            distribution[step.confidence.value] += 1

        return distribution

    def get_reasoning_trace(self, workflow_id: str) -> List[ReasoningStep]:
       
        return self._reasoning_traces.get(workflow_id, [])

    def get_decision_explanation(self, decision_id: str) -> Optional[DecisionExplanation]:
        
        return self._decision_explanations.get(decision_id)

    def get_workflow_explanations(self, workflow_id: str) -> List[DecisionExplanation]:
        
        return [
            exp for exp in self._decision_explanations.values()
            if exp.workflow_id == workflow_id
        ]

    def search_explanations(self,
                          agent_name: Optional[str] = None,
                          decision_type: Optional[str] = None,
                          min_confidence: Optional[float] = None,
                          date_from: Optional[datetime] = None,
                          date_to: Optional[datetime] = None) -> List[DecisionExplanation]:
        
        results = list(self._decision_explanations.values())

        if agent_name:
            results = [exp for exp in results if exp.agent_name == agent_name]

        if decision_type:
            results = [exp for exp in results if decision_type.lower() in exp.final_decision.lower()]

        if min_confidence is not None:
            results = [exp for exp in results if exp.confidence_score >= min_confidence]

        if date_from:
            results = [exp for exp in results if exp.created_at >= date_from]

        if date_to:
            results = [exp for exp in results if exp.created_at <= date_to]

        return results

    def get_explanation_stats(self) -> Dict[str, Any]:
        
        if not self._decision_explanations:
            return {"total_explanations": 0}

        explanations = list(self._decision_explanations.values())

        total_explanations = len(explanations)
        avg_confidence = sum(exp.confidence_score for exp in explanations) / total_explanations

        agent_stats = {}
        for exp in explanations:
            if exp.agent_name not in agent_stats:
                agent_stats[exp.agent_name] = 0
            agent_stats[exp.agent_name] += 1

        decision_types = {}
        for exp in explanations:
            decision_type = exp.final_decision.split()[0]
            if decision_type not in decision_types:
                decision_types[decision_type] = 0
            decision_types[decision_type] += 1

        return {
            "total_explanations": total_explanations,
            "average_confidence": avg_confidence,
            "agent_distribution": agent_stats,
            "decision_type_distribution": decision_types,
            "total_reasoning_steps": sum(len(exp.reasoning_chain) for exp in explanations)
        }

    def clear_old_explanations(self, days_to_keep: int = 90) -> int:
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        to_remove = []

        for decision_id, explanation in self._decision_explanations.items():
            if explanation.created_at < cutoff_date:
                to_remove.append(decision_id)

        for decision_id in to_remove:
            del self._decision_explanations[decision_id]

        self.logger.info(f"Cleared {len(to_remove)} old explanations")
        return len(to_remove)



reasoning_manager = ReasoningTraceManager()