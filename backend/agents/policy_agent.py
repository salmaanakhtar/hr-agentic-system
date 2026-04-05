"""
Policy Compliance Agent - LangChain Powered

Provides two modes:
- Query (Q&A): RAG-based free-text question answering grounded in embedded policy documents
- Compliance check: Evaluates a scenario against policy, returning COMPLIANT | NON_COMPLIANT | UNCLEAR

Tools:
  search_policy_chunks   - pgvector cosine similarity search over embedded policy chunks
  get_policy_document    - retrieve a policy document's metadata and chunk count

Decision criteria (compliance mode):
  COMPLIANT     : Scenario clearly aligns with the retrieved policy rules
  NON_COMPLIANT : Scenario clearly violates one or more retrieved policy rules
  UNCLEAR       : Insufficient policy coverage or the scenario is genuinely ambiguous
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field
from sqlalchemy import Float as SQLFloat
from sqlalchemy import cast, select

from agents.base import Agent
from agents.reasoning_system import (
    ConfidenceLevel,
    ReasoningStep,
    ReasoningType,
    reasoning_manager,
)
from agents.schemas import (
    ComplianceCheckInput,
    ComplianceCheckOutput,
    PolicyQueryInput,
    PolicyQueryOutput,
)
from app.database import AsyncSessionLocal
from app.models import PolicyChunk, PolicyDocument


# ---------------------------------------------------------------------------
# Tool Input Schemas (Pydantic V1 required for LangChain 0.1.x)
# ---------------------------------------------------------------------------

class SearchPolicyChunksInput(BaseModelV1):
    query: str = Field(description="Free-text query to search for in policy documents")
    top_k: int = Field(
        default=5,
        description="Number of most relevant chunks to return (1-10)",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional policy category filter: "
            "general, leave, expense, hiring, payroll, "
            "code_of_conduct, safety, other"
        ),
    )


class GetPolicyDocumentInput(BaseModelV1):
    document_id: int = Field(description="ID of the policy document to retrieve")


# ---------------------------------------------------------------------------
# Policy Compliance Agent
# ---------------------------------------------------------------------------

class PolicyComplianceAgent(Agent[PolicyQueryInput, PolicyQueryOutput]):
    """
    LangChain-powered agent for policy Q&A and compliance checking.

    Uses GPT-4o-mini to orchestrate RAG searches over pgvector-embedded policy
    chunks and produce grounded answers or COMPLIANT/NON_COMPLIANT/UNCLEAR
    decisions with citations.
    """

    def __init__(self):
        super().__init__(
            name="policy_compliance_agent",
            description=(
                "LangChain-powered agent that answers policy questions using RAG "
                "over embedded policy documents and evaluates scenarios for compliance, "
                "producing COMPLIANT/NON_COMPLIANT/UNCLEAR decisions with citations"
            ),
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.tools = [
            StructuredTool.from_function(
                coroutine=self._tool_search_policy_chunks,
                name="search_policy_chunks",
                description=(
                    "Search for relevant policy chunks using semantic similarity. "
                    "Embeds the query and finds the most relevant chunks from active policy documents. "
                    "Returns chunk content, document title, category, and similarity scores. "
                    "ALWAYS call this first to ground your answer in policy documents. "
                    "Call multiple times with different queries if the first result is insufficient."
                ),
                args_schema=SearchPolicyChunksInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_get_policy_document,
                name="get_policy_document",
                description=(
                    "Retrieve a policy document's metadata by ID. "
                    "Returns document title, category, description, and chunk count. "
                    "Use this to get more context about a document found via search_policy_chunks."
                ),
                args_schema=GetPolicyDocumentInput,
                return_direct=False,
            ),
        ]

        self.query_system_prompt = """You are a Policy Q&A Agent for an HR management system.

Your role is to answer employee questions about HR policies using the policy documents stored in the system.

TOOLS AVAILABLE:
1. search_policy_chunks  - Search for relevant policy chunks using semantic similarity
2. get_policy_document   - Retrieve full metadata for a specific policy document

PROCESS:
1. Call search_policy_chunks with the user's question to find relevant policy text
2. If needed, search again with a rephrased query to find more specific information
3. Optionally call get_policy_document for additional document context
4. Synthesise a clear, grounded answer based only on the retrieved policy content

KEY RULES:
- Base your answer only on retrieved policy content — do not invent policy rules
- If no relevant chunks are found, say so clearly and suggest contacting HR
- Always cite which documents your answer comes from
- Keep answers concise and practical

OUTPUT FORMAT (use this exact structure):
ANSWER: [Clear, direct answer to the question based on retrieved policy content]
REASONING: [2-3 sentences explaining how you arrived at this answer and what policy text supports it]
CONFIDENCE: [HIGH | MEDIUM | LOW]
SOURCES: [comma-separated list of document titles cited]"""

        self.compliance_system_prompt = """You are a Policy Compliance Agent for an HR management system.

Your role is to evaluate whether a described scenario or action complies with HR policies.

TOOLS AVAILABLE:
1. search_policy_chunks  - Search for relevant policy chunks using semantic similarity
2. get_policy_document   - Retrieve full metadata for a specific policy document

PROCESS:
1. Call search_policy_chunks to find policy text relevant to the scenario
2. Search again with different angles if the first search is insufficient
3. Evaluate the scenario against the retrieved policy rules
4. Make a clear COMPLIANT / NON_COMPLIANT / UNCLEAR decision

DECISION CRITERIA:
- COMPLIANT     : Scenario clearly aligns with the retrieved policy rules
- NON_COMPLIANT : Scenario clearly violates one or more retrieved policy rules
- UNCLEAR       : Retrieved policy does not cover this scenario, or rules are ambiguous

KEY RULES:
- Base your decision only on retrieved policy content — do not invent rules
- If no relevant policy is found, use UNCLEAR and recommend HR review
- Be specific about which policy rule is violated or satisfied
- Always cite which documents your decision is based on

OUTPUT FORMAT (use this exact structure):
DECISION: [COMPLIANT | NON_COMPLIANT | UNCLEAR]
REASONING: [2-3 sentences explaining which policy rules apply and why the decision was made]
CONFIDENCE: [HIGH | MEDIUM | LOW]
POLICY_REFERENCES: [comma-separated list of policy document titles referenced]
RECOMMENDATIONS: [any follow-up actions, e.g. seek manager approval; contact HR]"""

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.logger.info("PolicyComplianceAgent initialized with LangChain + GPT-4o-mini")

    # -------------------------------------------------------------------------
    # Agent Loop
    # -------------------------------------------------------------------------

    async def _run_agent_loop(
        self, system_prompt: str, user_input: str, max_iterations: int = 5
    ) -> tuple[str, list]:
        """Run the LangChain tool-calling agent loop."""
        messages = [
            {"role": "system", "content": system_prompt},
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

        return "ERROR: Max iterations reached without final answer", intermediate_steps

    # -------------------------------------------------------------------------
    # Execute Methods
    # -------------------------------------------------------------------------

    async def execute(self, input_data: PolicyQueryInput) -> PolicyQueryOutput:
        """Primary execute method — delegates to query mode."""
        return await self.query(input_data)

    async def query(self, input_data: PolicyQueryInput) -> PolicyQueryOutput:
        """
        Answer a free-text policy question using RAG over embedded policy chunks.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(
            f"Policy query for user {input_data.user_id}, "
            f"question: {input_data.question[:80]}, workflow {workflow_id}"
        )

        try:
            llm_input = (
                f"Please answer the following HR policy question:\n\n"
                f"QUESTION: {input_data.question}\n\n"
                "Search the policy documents for relevant information and provide a grounded answer with citations."
            )

            llm_output, intermediate_steps = await self._run_agent_loop(
                self.query_system_prompt, llm_input
            )
            self.logger.info(f"LLM query output: {llm_output[:200]}...")

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered policy Q&A",
                    input_data={
                        "user_id": input_data.user_id,
                        "question": input_data.question,
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps),
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=["rag_search", "policy_grounded"],
                    alternatives_considered=[],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            parsed = self._parse_query_output(llm_output)
            citations = self._extract_citations(intermediate_steps)

            return PolicyQueryOutput(
                success=True,
                message="Policy question answered",
                reasoning=parsed["reasoning"],
                answer=parsed["answer"],
                citations=citations,
                confidence=parsed["confidence_score"],
                sources=parsed["sources"],
            )

        except Exception as e:
            self.logger.error(f"Policy query error: {e}", exc_info=True)
            return PolicyQueryOutput(
                success=False,
                message=f"Policy query failed: {str(e)}",
                reasoning=f"System error: {str(e)}",
                answer="Unable to answer due to a system error. Please contact HR directly.",
                citations=[],
                confidence=0.0,
                sources=[],
            )

    async def check_compliance(
        self, input_data: ComplianceCheckInput
    ) -> ComplianceCheckOutput:
        """
        Evaluate whether a described scenario complies with HR policies.
        Returns COMPLIANT | NON_COMPLIANT | UNCLEAR with reasoning and citations.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(
            f"Compliance check for user {input_data.user_id}, "
            f"scenario: {input_data.scenario[:80]}, workflow {workflow_id}"
        )

        try:
            context_block = (
                f"\nADDITIONAL CONTEXT: {json.dumps(input_data.context)}"
                if input_data.context
                else ""
            )
            llm_input = (
                f"Please evaluate whether the following scenario complies with HR policy:\n\n"
                f"SCENARIO: {input_data.scenario}"
                f"{context_block}\n\n"
                "Search for relevant policy rules and make a COMPLIANT / NON_COMPLIANT / UNCLEAR decision."
            )

            llm_output, intermediate_steps = await self._run_agent_loop(
                self.compliance_system_prompt, llm_input
            )
            self.logger.info(f"LLM compliance output: {llm_output[:200]}...")

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered compliance check",
                    input_data={
                        "user_id": input_data.user_id,
                        "scenario": input_data.scenario,
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps),
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=["policy_retrieved", "scenario_evaluated"],
                    alternatives_considered=["compliant", "non_compliant", "unclear"],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            parsed = self._parse_compliance_output(llm_output)
            citations = self._extract_citations(intermediate_steps)

            return ComplianceCheckOutput(
                success=True,
                message=self._build_compliance_message(parsed["decision"]),
                reasoning=parsed["reasoning"],
                decision=parsed["decision"],
                confidence=parsed["confidence_score"],
                citations=citations,
                policy_references=parsed["policy_references"],
                recommendations=parsed["recommendations"],
            )

        except Exception as e:
            self.logger.error(f"Compliance check error: {e}", exc_info=True)

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="Compliance check failed",
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

            return ComplianceCheckOutput(
                success=False,
                message=f"Compliance check failed: {str(e)}",
                reasoning=f"System error: {str(e)}",
                decision="UNCLEAR",
                confidence=0.0,
                citations=[],
                policy_references=[],
                recommendations=["Contact HR directly for policy guidance"],
            )

    # -------------------------------------------------------------------------
    # Tool Functions (called autonomously by the LLM)
    # -------------------------------------------------------------------------

    async def _tool_search_policy_chunks(
        self, query: str, top_k: int = 5, category: Optional[str] = None
    ) -> str:
        """Embed the query and run pgvector cosine similarity search over policy chunks."""
        try:
            top_k = max(1, min(10, top_k))

            embed_response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=query,
                model="text-embedding-3-small",
            )
            query_embedding = embed_response.data[0].embedding

            async with AsyncSessionLocal() as db:
                distance_expr = cast(
                    PolicyChunk.embedding.op("<=>")(query_embedding),
                    SQLFloat,
                ).label("distance")
                stmt = (
                    select(
                        PolicyChunk.id,
                        PolicyChunk.document_id,
                        PolicyChunk.chunk_index,
                        PolicyChunk.content,
                        PolicyDocument.title.label("doc_title"),
                        PolicyDocument.category.label("doc_category"),
                        distance_expr,
                    )
                    .join(PolicyDocument, PolicyChunk.document_id == PolicyDocument.id)
                    .where(PolicyDocument.is_active == True)
                    .where(PolicyChunk.embedding != None)
                )
                if category:
                    stmt = stmt.where(PolicyDocument.category == category)

                stmt = stmt.order_by(distance_expr).limit(top_k)
                result = await db.execute(stmt)
                rows = result.all()

                chunks_data = []
                for chunk_id, document_id, chunk_index, content, doc_title, doc_category, distance in rows:
                    similarity = round(1.0 - float(distance), 4)
                    chunks_data.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "document_title": doc_title,
                        "document_category": doc_category,
                        "chunk_index": chunk_index,
                        "content": content,
                        "similarity_score": similarity,
                    })

            if not chunks_data:
                return json.dumps({
                    "success": True,
                    "chunks": [],
                    "total_found": 0,
                    "message": (
                        "No policy chunks found matching this query. "
                        "The policy database may be empty or no relevant documents exist. "
                        "Recommend contacting HR directly."
                    ),
                })

            return json.dumps({
                "success": True,
                "chunks": chunks_data,
                "total_found": len(chunks_data),
                "message": (
                    f"Found {len(chunks_data)} relevant policy chunk(s) for query: '{query}'. "
                    f"Top result from '{chunks_data[0]['document_title']}' "
                    f"(similarity: {chunks_data[0]['similarity_score']:.3f})."
                ),
            })

        except Exception as e:
            self.logger.error(f"search_policy_chunks tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Policy search failed: {str(e)}",
            })

    async def _tool_get_policy_document(self, document_id: int) -> str:
        """Retrieve a policy document's metadata and chunk count."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(PolicyDocument).where(PolicyDocument.id == document_id)
                )
                doc = result.scalar_one_or_none()

                if not doc:
                    return json.dumps({
                        "success": False,
                        "message": f"Policy document {document_id} not found.",
                    })

                chunk_result = await db.execute(
                    select(PolicyChunk).where(PolicyChunk.document_id == document_id)
                )
                chunk_count = len(chunk_result.scalars().all())

                return json.dumps({
                    "success": True,
                    "document_id": doc.id,
                    "title": doc.title,
                    "description": doc.description,
                    "category": doc.category,
                    "filename": doc.filename,
                    "chunk_count": chunk_count,
                    "is_active": doc.is_active,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "message": (
                        f"Policy document: '{doc.title}' (category: {doc.category}). "
                        f"Description: {doc.description or 'None'}. "
                        f"Contains {chunk_count} embedded chunk(s)."
                    ),
                })

        except Exception as e:
            self.logger.error(f"get_policy_document tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve policy document: {str(e)}",
            })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_query_output(self, llm_output: str) -> Dict[str, Any]:
        """Parse the structured query output fields from the LLM response."""
        _stop = ["ANSWER:", "REASONING:", "CONFIDENCE:", "SOURCES:"]
        result: Dict[str, Any] = {
            "answer": llm_output,
            "reasoning": "",
            "confidence_score": 0.5,
            "sources": [],
        }

        if "ANSWER:" in llm_output:
            lines, capture = [], False
            for line in llm_output.split("\n"):
                if "ANSWER:" in line:
                    capture = True
                    lines.append(line.replace("ANSWER:", "").strip())
                elif capture and line.strip() and not any(k in line for k in _stop[1:]):
                    lines.append(line.strip())
                elif capture and any(k in line for k in _stop[1:]):
                    break
            if lines:
                result["answer"] = " ".join(lines)

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
                result["reasoning"] = " ".join(lines)

        if "SOURCES:" in llm_output:
            for line in llm_output.split("\n"):
                if "SOURCES:" in line:
                    text = line.replace("SOURCES:", "").strip()
                    result["sources"] = [s.strip() for s in text.split(",") if s.strip()]
                    break

        for label, score in {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.items():
            if f"CONFIDENCE: {label}" in llm_output:
                result["confidence_score"] = score
                break

        return result

    def _parse_compliance_output(self, llm_output: str) -> Dict[str, Any]:
        """Parse the structured compliance output fields from the LLM response."""
        _stop = ["DECISION:", "REASONING:", "CONFIDENCE:", "POLICY_REFERENCES:", "RECOMMENDATIONS:"]
        result: Dict[str, Any] = {
            "decision": "UNCLEAR",
            "reasoning": llm_output,
            "confidence_score": 0.5,
            "policy_references": [],
            "recommendations": [],
        }

        if "DECISION:" in llm_output:
            for line in llm_output.split("\n"):
                if "DECISION:" in line:
                    if "NON_COMPLIANT" in line:
                        result["decision"] = "NON_COMPLIANT"
                    elif "COMPLIANT" in line:
                        result["decision"] = "COMPLIANT"
                    else:
                        result["decision"] = "UNCLEAR"
                    break

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
                result["reasoning"] = " ".join(lines)

        if "POLICY_REFERENCES:" in llm_output:
            for line in llm_output.split("\n"):
                if "POLICY_REFERENCES:" in line:
                    text = line.replace("POLICY_REFERENCES:", "").strip()
                    result["policy_references"] = [s.strip() for s in text.split(",") if s.strip()]
                    break

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
                joined = " ".join(lines)
                result["recommendations"] = (
                    [r.strip() for r in joined.split(";") if r.strip()] or [joined]
                )

        for label, score in {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.items():
            if f"CONFIDENCE: {label}" in llm_output:
                result["confidence_score"] = score
                break

        return result

    def _extract_citations(self, intermediate_steps: list) -> List[Dict[str, Any]]:
        """Extract unique chunk citations from search_policy_chunks tool results."""
        seen_chunk_ids: set = set()
        citations: List[Dict[str, Any]] = []

        for tool_name, _, result in intermediate_steps:
            if tool_name == "search_policy_chunks":
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                    for chunk in data.get("chunks", []):
                        chunk_id = chunk.get("chunk_id")
                        if chunk_id and chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id)
                            citations.append({
                                "chunk_id": chunk_id,
                                "document_id": chunk.get("document_id"),
                                "document_title": chunk.get("document_title"),
                                "document_category": chunk.get("document_category"),
                                "chunk_index": chunk.get("chunk_index"),
                                "similarity_score": chunk.get("similarity_score"),
                                "excerpt": chunk.get("content", "")[:200],
                            })
                except Exception as e:
                    self.logger.warning(f"Could not parse citations from intermediate steps: {e}")

        return citations

    def _parse_confidence(self, llm_output: str) -> ConfidenceLevel:
        if "CONFIDENCE: HIGH" in llm_output:
            return ConfidenceLevel.HIGH
        elif "CONFIDENCE: MEDIUM" in llm_output:
            return ConfidenceLevel.MEDIUM
        elif "CONFIDENCE: LOW" in llm_output:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    def _build_compliance_message(self, decision: str) -> str:
        return {
            "COMPLIANT": "Scenario is compliant with HR policy",
            "NON_COMPLIANT": "Scenario violates one or more HR policy rules",
            "UNCLEAR": "Policy coverage is insufficient — HR review recommended",
        }.get(decision, "Compliance check completed")

    def get_input_schema(self) -> type[PolicyQueryInput]:
        return PolicyQueryInput

    def get_output_schema(self) -> type[PolicyQueryOutput]:
        return PolicyQueryOutput


# Agent is registered via agents/register_agents.py on application startup
