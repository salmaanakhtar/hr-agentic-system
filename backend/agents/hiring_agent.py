"""
Hiring Pipeline Agent - LangChain Powered

Evaluates candidate applications against job postings using:
- CV parsing (pypdf + regex extraction)
- Semantic embeddings (OpenAI text-embedding-3-small, 1536-dim)
- Cosine similarity computed with numpy (embeddings loaded from pgvector)
- Skill coverage analysis (intersection of required vs candidate skills)
- SHORTLIST / REVIEW / PASS decisions with reasoning traces

Decision thresholds:
  SHORTLIST : similarity >= 0.75 AND skill_coverage >= 0.60
  REVIEW    : similarity >= 0.50 (but below SHORTLIST threshold)
  PASS      : similarity < 0.50
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import logging
import os
import json
import asyncio

import numpy as np
import openai
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel as BaseModelV1, Field  # LangChain 0.1.x uses Pydantic V1

from agents.base import Agent
from agents.schemas import HiringApplicationInput, HiringApplicationOutput
from agents.reasoning_system import reasoning_manager, ReasoningStep, ReasoningType, ConfidenceLevel
from app.database import AsyncSessionLocal
from app.models import JobPosting, Candidate, JobApplication, ApplicationStatus


# ---------------------------------------------------------------------------
# Tool Input Schemas (Pydantic V1 required for LangChain 0.1.x)
# ---------------------------------------------------------------------------

class ParseCVInput(BaseModelV1):
    candidate_id: int = Field(description="ID of the candidate whose CV to parse")
    cv_path: str = Field(description="Full filesystem path to the candidate's PDF CV file")


class GenerateEmbeddingInput(BaseModelV1):
    candidate_id: int = Field(description="ID of the candidate whose CV embedding to generate")


class GetJobRequirementsInput(BaseModelV1):
    job_id: int = Field(description="ID of the job posting to retrieve requirements for")


class SearchCandidatesInput(BaseModelV1):
    job_id: int = Field(description="ID of the job posting to match the candidate against")
    candidate_id: int = Field(description="ID of the candidate to score and rank")


class CheckDuplicateCandidateInput(BaseModelV1):
    candidate_id: int = Field(description="ID of the candidate to check for duplicate applications")
    job_id: int = Field(description="ID of the job posting to check duplicates for")
    current_application_id: int = Field(description="ID of the current application to exclude from the duplicate check")


# ---------------------------------------------------------------------------
# Hiring Agent
# ---------------------------------------------------------------------------

class HiringAgent(Agent[HiringApplicationInput, HiringApplicationOutput]):
    """
    LangChain-powered agent for evaluating candidate applications.

    Uses GPT-4o-mini to analyze CV data, semantic similarity, and skill coverage
    to make autonomous SHORTLIST/REVIEW/PASS decisions with natural language reasoning.
    """

    def __init__(self):
        super().__init__(
            name="hiring_agent",
            description=(
                "LangChain-powered agent that evaluates candidate applications using "
                "CV parsing, semantic embeddings, and GPT-4o-mini for autonomous "
                "SHORTLIST/REVIEW/PASS decisions"
            )
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self._openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.tools = [
            StructuredTool.from_function(
                coroutine=self._tool_parse_cv,
                name="parse_cv",
                description=(
                    "Parse the candidate's PDF CV to extract structured information: "
                    "skills, years of experience, education history, and current job title. "
                    "Saves extracted data to the candidate record. "
                    "ALWAYS call this first to get the candidate profile data."
                ),
                args_schema=ParseCVInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_generate_embedding,
                name="generate_embedding",
                description=(
                    "Generate a semantic embedding vector (OpenAI text-embedding-3-small, 1536-dim) "
                    "from the candidate's CV text and save it to the candidate record. "
                    "This embedding is required for semantic similarity matching. "
                    "Call this after parse_cv and before search_candidates_for_job."
                ),
                args_schema=GenerateEmbeddingInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_get_job_requirements,
                name="get_job_requirements",
                description=(
                    "Retrieve the full job posting details including title, description, "
                    "required skills list, experience requirements, department, and employment type. "
                    "Call this to understand what the role requires before making a decision."
                ),
                args_schema=GetJobRequirementsInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_search_candidates_for_job,
                name="search_candidates_for_job",
                description=(
                    "Compute the semantic similarity score between this candidate's CV embedding "
                    "and the job posting embedding, and calculate skill coverage (fraction of "
                    "required skills matched by this candidate). Returns similarity_score (0.0-1.0), "
                    "skill_coverage (0.0-1.0), matched/missing skills, and ranking context. "
                    "Call this after generate_embedding to get the scores needed for the decision."
                ),
                args_schema=SearchCandidatesInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                coroutine=self._tool_check_duplicate_candidate,
                name="check_duplicate_candidate",
                description=(
                    "Check whether this candidate already has an active application for this same job "
                    "(excluding the current application being evaluated). Returns a duplicate flag "
                    "and details of any existing application."
                ),
                args_schema=CheckDuplicateCandidateInput,
                return_direct=False,
            ),
        ]

        self.system_prompt = """You are a Hiring Pipeline Agent for an HR management system.

Your role is to evaluate candidate applications against job requirements and make
autonomous SHORTLIST/REVIEW/PASS decisions using CV parsing, semantic similarity,
and skill matching.

TOOLS AVAILABLE:
1. parse_cv                  - Extract skills, experience, education, and title from the candidate's CV
2. generate_embedding        - Generate a semantic embedding (required before similarity search)
3. get_job_requirements      - Retrieve job posting details and required skills
4. search_candidates_for_job - Compute semantic similarity and skill coverage scores
5. check_duplicate_candidate - Check for duplicate applications from this candidate

DECISION PROCESS:
1. Call parse_cv to extract structured profile data from the candidate's CV
2. Call generate_embedding to enable semantic matching
3. Call get_job_requirements to understand what the role requires
4. Call search_candidates_for_job to compute similarity score and skill coverage
5. Call check_duplicate_candidate to verify no duplicate submission exists
6. Analyze all results and make a clear final SHORTLIST/REVIEW/PASS decision

DECISION THRESHOLDS:
- SHORTLIST: similarity_score >= 0.75 AND skill_coverage >= 0.60
- REVIEW:    similarity_score >= 0.50 (but does not meet full SHORTLIST threshold)
- PASS:      similarity_score < 0.50

FACTORS THAT CAN UPGRADE A BORDERLINE DECISION TO SHORTLIST:
- Candidate experience exceeds job requirements by 2 or more years
- Strong title or role match to the job posting
- Additional highly relevant skills beyond the required set

FACTORS THAT CAN DOWNGRADE TO PASS:
- Duplicate application detected (same candidate, same job, active status)
- CV parsing failed or very low confidence (< 0.25)
- No CV embedding could be generated

OUTPUT FORMAT (use this exact structure):
DECISION: [SHORTLIST | REVIEW | PASS]
REASONING: [2-3 sentences explaining your analysis and the key factors in your decision]
CONFIDENCE: [HIGH | MEDIUM | LOW]
SIMILARITY_SCORE: [float 0.0-1.0 as returned by search_candidates_for_job]
SKILL_COVERAGE: [float 0.0-1.0 as returned by search_candidates_for_job]
FACTORS: [comma-separated list of key factors considered]
RECOMMENDATIONS: [specific next steps for the hiring manager, e.g. which skills to probe in interview]

Be thorough, call all required tools in order, and provide clear evidence-based reasoning."""

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.logger.info("Hiring Agent initialized with LangChain + GPT-4o-mini")

    # -------------------------------------------------------------------------
    # Agent Loop (same pattern as LeaveAgent / ExpenseAgent)
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

    async def execute(self, input_data: HiringApplicationInput) -> HiringApplicationOutput:
        """
        Execute candidate application evaluation using the LangChain agent.

        The LLM autonomously calls tools to parse the CV, generate embeddings,
        retrieve job requirements, and compute similarity scores before making
        a SHORTLIST/REVIEW/PASS decision.
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(
            f"Evaluating application {input_data.application_id} for job {input_data.job_id}, "
            f"candidate {input_data.candidate_id}, workflow {workflow_id}"
        )

        try:
            # Load candidate and job for the LLM prompt context
            async with AsyncSessionLocal() as db:
                cand_result = await db.execute(
                    select(Candidate).where(Candidate.id == input_data.candidate_id)
                )
                candidate = cand_result.scalar_one_or_none()

                job_result = await db.execute(
                    select(JobPosting).where(JobPosting.id == input_data.job_id)
                )
                job = job_result.scalar_one_or_none()

            if not candidate or not job:
                return HiringApplicationOutput(
                    success=False,
                    message="Candidate or job posting not found",
                    reasoning="Could not load candidate or job from the database",
                    decision="REVIEW",
                    confidence=0.0,
                    application_id=input_data.application_id,
                )

            cv_path = candidate.cv_path or ""

            llm_input = f"""Evaluate the following candidate application and make a SHORTLIST/REVIEW/PASS decision:

APPLICATION ID: {input_data.application_id}
JOB ID: {input_data.job_id}
CANDIDATE ID: {input_data.candidate_id}
CANDIDATE NAME: {candidate.full_name}
CANDIDATE EMAIL: {candidate.email}
CV FILE PATH: {cv_path}

Please use the available tools in this order:
1. Call parse_cv with candidate_id={input_data.candidate_id} and cv_path="{cv_path}" to extract structured profile data
2. Call generate_embedding with candidate_id={input_data.candidate_id} to create the semantic embedding
3. Call get_job_requirements with job_id={input_data.job_id} to understand what this role needs
4. Call search_candidates_for_job with job_id={input_data.job_id} and candidate_id={input_data.candidate_id} to get similarity and skill coverage scores
5. Call check_duplicate_candidate with candidate_id={input_data.candidate_id}, job_id={input_data.job_id}, and current_application_id={input_data.application_id}

Then make your final SHORTLIST/REVIEW/PASS decision following the criteria in your instructions.
"""

            self.logger.info("Invoking LLM agent for candidate application evaluation")
            llm_output, intermediate_steps = await self._run_agent_loop(llm_input)
            self.logger.info(f"LLM decision output: {llm_output[:200]}...")

            # Record reasoning trace
            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="LLM-powered candidate application evaluation",
                    input_data={
                        "application_id": input_data.application_id,
                        "job_id": input_data.job_id,
                        "candidate_id": input_data.candidate_id,
                    },
                    output_data={
                        "llm_output": llm_output,
                        "intermediate_steps": str(intermediate_steps),
                    },
                    confidence=self._parse_confidence(llm_output),
                    reasoning=llm_output,
                    factors=self._parse_factors(llm_output),
                    alternatives_considered=["shortlist", "review", "pass"],
                    agent_name=self.name,
                    workflow_id=workflow_id,
                )
            )

            decision_info = self._parse_llm_decision(llm_output)
            scores = self._extract_scores_from_steps(intermediate_steps)

            # Map LLM decision to application status
            status_map = {
                "SHORTLIST": ApplicationStatus.SHORTLISTED.value,
                "REVIEW": ApplicationStatus.APPLIED.value,   # stays in applied — manager reviews
                "PASS": ApplicationStatus.PASSED.value,
            }
            new_status = status_map.get(decision_info["decision"], ApplicationStatus.APPLIED.value)

            # Update application with decision, scores, and status
            async with AsyncSessionLocal() as db:
                app_result = await db.execute(
                    select(JobApplication).where(JobApplication.id == input_data.application_id)
                )
                application = app_result.scalar_one()

                application.llm_decision = decision_info["decision"]
                application.llm_reasoning = decision_info["reasoning"]
                application.status = new_status
                application.similarity_score = scores.get("similarity_score")
                application.skill_coverage = scores.get("skill_coverage")

                await db.flush()
                await db.commit()

            # Re-rank all applications for this job (rank 1 = highest similarity_score)
            await self._rerank_applications(input_data.job_id)

            return HiringApplicationOutput(
                success=decision_info["decision"] != "PASS",
                message=self._build_message(decision_info["decision"]),
                reasoning=decision_info["reasoning"],
                decision=decision_info["decision"],
                confidence=decision_info["confidence_score"],
                application_id=input_data.application_id,
                similarity_score=scores.get("similarity_score"),
                skill_coverage=scores.get("skill_coverage"),
                factors=self._parse_factors(llm_output),
                recommendations=(
                    [decision_info["recommendations"]]
                    if decision_info.get("recommendations")
                    else []
                ),
            )

        except Exception as e:
            self.logger.error(f"Error evaluating application: {e}", exc_info=True)

            await reasoning_manager.record_reasoning_step(
                ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    description="Candidate application evaluation failed",
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

            return HiringApplicationOutput(
                success=False,
                message=f"Failed to evaluate application: {str(e)}",
                reasoning=f"System error during evaluation: {str(e)}",
                decision="REVIEW",  # Fail-safe to manager review, never silently pass
                confidence=0.0,
                application_id=input_data.application_id,
                factors=["system_error"],
            )

    # -------------------------------------------------------------------------
    # Tool Functions (called autonomously by the LLM)
    # -------------------------------------------------------------------------

    async def _tool_parse_cv(self, candidate_id: int, cv_path: str) -> str:
        """Parse the candidate's CV and save extracted fields to the candidate record."""
        try:
            from agents.tools.cv_parser import parse_cv as _parse_cv

            # cv_parser is synchronous — run in thread pool
            parse_result = await asyncio.to_thread(_parse_cv, cv_path)

            # Persist extracted data to the candidate record
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Candidate).where(Candidate.id == candidate_id)
                )
                candidate = result.scalar_one_or_none()
                if candidate:
                    candidate.cv_text = parse_result.raw_text
                    candidate.skills = parse_result.skills
                    candidate.experience_years = parse_result.experience_years
                    candidate.education = parse_result.education
                    candidate.current_title = parse_result.current_title
                    await db.commit()

            return json.dumps({
                "success": True,
                "candidate_id": candidate_id,
                "skills": parse_result.skills,
                "skill_count": len(parse_result.skills),
                "experience_years": parse_result.experience_years,
                "education_count": len(parse_result.education),
                "current_title": parse_result.current_title,
                "parse_confidence": parse_result.parse_confidence,
                "raw_text_length": len(parse_result.raw_text),
                "message": (
                    f"CV parsed successfully. Found {len(parse_result.skills)} skills, "
                    f"{parse_result.experience_years or 0} years experience, "
                    f"{len(parse_result.education)} education entries. "
                    f"Parse confidence: {parse_result.parse_confidence:.0%}."
                    + (
                        " LOW CONFIDENCE - CV may be sparse or poorly formatted."
                        if parse_result.parse_confidence < 0.25
                        else ""
                    )
                ),
            })

        except Exception as e:
            self.logger.error(f"parse_cv tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"CV parsing failed: {str(e)}. Treat as low-confidence application.",
            })

    async def _tool_generate_embedding(self, candidate_id: int) -> str:
        """Generate OpenAI text-embedding-3-small from candidate's CV text and save to DB."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Candidate).where(Candidate.id == candidate_id)
                )
                candidate = result.scalar_one_or_none()
                if not candidate or not candidate.cv_text:
                    return json.dumps({
                        "success": False,
                        "message": "No CV text found for candidate. Run parse_cv first.",
                    })
                cv_text = candidate.cv_text

            # Truncate to stay well within the 8192-token model limit (~32k chars)
            text_for_embedding = cv_text[:32000]

            response = await self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text_for_embedding,
            )
            embedding = response.data[0].embedding  # list of 1536 floats

            # Persist embedding to the candidate record
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Candidate).where(Candidate.id == candidate_id)
                )
                candidate = result.scalar_one()
                candidate.cv_embedding = embedding
                await db.commit()

            return json.dumps({
                "success": True,
                "candidate_id": candidate_id,
                "embedding_dimensions": len(embedding),
                "message": (
                    f"Embedding generated and saved ({len(embedding)}-dim vector). "
                    "Ready for semantic similarity search."
                ),
            })

        except Exception as e:
            self.logger.error(f"generate_embedding tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Embedding generation failed: {str(e)}. Similarity search will not be available.",
            })

    async def _tool_get_job_requirements(self, job_id: int) -> str:
        """Fetch job posting details and requirements from the database."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(JobPosting).where(JobPosting.id == job_id)
                )
                job = result.scalar_one_or_none()

            if not job:
                return json.dumps({
                    "success": False,
                    "message": f"Job posting {job_id} not found.",
                })

            return json.dumps({
                "success": True,
                "job_id": job.id,
                "title": job.title,
                "department": job.department,
                "description": job.description[:500],       # Truncated for token efficiency
                "requirements": job.requirements[:500],
                "required_skills": job.required_skills or [],
                "required_skill_count": len(job.required_skills or []),
                "experience_years_required": job.experience_years,
                "employment_type": job.employment_type,
                "location": job.location,
                "status": job.status,
                "message": (
                    f"Job '{job.title}' in {job.department} requires "
                    f"{len(job.required_skills or [])} specific skills and "
                    f"{job.experience_years}+ years of experience."
                ),
            })

        except Exception as e:
            self.logger.error(f"get_job_requirements tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve job requirements: {str(e)}",
            })

    async def _tool_search_candidates_for_job(self, job_id: int, candidate_id: int) -> str:
        """
        Compute semantic similarity (cosine) and skill coverage for this candidate vs the job.
        Returns scores + ranking context among existing applicants.
        """
        try:
            async with AsyncSessionLocal() as db:
                job_result = await db.execute(
                    select(JobPosting).where(JobPosting.id == job_id)
                )
                job = job_result.scalar_one_or_none()

                cand_result = await db.execute(
                    select(Candidate).where(Candidate.id == candidate_id)
                )
                candidate = cand_result.scalar_one_or_none()

                if not job or not candidate:
                    return json.dumps({
                        "success": False,
                        "message": "Job or candidate not found.",
                    })

                # Cosine similarity via numpy (OpenAI embeddings are L2-normalized)
                similarity_score = None
                if job.embedding is not None and candidate.cv_embedding is not None:
                    job_vec = np.array(job.embedding, dtype=np.float32)
                    cand_vec = np.array(candidate.cv_embedding, dtype=np.float32)
                    norm_j = np.linalg.norm(job_vec)
                    norm_c = np.linalg.norm(cand_vec)
                    if norm_j > 0 and norm_c > 0:
                        similarity_score = float(
                            np.dot(job_vec, cand_vec) / (norm_j * norm_c)
                        )
                        similarity_score = round(max(0.0, min(1.0, similarity_score)), 4)

                # Skill coverage: how many required skills appear in candidate's skill list
                required_skills = [s.lower().strip() for s in (job.required_skills or [])]
                candidate_skills = [s.lower().strip() for s in (candidate.skills or [])]
                matched_skills: List[str] = []
                missing_skills: List[str] = []
                skill_coverage = 0.0

                if required_skills:
                    for req in required_skills:
                        if any(req in cs or cs in req for cs in candidate_skills):
                            matched_skills.append(req)
                        else:
                            missing_skills.append(req)
                    skill_coverage = round(len(matched_skills) / len(required_skills), 4)

                # How many other applicants already have a higher similarity score (ranking context)
                apps_result = await db.execute(
                    select(JobApplication).where(
                        and_(
                            JobApplication.job_id == job_id,
                            JobApplication.similarity_score.isnot(None),
                        )
                    )
                )
                scored_apps = apps_result.scalars().all()
                better_applicants = sum(
                    1 for a in scored_apps
                    if a.candidate_id != candidate_id
                    and a.similarity_score is not None
                    and similarity_score is not None
                    and a.similarity_score > similarity_score
                )

            # Preliminary threshold-based verdict (LLM may override with context)
            if similarity_score is None:
                preliminary = "REVIEW"
                preliminary_reason = "No embedding available — cannot compute similarity"
            elif similarity_score >= 0.75 and skill_coverage >= 0.60:
                preliminary = "SHORTLIST"
                preliminary_reason = (
                    f"High similarity ({similarity_score:.2f}) and good skill coverage ({skill_coverage:.0%})"
                )
            elif similarity_score >= 0.50:
                preliminary = "REVIEW"
                preliminary_reason = f"Moderate similarity ({similarity_score:.2f})"
            else:
                preliminary = "PASS"
                preliminary_reason = f"Low similarity score ({similarity_score:.2f})"

            return json.dumps({
                "success": True,
                "candidate_id": candidate_id,
                "job_id": job_id,
                "similarity_score": similarity_score,
                "skill_coverage": skill_coverage,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "total_required_skills": len(required_skills),
                "candidate_skill_count": len(candidate_skills),
                "applicants_scoring_higher": better_applicants,
                "preliminary_decision": preliminary,
                "message": (
                    f"Candidate scores: similarity={similarity_score:.4f if similarity_score is not None else 'N/A'}, "
                    f"skill_coverage={skill_coverage:.0%} ({len(matched_skills)}/{len(required_skills)} required skills). "
                    f"Preliminary verdict: {preliminary} — {preliminary_reason}. "
                    f"Missing skills: {', '.join(missing_skills) if missing_skills else 'none'}."
                ),
            })

        except Exception as e:
            self.logger.error(f"search_candidates_for_job tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Similarity search failed: {str(e)}",
            })

    async def _tool_check_duplicate_candidate(
        self,
        candidate_id: int,
        job_id: int,
        current_application_id: int,
    ) -> str:
        """Check for duplicate active applications from this candidate for this job."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(JobApplication).where(
                        and_(
                            JobApplication.candidate_id == candidate_id,
                            JobApplication.job_id == job_id,
                            JobApplication.id != current_application_id,
                            JobApplication.status.notin_([
                                ApplicationStatus.PASSED.value,
                                ApplicationStatus.REJECTED.value,
                            ]),
                        )
                    )
                )
                duplicates = result.scalars().all()

            if duplicates:
                dup_info = [
                    f"Application #{d.id} (status: {d.status}, applied: {d.applied_at.date()})"
                    for d in duplicates
                ]
                return json.dumps({
                    "is_duplicate": True,
                    "duplicate_count": len(duplicates),
                    "duplicates": dup_info,
                    "message": (
                        f"DUPLICATE DETECTED: {len(duplicates)} existing active application(s) "
                        f"from this candidate for this job: {'; '.join(dup_info)}. "
                        "Consider downgrading to PASS."
                    ),
                })

            return json.dumps({
                "is_duplicate": False,
                "duplicate_count": 0,
                "duplicates": [],
                "message": "No duplicate applications found. This is a fresh application.",
            })

        except Exception as e:
            self.logger.error(f"check_duplicate_candidate tool error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": f"Duplicate check failed: {str(e)}",
            })

    # -------------------------------------------------------------------------
    # Re-ranking
    # -------------------------------------------------------------------------

    async def _rerank_applications(self, job_id: int) -> None:
        """Re-rank all applications for a job by descending similarity_score (rank 1 = best)."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(JobApplication)
                    .where(JobApplication.job_id == job_id)
                    .order_by(JobApplication.similarity_score.desc().nulls_last())
                )
                applications = result.scalars().all()

                for rank, app in enumerate(applications, start=1):
                    app.rank = rank

                await db.commit()

            self.logger.info(f"Re-ranked {len(applications)} applications for job {job_id}")

        except Exception as e:
            self.logger.error(f"Re-ranking failed for job {job_id}: {e}")

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _parse_llm_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse structured fields from the LLM's formatted output."""
        decision_info: Dict[str, Any] = {
            "decision": "REVIEW",  # Safe default if parsing fails
            "reasoning": llm_output,
            "recommendations": "Review the application manually",
            "confidence_score": 0.5,
        }

        _stop = ["CONFIDENCE:", "FACTORS:", "RECOMMENDATIONS:", "DECISION:",
                 "SIMILARITY_SCORE:", "SKILL_COVERAGE:"]

        # Parse DECISION
        if "DECISION:" in llm_output:
            decision_lines = [l for l in llm_output.split("\n") if "DECISION:" in l]
            if decision_lines:
                line = decision_lines[0]
                if "SHORTLIST" in line:
                    decision_info["decision"] = "SHORTLIST"
                elif "PASS" in line:
                    decision_info["decision"] = "PASS"
                elif "REVIEW" in line:
                    decision_info["decision"] = "REVIEW"

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

        # Parse CONFIDENCE into a numeric score
        for label, score in {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.items():
            if f"CONFIDENCE: {label}" in llm_output:
                decision_info["confidence_score"] = score
                break

        return decision_info

    def _extract_scores_from_steps(self, intermediate_steps: list) -> Dict[str, Any]:
        """Extract similarity_score and skill_coverage from the search_candidates_for_job result."""
        for tool_name, _, result in intermediate_steps:
            if tool_name == "search_candidates_for_job":
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                    return {
                        "similarity_score": data.get("similarity_score"),
                        "skill_coverage": data.get("skill_coverage"),
                    }
                except Exception as e:
                    self.logger.warning(f"Could not parse scores from intermediate steps: {e}")
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
            factors_lines = [l for l in llm_output.split("\n") if "FACTORS:" in l]
            if factors_lines:
                text = factors_lines[0].replace("FACTORS:", "").strip()
                return [f.strip() for f in text.split(",") if f.strip()]
        return ["semantic_similarity", "skill_coverage", "experience_match", "cv_parse_quality"]

    def _build_message(self, decision: str) -> str:
        return {
            "SHORTLIST": "Candidate shortlisted for further review",
            "REVIEW": "Application submitted for manager review",
            "PASS": "Candidate did not meet the minimum requirements",
        }.get(decision, "Application processed")

    def get_input_schema(self) -> type[HiringApplicationInput]:
        return HiringApplicationInput

    def get_output_schema(self) -> type[HiringApplicationOutput]:
        return HiringApplicationOutput


# Agent is registered via agents/register_agents.py on application startup
