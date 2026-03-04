"""
Hiring Pipeline API Endpoints

Provides REST API for job posting management, candidate upload with CV,
application submission (triggers HiringAgent evaluation), and hiring reports.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional

import aiofiles
import openai
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.database import get_async_db
from app.models import (
    User,
    JobPosting,
    Candidate,
    JobApplication,
    ApplicationStatus,
    JobStatus,
)
from agents.hiring_agent import HiringAgent
from agents.schemas import (
    HiringApplicationInput,
    JobPostingCreate,
    JobPostingUpdate,
    ApplicationCreate,
    ApplicationStatusUpdate,
    InterviewScheduleCreate,
)
from app.main import get_current_user

router = APIRouter(prefix="/api/hiring", tags=["hiring"])
logger = logging.getLogger(__name__)

# Resolve uploads directory relative to this file's location
_CVS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "uploads",
    "cvs",
)
_ALLOWED_CV_EXTENSIONS = {".pdf"}
_MAX_CV_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_role(user: User, *roles: str):
    if user.role.name not in roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Required role(s): {', '.join(roles)}"
        )


async def _generate_job_embedding(text: str) -> Optional[List[float]]:
    """Generate OpenAI text-embedding-3-small from job text."""
    try:
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:32000],
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Failed to generate job embedding: {e}")
        return None


def _job_to_dict(job: JobPosting) -> dict:
    return {
        "id": job.id,
        "title": job.title,
        "department": job.department,
        "description": job.description,
        "requirements": job.requirements,
        "required_skills": job.required_skills or [],
        "experience_years": job.experience_years,
        "employment_type": job.employment_type,
        "location": job.location,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "status": job.status,
        "created_by": job.created_by,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "has_embedding": job.embedding is not None,
    }


def _candidate_to_dict(candidate: Candidate) -> dict:
    return {
        "id": candidate.id,
        "first_name": candidate.first_name,
        "last_name": candidate.last_name,
        "full_name": candidate.full_name,
        "email": candidate.email,
        "phone": candidate.phone,
        "cv_filename": candidate.cv_filename,
        "cv_url": candidate.cv_url,
        "skills": candidate.skills or [],
        "experience_years": candidate.experience_years,
        "education": candidate.education or [],
        "current_title": candidate.current_title,
        "linkedin_url": candidate.linkedin_url,
        "has_embedding": candidate.cv_embedding is not None,
        "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
        "updated_at": candidate.updated_at.isoformat() if candidate.updated_at else None,
    }


def _application_to_dict(
    app: JobApplication,
    job: Optional[JobPosting] = None,
    candidate: Optional[Candidate] = None,
) -> dict:
    return {
        "id": app.id,
        "job_id": app.job_id,
        "candidate_id": app.candidate_id,
        "status": app.status,
        "similarity_score": app.similarity_score,
        "skill_coverage": app.skill_coverage,
        "rank": app.rank,
        "llm_decision": app.llm_decision,
        "llm_reasoning": app.llm_reasoning,
        "interview_date": app.interview_date.isoformat() if app.interview_date else None,
        "interview_notes": app.interview_notes,
        "reviewed_by": app.reviewed_by,
        "reviewed_at": app.reviewed_at.isoformat() if app.reviewed_at else None,
        "applied_at": app.applied_at.isoformat() if app.applied_at else None,
        "updated_at": app.updated_at.isoformat() if app.updated_at else None,
        "job": (
            {"id": job.id, "title": job.title, "department": job.department}
            if job else None
        ),
        "candidate": (
            {"id": candidate.id, "full_name": candidate.full_name, "email": candidate.email}
            if candidate else None
        ),
    }


# ---------------------------------------------------------------------------
# 5.4.1: Job Posting Endpoints
# ---------------------------------------------------------------------------

@router.post("/jobs")
async def create_job_posting(
    job_data: JobPostingCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Create a new job posting. HR/Manager/Admin only.
    Automatically generates a semantic embedding (text-embedding-3-small) for CV matching.
    """
    _require_role(current_user, "hr", "manager", "admin")

    valid_employment_types = ["full_time", "part_time", "contract"]
    if job_data.employment_type not in valid_employment_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid employment_type. Must be one of: {', '.join(valid_employment_types)}"
        )

    job = JobPosting(
        title=job_data.title,
        department=job_data.department,
        description=job_data.description,
        requirements=job_data.requirements,
        required_skills=job_data.required_skills,
        experience_years=job_data.experience_years,
        employment_type=job_data.employment_type,
        location=job_data.location,
        salary_min=job_data.salary_min,
        salary_max=job_data.salary_max,
        status=JobStatus.OPEN.value,
        created_by=current_user.id,
    )
    db.add(job)
    await db.flush()
    job_id = job.id
    await db.commit()

    # Generate embedding after commit (non-blocking failure — job still created)
    embedding_text = (
        f"{job_data.title}\n\n"
        f"{job_data.description}\n\n"
        f"Requirements: {job_data.requirements}\n\n"
        f"Required Skills: {', '.join(job_data.required_skills)}"
    )
    embedding = await _generate_job_embedding(embedding_text)

    if embedding:
        result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
        saved_job = result.scalar_one()
        saved_job.embedding = embedding
        await db.commit()
        logger.info(f"Generated embedding for job {job_id}")

    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    saved_job = result.scalar_one()

    return {"message": "Job posting created successfully", "job": _job_to_dict(saved_job)}


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: draft, open, closed, on_hold"),
    department: Optional[str] = Query(None, description="Filter by department name"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List job postings with optional filtering. All authenticated users."""
    filters = []
    if status:
        filters.append(JobPosting.status == status)
    if department:
        filters.append(JobPosting.department == department)

    query = (
        select(JobPosting)
        .order_by(JobPosting.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)
    jobs = result.scalars().all()

    return {
        "jobs": [_job_to_dict(j) for j in jobs],
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a single job posting by ID."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return _job_to_dict(job)


@router.put("/jobs/{job_id}")
async def update_job(
    job_id: int,
    job_data: JobPostingUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Update a job posting. HR/Manager/Admin only. Regenerates embedding if content fields change."""
    _require_role(current_user, "hr", "manager", "admin")

    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    content_fields = {"title", "description", "requirements", "required_skills"}
    regenerate_embedding = False
    update_data = job_data.model_dump(exclude_none=True)

    for field, value in update_data.items():
        setattr(job, field, value)
        if field in content_fields:
            regenerate_embedding = True

    job.updated_at = datetime.utcnow()
    await db.commit()

    if regenerate_embedding:
        embedding_text = (
            f"{job.title}\n\n"
            f"{job.description}\n\n"
            f"Requirements: {job.requirements}\n\n"
            f"Required Skills: {', '.join(job.required_skills or [])}"
        )
        embedding = await _generate_job_embedding(embedding_text)
        if embedding:
            job.embedding = embedding
            await db.commit()
            logger.info(f"Regenerated embedding for job {job_id}")

    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    updated_job = result.scalar_one()
    return {"message": "Job posting updated successfully", "job": _job_to_dict(updated_job)}


# ---------------------------------------------------------------------------
# 5.4.2: Candidate Endpoints
# ---------------------------------------------------------------------------

@router.post("/candidates")
async def upload_candidate(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    phone: Optional[str] = Form(None),
    linkedin_url: Optional[str] = Form(None),
    cv: UploadFile = File(..., description="Candidate CV (PDF only, max 10MB)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Upload a new candidate with their CV. HR/Manager/Admin only.
    CV is stored on disk; parsing and embedding generation happen during application evaluation.
    """
    _require_role(current_user, "hr", "manager", "admin")

    # Check email uniqueness
    result = await db.execute(select(Candidate).where(Candidate.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="A candidate with this email already exists")

    # Validate CV file type
    ext = os.path.splitext(cv.filename or "")[1].lower()
    if ext not in _ALLOWED_CV_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted for CVs")

    file_bytes = await cv.read()
    if len(file_bytes) > _MAX_CV_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"CV file too large ({len(file_bytes) // 1024}KB). Maximum is 10MB."
        )

    # Save with UUID filename to prevent collisions
    cv_filename = f"{uuid.uuid4().hex}.pdf"
    os.makedirs(_CVS_DIR, exist_ok=True)
    cv_path = os.path.join(_CVS_DIR, cv_filename)

    async with aiofiles.open(cv_path, "wb") as f:
        await f.write(file_bytes)

    logger.info(f"Saved CV: {cv_filename} ({len(file_bytes)} bytes)")

    candidate = Candidate(
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone,
        linkedin_url=linkedin_url,
        cv_filename=cv_filename,
        cv_path=cv_path,
    )
    db.add(candidate)
    await db.flush()
    candidate_id = candidate.id
    await db.commit()

    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    saved_candidate = result.scalar_one()

    return {
        "message": "Candidate uploaded successfully",
        "candidate": _candidate_to_dict(saved_candidate),
    }


@router.get("/candidates")
async def list_candidates(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List all candidates. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    result = await db.execute(
        select(Candidate)
        .order_by(Candidate.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    candidates = result.scalars().all()

    return {
        "candidates": [_candidate_to_dict(c) for c in candidates],
        "total": len(candidates),
        "limit": limit,
        "offset": offset,
    }


@router.get("/candidates/{candidate_id}")
async def get_candidate(
    candidate_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a single candidate profile. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return _candidate_to_dict(candidate)


# ---------------------------------------------------------------------------
# 5.4.3: Application Endpoints
# ---------------------------------------------------------------------------

@router.post("/applications")
async def create_application(
    app_data: ApplicationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Submit a candidate application and trigger HiringAgent evaluation. HR/Manager/Admin only.

    The HiringAgent autonomously:
    1. Parses the candidate CV (skills, experience, education, title)
    2. Generates a semantic embedding for the CV
    3. Retrieves job requirements
    4. Computes cosine similarity and skill coverage scores
    5. Checks for duplicate applications
    6. Makes a SHORTLIST/REVIEW/PASS decision with full reasoning
    """
    _require_role(current_user, "hr", "manager", "admin")

    # Verify job exists and is open
    job_result = await db.execute(select(JobPosting).where(JobPosting.id == app_data.job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")
    if job.status != JobStatus.OPEN.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job posting is not open (status: {job.status})"
        )

    # Verify candidate exists and has a CV on disk
    cand_result = await db.execute(select(Candidate).where(Candidate.id == app_data.candidate_id))
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.cv_path or not os.path.exists(candidate.cv_path):
        raise HTTPException(status_code=400, detail="Candidate CV file not found on disk")

    # Block duplicate active applications
    dup_result = await db.execute(
        select(JobApplication).where(
            and_(
                JobApplication.job_id == app_data.job_id,
                JobApplication.candidate_id == app_data.candidate_id,
                JobApplication.status.notin_([
                    ApplicationStatus.PASSED.value,
                    ApplicationStatus.REJECTED.value,
                ]),
            )
        )
    )
    existing = dup_result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Candidate already has an active application for this job (application #{existing.id})"
        )

    # Create application record
    application = JobApplication(
        job_id=app_data.job_id,
        candidate_id=app_data.candidate_id,
        status=ApplicationStatus.APPLIED.value,
    )
    db.add(application)
    await db.flush()
    application_id = application.id
    await db.commit()

    logger.info(
        f"Created application {application_id}: candidate {app_data.candidate_id} "
        f"for job {app_data.job_id} — triggering HiringAgent"
    )

    # Run HiringAgent evaluation
    agent_input = HiringApplicationInput(
        application_id=application_id,
        job_id=app_data.job_id,
        candidate_id=app_data.candidate_id,
        submitted_by=current_user.id,
    )
    agent = HiringAgent()
    return await agent.execute(agent_input)


@router.get("/applications")
async def list_applications(
    job_id: Optional[int] = Query(None, description="Filter by job posting ID"),
    candidate_id: Optional[int] = Query(None, description="Filter by candidate ID"),
    status: Optional[str] = Query(None, description="Filter by application status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List applications with optional filtering. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    filters = []
    if job_id is not None:
        filters.append(JobApplication.job_id == job_id)
    if candidate_id is not None:
        filters.append(JobApplication.candidate_id == candidate_id)
    if status:
        filters.append(JobApplication.status == status)

    query = (
        select(JobApplication)
        .order_by(JobApplication.rank.asc().nulls_last(), JobApplication.applied_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)
    applications = result.scalars().all()

    output = []
    for app in applications:
        job_res = await db.execute(select(JobPosting).where(JobPosting.id == app.job_id))
        job = job_res.scalar_one_or_none()
        cand_res = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
        cand = cand_res.scalar_one_or_none()
        output.append(_application_to_dict(app, job, cand))

    return {
        "applications": output,
        "total": len(output),
        "limit": limit,
        "offset": offset,
    }


@router.put("/applications/{application_id}/status")
async def update_application_status(
    application_id: int,
    status_data: ApplicationStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Manually update an application status. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    valid_statuses = [s.value for s in ApplicationStatus]
    if status_data.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    result = await db.execute(
        select(JobApplication).where(JobApplication.id == application_id)
    )
    application = result.scalar_one_or_none()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    application_id_cached = application.id
    application.status = status_data.status
    application.reviewed_by = current_user.id
    application.reviewed_at = datetime.utcnow()
    application.updated_at = datetime.utcnow()
    if status_data.notes:
        application.interview_notes = status_data.notes
    await db.commit()

    logger.info(
        f"User {current_user.id} updated application {application_id_cached} "
        f"status to {status_data.status}"
    )

    return {
        "message": "Application status updated",
        "application_id": application_id_cached,
        "new_status": status_data.status,
    }


@router.post("/applications/{application_id}/schedule-interview")
async def schedule_interview(
    application_id: int,
    schedule_data: InterviewScheduleCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Schedule an interview for an application. Sets status to 'interviewing'. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    result = await db.execute(
        select(JobApplication).where(JobApplication.id == application_id)
    )
    application = result.scalar_one_or_none()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    try:
        interview_dt = datetime.fromisoformat(schedule_data.interview_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid interview_date format. Use ISO 8601 (e.g. 2025-06-15T14:00:00)"
        )

    application_id_cached = application.id
    application.interview_date = interview_dt
    application.interview_notes = schedule_data.interview_notes
    application.status = ApplicationStatus.INTERVIEWING.value
    application.updated_at = datetime.utcnow()
    await db.commit()

    logger.info(
        f"User {current_user.id} scheduled interview for application {application_id_cached} "
        f"on {interview_dt.isoformat()}"
    )

    return {
        "message": "Interview scheduled successfully",
        "application_id": application_id_cached,
        "interview_date": interview_dt.isoformat(),
        "status": ApplicationStatus.INTERVIEWING.value,
    }


# ---------------------------------------------------------------------------
# 5.4.4: Reports
# ---------------------------------------------------------------------------

@router.get("/reports")
async def get_hiring_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get hiring pipeline statistics and LLM decision breakdown. HR/Manager/Admin only."""
    _require_role(current_user, "hr", "manager", "admin")

    # Jobs by status
    job_status_result = await db.execute(
        select(JobPosting.status, func.count(JobPosting.id)).group_by(JobPosting.status)
    )
    jobs_by_status = {row[0]: row[1] for row in job_status_result.all()}

    # Total candidates
    total_candidates_result = await db.execute(select(func.count(Candidate.id)))
    total_candidates = total_candidates_result.scalar() or 0

    # Applications by status
    app_status_result = await db.execute(
        select(JobApplication.status, func.count(JobApplication.id))
        .group_by(JobApplication.status)
    )
    apps_by_status = {row[0]: row[1] for row in app_status_result.all()}
    total_apps = sum(apps_by_status.values())

    # LLM decision counts
    shortlist_result = await db.execute(
        select(func.count(JobApplication.id)).where(JobApplication.llm_decision == "SHORTLIST")
    )
    review_result = await db.execute(
        select(func.count(JobApplication.id)).where(JobApplication.llm_decision == "REVIEW")
    )
    pass_result = await db.execute(
        select(func.count(JobApplication.id)).where(JobApplication.llm_decision == "PASS")
    )
    shortlisted = shortlist_result.scalar() or 0
    reviewed = review_result.scalar() or 0
    passed = pass_result.scalar() or 0

    # Average similarity score for shortlisted applications
    avg_sim_result = await db.execute(
        select(func.avg(JobApplication.similarity_score)).where(
            JobApplication.status == ApplicationStatus.SHORTLISTED.value
        )
    )
    avg_similarity = avg_sim_result.scalar()

    return {
        "summary": {
            "total_jobs": sum(jobs_by_status.values()),
            "open_jobs": jobs_by_status.get(JobStatus.OPEN.value, 0),
            "total_candidates": total_candidates,
            "total_applications": total_apps,
        },
        "jobs_by_status": jobs_by_status,
        "applications_by_status": apps_by_status,
        "llm_decisions": {
            "shortlisted": shortlisted,
            "reviewed": reviewed,
            "passed": passed,
            "shortlist_rate": round((shortlisted / total_apps * 100) if total_apps > 0 else 0, 1),
        },
        "performance": {
            "avg_similarity_shortlisted": round(avg_similarity, 4) if avg_similarity else None,
        },
    }
