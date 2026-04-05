"""
Policy Compliance API Endpoints (Phase 7.4)

Provides REST API for uploading policy documents (with auto-parse + embed),
listing/retrieving/deleting documents, running policy Q&A and compliance checks
via the PolicyComplianceAgent, and reporting.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.policy_agent import PolicyComplianceAgent
from agents.schemas import ComplianceCheckInput, PolicyQueryInput
from agents.tools.policy_parser import parse_and_chunk_policy
from app.database import get_async_db
from app.main import get_current_user
from app.models import PolicyCategory, PolicyChunk, PolicyDocument, User

router = APIRouter(prefix="/api/policies", tags=["policies"])
logger = logging.getLogger(__name__)

_UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "policies"
)

_VALID_CATEGORIES = {c.value for c in PolicyCategory}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_role(user: User, *roles: str):
    if user.role.name not in roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Required role(s): {', '.join(roles)}",
        )


def _doc_to_dict(doc: PolicyDocument, chunk_count: int = 0) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "description": doc.description,
        "category": doc.category,
        "filename": doc.filename,
        "uploaded_by": doc.uploaded_by,
        "is_active": doc.is_active,
        "chunk_count": chunk_count,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


# ---------------------------------------------------------------------------
# 7.4.1 — POST /documents  (HR/Admin — multipart PDF upload)
# ---------------------------------------------------------------------------

@router.post("/documents", status_code=201)
async def upload_policy_document(
    title: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Upload a PDF policy document.

    The file is saved to disk, parsed into text chunks (pypdf sliding-window),
    each chunk is embedded with OpenAI text-embedding-3-small and stored in the
    policy_chunks table for pgvector RAG search.

    HR/Admin only.
    """
    _require_role(current_user, "hr", "admin")

    if category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    current_user_id = current_user.id

    # Save file to disk
    os.makedirs(_UPLOADS_DIR, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(_UPLOADS_DIR, unique_filename)

    try:
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:  # 20 MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20 MB")

        with open(file_path, "wb") as f:
            f.write(contents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save policy file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Parse and chunk (synchronous — run in thread pool)
    try:
        chunks_text = await asyncio.to_thread(parse_and_chunk_policy, file_path)
    except Exception as e:
        logger.error(f"Policy parsing failed: {e}", exc_info=True)
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    if not chunks_text:
        os.remove(file_path)
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the PDF. Ensure it is a text-based (not scanned) PDF.",
        )

    # Embed all chunks
    try:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def _embed_all(texts: list[str]) -> list[list[float]]:
            response = openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small",
            )
            return [item.embedding for item in response.data]

        embeddings = await asyncio.to_thread(_embed_all, chunks_text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

    # Persist PolicyDocument
    doc = PolicyDocument(
        title=title,
        description=description,
        category=category,
        filename=unique_filename,
        file_path=file_path,
        uploaded_by=current_user_id,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    doc_id = doc.id

    # Persist PolicyChunks
    for idx, (text, embedding) in enumerate(zip(chunks_text, embeddings)):
        chunk = PolicyChunk(
            document_id=doc_id,
            chunk_index=idx,
            content=text,
            embedding=embedding,
            token_count=len(text.split()),
        )
        db.add(chunk)

    await db.commit()

    logger.info(
        f"Policy document '{title}' uploaded by user {current_user_id}: "
        f"{len(chunks_text)} chunks embedded (doc_id={doc_id})"
    )

    return {
        "message": "Policy document uploaded and indexed successfully",
        "document_id": doc_id,
        "title": title,
        "category": category,
        "filename": unique_filename,
        "chunk_count": len(chunks_text),
    }


# ---------------------------------------------------------------------------
# 7.4.2 — GET /documents
# ---------------------------------------------------------------------------

@router.get("/documents")
async def list_policy_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Return only active documents"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List policy documents. All authenticated users."""
    filters = []
    if active_only:
        filters.append(PolicyDocument.is_active == True)
    if category:
        if category not in _VALID_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}",
            )
        filters.append(PolicyDocument.category == category)

    from sqlalchemy import and_
    query = select(PolicyDocument)
    if filters:
        query = query.where(and_(*filters))
    query = query.order_by(PolicyDocument.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    docs = result.scalars().all()

    output = []
    for doc in docs:
        count_result = await db.execute(
            select(func.count(PolicyChunk.id)).where(PolicyChunk.document_id == doc.id)
        )
        chunk_count = count_result.scalar() or 0
        output.append(_doc_to_dict(doc, chunk_count))

    return {
        "documents": output,
        "total": len(output),
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# 7.4.3 — GET /documents/{id}
# ---------------------------------------------------------------------------

@router.get("/documents/{document_id}")
async def get_policy_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a single policy document with chunk count. All authenticated users."""
    result = await db.execute(
        select(PolicyDocument).where(PolicyDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Policy document not found")

    count_result = await db.execute(
        select(func.count(PolicyChunk.id)).where(PolicyChunk.document_id == document_id)
    )
    chunk_count = count_result.scalar() or 0

    return _doc_to_dict(doc, chunk_count)


# ---------------------------------------------------------------------------
# 7.4.4 — DELETE /documents/{id}  (HR/Admin — soft delete)
# ---------------------------------------------------------------------------

@router.delete("/documents/{document_id}")
async def delete_policy_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Soft-delete a policy document (sets is_active=False).
    The document and its chunks remain in the database and are excluded
    from all RAG searches.

    HR/Admin only.
    """
    _require_role(current_user, "hr", "admin")

    result = await db.execute(
        select(PolicyDocument).where(PolicyDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Policy document not found")

    if not doc.is_active:
        raise HTTPException(status_code=400, detail="Policy document is already inactive")

    doc.is_active = False
    doc_title = doc.title  # cache before commit
    deactivator_id = current_user.id  # cache before commit
    await db.commit()

    logger.info(
        f"Policy document {document_id} deactivated by user {deactivator_id}"
    )

    return {
        "message": "Policy document deactivated successfully",
        "document_id": document_id,
        "title": doc_title,
        "is_active": False,
    }


# ---------------------------------------------------------------------------
# 7.4.5 — POST /query  (All authenticated users)
# ---------------------------------------------------------------------------

@router.post("/query")
async def query_policy(
    body: PolicyQueryInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Answer a free-text HR policy question using RAG over embedded policy chunks.

    The PolicyComplianceAgent searches relevant chunks via pgvector cosine
    similarity, then generates a grounded answer with citations.
    """
    # Override user_id from JWT
    body = PolicyQueryInput(user_id=current_user.id, question=body.question)

    try:
        agent = PolicyComplianceAgent()
        result = await agent.query(body)
        return result.dict()
    except Exception as e:
        logger.error(f"Policy query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Policy query failed: {str(e)}")


# ---------------------------------------------------------------------------
# 7.4.6 — POST /compliance-check  (All authenticated users)
# ---------------------------------------------------------------------------

@router.post("/compliance-check")
async def compliance_check(
    body: ComplianceCheckInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Evaluate whether a described scenario complies with HR policies.

    Returns COMPLIANT | NON_COMPLIANT | UNCLEAR with reasoning, citations,
    and recommendations.
    """
    # Override user_id from JWT
    body = ComplianceCheckInput(
        user_id=current_user.id,
        scenario=body.scenario,
        context=body.context,
    )

    try:
        agent = PolicyComplianceAgent()
        result = await agent.check_compliance(body)
        return result.dict()
    except Exception as e:
        logger.error(f"Compliance check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


# ---------------------------------------------------------------------------
# 7.4.7 — GET /reports  (HR/Admin)
# ---------------------------------------------------------------------------

@router.get("/reports")
async def get_policy_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Policy document statistics. HR/Admin only."""
    _require_role(current_user, "hr", "admin")

    # Active / inactive document counts
    active_result = await db.execute(
        select(func.count(PolicyDocument.id)).where(PolicyDocument.is_active == True)
    )
    inactive_result = await db.execute(
        select(func.count(PolicyDocument.id)).where(PolicyDocument.is_active == False)
    )
    total_active = active_result.scalar() or 0
    total_inactive = inactive_result.scalar() or 0

    # Documents per category (active only)
    category_counts: dict = {}
    for cat in PolicyCategory:
        cat_result = await db.execute(
            select(func.count(PolicyDocument.id)).where(
                PolicyDocument.category == cat.value,
                PolicyDocument.is_active == True,
            )
        )
        category_counts[cat.value] = cat_result.scalar() or 0

    # Total embedded chunks (active documents)
    chunk_result = await db.execute(
        select(func.count(PolicyChunk.id))
        .join(PolicyDocument, PolicyChunk.document_id == PolicyDocument.id)
        .where(PolicyDocument.is_active == True)
    )
    total_chunks = chunk_result.scalar() or 0

    # Most recently uploaded active document
    recent_result = await db.execute(
        select(PolicyDocument)
        .where(PolicyDocument.is_active == True)
        .order_by(PolicyDocument.created_at.desc())
        .limit(1)
    )
    recent_doc = recent_result.scalar_one_or_none()

    return {
        "summary": {
            "total_active_documents": total_active,
            "total_inactive_documents": total_inactive,
            "total_documents": total_active + total_inactive,
            "total_embedded_chunks": total_chunks,
        },
        "documents_by_category": category_counts,
        "most_recent_document": _doc_to_dict(recent_doc) if recent_doc else None,
    }
