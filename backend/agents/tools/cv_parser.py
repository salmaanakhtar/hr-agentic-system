"""
CV Parser Tool for Phase 5 Hiring Agent.

Uses pypdf for text extraction and regex patterns for structured data extraction:
- Skills: section-based extraction + known-keyword fallback
- Experience years: explicit statement + date range calculation
- Education: degree keyword detection with institution and year
- Current title: top-of-CV heuristic
- Called via asyncio.to_thread from async code (synchronous)
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.schemas import CVParseResult

logger = logging.getLogger(__name__)

_CURRENT_YEAR = datetime.utcnow().year


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

_KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "git", "linux", "bash", "rest", "graphql", "grpc", "microservices",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "spark", "hadoop", "kafka",
    "agile", "scrum", "jira", "ci/cd", "devops",
    "html", "css", "sass", "webpack", "figma",
]

_SKILLS_SECTION_RE = re.compile(
    r"(?:technical\s+)?skills?[ \t]*:?[ \t]*\n+([\s\S]*?)(?:\n{2,}|\Z)",
    re.IGNORECASE,
)


def _extract_skills(text: str) -> list[str]:
    skills: set[str] = set()
    lower = text.lower()

    # 1. Try to parse a dedicated skills section
    section_match = _SKILLS_SECTION_RE.search(text)
    if section_match:
        section_text = section_match.group(1)
        items = re.split(r"[,\n•·▪\-–|/]", section_text)
        for item in items:
            item = item.strip().strip("*").strip()
            # Accept items that look like skill names (not pure numbers/long sentences)
            if 2 <= len(item) <= 50 and not re.match(r"^[\d\s]+$", item):
                skills.add(item.lower())

    # 2. Known-skill keyword scan across the full CV text
    for skill in _KNOWN_SKILLS:
        if " " in skill:
            if skill in lower:
                skills.add(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", lower):
                skills.add(skill)

    return sorted(skills)


# ---------------------------------------------------------------------------
# Experience years
# ---------------------------------------------------------------------------

_YEARS_STATED_RE = re.compile(
    r"\b(\d+)\+?\s+years?\s+(?:of\s+)?(?:professional\s+)?experience\b",
    re.IGNORECASE,
)

# Matches: "2018 - 2023", "Jan 2018 - Dec 2023", "2020 - Present"
_DATE_RANGE_RE = re.compile(
    r"\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?"
    r"(\d{4})\s*[-–—]\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?"
    r"(present|current|now|\d{4})\b",
    re.IGNORECASE,
)


def _extract_experience_years(text: str) -> Optional[int]:
    # 1. Explicit statement takes priority
    stated = _YEARS_STATED_RE.findall(text)
    if stated:
        return max(int(y) for y in stated)

    # 2. Sum date ranges found in the text (heuristic)
    total = 0
    for match in _DATE_RANGE_RE.finditer(text):
        start = int(match.group(1))
        end_str = match.group(2)
        end = _CURRENT_YEAR if end_str.lower() in ("present", "current", "now") else int(end_str)
        years = max(0, end - start)
        if 0 < years <= 50:  # sanity bounds
            total += years

    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

_DEGREE_RE = re.compile(
    r"\b(b\.?sc\.?|m\.?sc\.?|ph\.?d\.?|bachelor(?:'?s)?|master(?:'?s)?|doctorate|"
    r"b\.?eng\.?|m\.?eng\.?|b\.?tech\.?|m\.?tech\.?|b\.?a\.?|m\.?a\.?|mba|llb|llm|associate)\b",
    re.IGNORECASE,
)

_INSTITUTION_RE = re.compile(
    r"\b([A-Z][a-zA-Z\s&'\-]+(?:University|College|Institute|School|Academy|Polytechnic))\b",
)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _extract_education(text: str) -> list[dict]:
    entries = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        if not _DEGREE_RE.search(line):
            continue

        # Use surrounding context to find institution and year
        context = " ".join(lines[max(0, i - 1) : min(len(lines), i + 3)])

        degree_match = _DEGREE_RE.search(line)
        degree = degree_match.group(0).strip() if degree_match else line.strip()

        year_matches = _YEAR_RE.findall(context)
        year = max(int(y) for y in year_matches) if year_matches else None

        inst_match = _INSTITUTION_RE.search(context)
        institution = inst_match.group(1).strip() if inst_match else None

        entries.append({"degree": degree, "institution": institution, "year": year})

    return entries


# ---------------------------------------------------------------------------
# Current job title
# ---------------------------------------------------------------------------

_TITLE_KEYWORDS_RE = re.compile(
    r"\b(?:engineer|developer|manager|analyst|designer|architect|lead|senior|junior|"
    r"director|consultant|specialist|coordinator|officer|executive|scientist|"
    r"administrator|technician|intern)\b",
    re.IGNORECASE,
)


def _extract_current_title(text: str) -> Optional[str]:
    """
    Scan the first 25 lines for a line that looks like a job title.
    Skips lines containing contact details (email, phone, URLs).
    """
    for line in text.split("\n")[:25]:
        line = line.strip()
        if 3 <= len(line) <= 80 and _TITLE_KEYWORDS_RE.search(line):
            # Skip contact-info lines
            if re.search(r"[@\d{7,}http]", line):
                continue
            return line
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cv(file_path: str) -> CVParseResult:
    """
    Main entry point. Extracts text from a PDF CV at file_path and returns
    a fully populated CVParseResult.

    This function is synchronous and must be called via asyncio.to_thread
    from async code.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"CV file not found: {file_path}")
        return CVParseResult(raw_text="", parse_confidence=0.0)

    # 1. Extract raw text from all PDF pages
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n".join(pages).strip()
    except Exception as exc:
        logger.error(f"PDF text extraction failed for {file_path}: {exc}")
        return CVParseResult(raw_text="", parse_confidence=0.0)

    if not raw_text:
        logger.warning(f"No text extracted from CV: {file_path}")
        return CVParseResult(raw_text="", parse_confidence=0.0)

    # 2. Extract structured fields
    skills = _extract_skills(raw_text)
    experience_years = _extract_experience_years(raw_text)
    education = _extract_education(raw_text)
    current_title = _extract_current_title(raw_text)

    # 3. Confidence: fraction of the 4 structured fields successfully extracted
    fields_found = sum([
        bool(skills),
        experience_years is not None,
        bool(education),
        bool(current_title),
    ])
    parse_confidence = round(fields_found / 4.0, 2)

    logger.info(
        f"CV parsed - skills: {len(skills)}, experience: {experience_years}yr, "
        f"education: {len(education)}, title: {current_title!r}, "
        f"confidence: {parse_confidence}"
    )

    return CVParseResult(
        raw_text=raw_text,
        skills=skills,
        experience_years=experience_years,
        education=education,
        current_title=current_title,
        parse_confidence=parse_confidence,
    )
