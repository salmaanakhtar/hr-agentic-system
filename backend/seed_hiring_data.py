"""
Seed hiring pipeline data: job postings, candidates, and applications.
Run once: python seed_hiring_data.py

Note: Seeded candidates have no real CV files on disk and no embeddings.
      They represent pre-processed records for demo/testing purposes.
      LLM decisions and scores are set directly without running the agent.
"""

from app.database import SessionLocal
from app.models import (
    JobPosting,
    Candidate,
    JobApplication,
    JobStatus,
    ApplicationStatus,
    User,
)
from datetime import datetime


def seed():
    db = SessionLocal()
    try:
        # Find a user to act as created_by (use first HR or manager user)
        creator = db.query(User).first()
        if not creator:
            print("[ERROR] No users found — run the app first to seed roles and users")
            return
        creator_id = creator.id
        print(f"Using user_id={creator_id} as job creator")

        # -----------------------------------------------------------------------
        # Job Postings
        # -----------------------------------------------------------------------
        job_specs = [
            {
                "title": "Senior Backend Engineer",
                "department": "Engineering",
                "description": (
                    "We are looking for a Senior Backend Engineer to join our platform team. "
                    "You will design and build scalable APIs, own service reliability, and mentor "
                    "junior engineers. Our stack is Python (FastAPI), PostgreSQL, and AWS."
                ),
                "requirements": (
                    "5+ years of backend engineering experience. "
                    "Strong Python skills, experience with FastAPI or Django. "
                    "PostgreSQL query optimisation. REST API design. "
                    "Familiarity with Docker and CI/CD pipelines."
                ),
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "REST API", "Docker", "AWS"],
                "experience_years": 5,
                "employment_type": "full_time",
                "location": "London, UK (Hybrid)",
                "salary_min": 70000.0,
                "salary_max": 95000.0,
                "status": JobStatus.OPEN.value,
            },
            {
                "title": "Frontend Engineer (Angular)",
                "department": "Engineering",
                "description": (
                    "Join our product team to build beautiful, performant user interfaces. "
                    "You will work closely with designers and backend engineers to deliver "
                    "features used by thousands of HR professionals daily."
                ),
                "requirements": (
                    "3+ years of Angular experience (v14+). "
                    "TypeScript, RxJS, SCSS. "
                    "Strong understanding of component architecture and state management. "
                    "Experience with REST API integration and Playwright or Cypress for E2E testing."
                ),
                "required_skills": ["Angular", "TypeScript", "RxJS", "SCSS", "REST API", "Playwright"],
                "experience_years": 3,
                "employment_type": "full_time",
                "location": "London, UK (Hybrid)",
                "salary_min": 55000.0,
                "salary_max": 75000.0,
                "status": JobStatus.OPEN.value,
            },
            {
                "title": "ML Engineer",
                "department": "AI & Data",
                "description": (
                    "We are building the next generation of AI-powered HR tools. "
                    "You will develop and maintain ML models for document understanding, "
                    "semantic search, and intelligent workflow automation."
                ),
                "requirements": (
                    "4+ years of ML engineering experience. "
                    "Python, PyTorch or TensorFlow, scikit-learn. "
                    "Experience with NLP and embeddings (OpenAI, HuggingFace). "
                    "Vector databases (pgvector, Pinecone). LangChain or LlamaIndex. "
                    "Ability to productionise models via REST APIs."
                ),
                "required_skills": [
                    "Python", "PyTorch", "NLP", "OpenAI", "LangChain",
                    "pgvector", "scikit-learn", "FastAPI",
                ],
                "experience_years": 4,
                "employment_type": "full_time",
                "location": "Remote (UK)",
                "salary_min": 80000.0,
                "salary_max": 110000.0,
                "status": JobStatus.OPEN.value,
            },
        ]

        seeded_jobs = []
        jobs_created = 0
        for spec in job_specs:
            exists = db.query(JobPosting).filter(
                JobPosting.title == spec["title"],
                JobPosting.department == spec["department"],
            ).first()
            if exists:
                print(f"[SKIP] Job already exists: {spec['title']}")
                seeded_jobs.append(exists)
            else:
                job = JobPosting(**spec, created_by=creator_id)
                db.add(job)
                db.flush()
                seeded_jobs.append(job)
                jobs_created += 1
                print(f"[OK]   Seeded job: {spec['title']}")

        db.commit()
        print(f"\nJobs: {jobs_created} seeded, {len(job_specs) - jobs_created} already existed")

        # Refresh to get IDs
        for job in seeded_jobs:
            db.refresh(job)

        # -----------------------------------------------------------------------
        # Candidates
        # -----------------------------------------------------------------------
        candidate_specs = [
            {
                "first_name": "Alice",
                "last_name": "Chen",
                "email": "alice.chen@example.com",
                "phone": "+44 7700 900001",
                "current_title": "Senior Software Engineer",
                "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "REST API", "Redis"],
                "experience_years": 6,
                "education": [
                    {"degree": "BSc Computer Science", "institution": "UCL", "year": 2018}
                ],
                "linkedin_url": "https://linkedin.com/in/alice-chen-example",
                "cv_filename": None,
                "cv_path": None,
            },
            {
                "first_name": "Ben",
                "last_name": "Okafor",
                "email": "ben.okafor@example.com",
                "phone": "+44 7700 900002",
                "current_title": "Backend Developer",
                "skills": ["Python", "Django", "PostgreSQL", "REST API", "Linux"],
                "experience_years": 3,
                "education": [
                    {"degree": "MEng Software Engineering", "institution": "University of Manchester", "year": 2021}
                ],
                "linkedin_url": None,
                "cv_filename": None,
                "cv_path": None,
            },
            {
                "first_name": "Clara",
                "last_name": "Müller",
                "email": "clara.muller@example.com",
                "phone": "+44 7700 900003",
                "current_title": "Frontend Developer",
                "skills": ["Angular", "TypeScript", "RxJS", "SCSS", "REST API", "Jest", "Playwright"],
                "experience_years": 4,
                "education": [
                    {"degree": "BSc Web Technologies", "institution": "University of Edinburgh", "year": 2020}
                ],
                "linkedin_url": "https://linkedin.com/in/clara-muller-example",
                "cv_filename": None,
                "cv_path": None,
            },
            {
                "first_name": "David",
                "last_name": "Park",
                "email": "david.park@example.com",
                "phone": "+44 7700 900004",
                "current_title": "ML Research Engineer",
                "skills": [
                    "Python", "PyTorch", "NLP", "OpenAI", "LangChain",
                    "pgvector", "FastAPI", "scikit-learn", "HuggingFace",
                ],
                "experience_years": 5,
                "education": [
                    {"degree": "MSc Artificial Intelligence", "institution": "Imperial College London", "year": 2019}
                ],
                "linkedin_url": "https://linkedin.com/in/david-park-example",
                "cv_filename": None,
                "cv_path": None,
            },
        ]

        seeded_candidates = []
        candidates_created = 0
        for spec in candidate_specs:
            exists = db.query(Candidate).filter(Candidate.email == spec["email"]).first()
            if exists:
                print(f"[SKIP] Candidate already exists: {spec['email']}")
                seeded_candidates.append(exists)
            else:
                candidate = Candidate(**spec)
                db.add(candidate)
                db.flush()
                seeded_candidates.append(candidate)
                candidates_created += 1
                print(f"[OK]   Seeded candidate: {spec['first_name']} {spec['last_name']}")

        db.commit()
        print(f"\nCandidates: {candidates_created} seeded, {len(candidate_specs) - candidates_created} already existed")

        for c in seeded_candidates:
            db.refresh(c)

        # -----------------------------------------------------------------------
        # Applications (pre-processed — decisions set directly, no agent call)
        # -----------------------------------------------------------------------
        # Map by name for clarity
        job_backend   = seeded_jobs[0]   # Senior Backend Engineer
        job_frontend  = seeded_jobs[1]   # Frontend Engineer (Angular)
        job_ml        = seeded_jobs[2]   # ML Engineer

        alice   = seeded_candidates[0]
        ben     = seeded_candidates[1]
        clara   = seeded_candidates[2]
        david   = seeded_candidates[3]

        application_specs = [
            # Alice (strong backend) -> Senior Backend Engineer: SHORTLIST
            {
                "job": job_backend,
                "candidate": alice,
                "status": ApplicationStatus.SHORTLISTED.value,
                "similarity_score": 0.88,
                "skill_coverage": 0.83,
                "rank": 1,
                "llm_decision": "SHORTLIST",
                "llm_reasoning": (
                    "Alice Chen is an excellent match for the Senior Backend Engineer role. "
                    "She has 6 years of experience (exceeds the 5-year requirement) and covers "
                    "5 of 6 required skills. Semantic similarity of 0.88 is well above the "
                    "SHORTLIST threshold. Recommend probing AWS architecture experience in interview."
                ),
                "applied_at": datetime(2026, 2, 15, 10, 0),
            },
            # Ben (mid-level backend, below threshold) -> Senior Backend Engineer: REVIEW
            {
                "job": job_backend,
                "candidate": ben,
                "status": ApplicationStatus.APPLIED.value,
                "similarity_score": 0.61,
                "skill_coverage": 0.50,
                "rank": 2,
                "llm_decision": "REVIEW",
                "llm_reasoning": (
                    "Ben Okafor shows moderate alignment with the Senior Backend Engineer role. "
                    "Similarity score of 0.61 is above the REVIEW threshold but does not meet "
                    "SHORTLIST criteria. He lacks Docker and AWS experience and has only 3 years "
                    "vs the required 5. Recommend manager review to assess growth trajectory."
                ),
                "applied_at": datetime(2026, 2, 16, 9, 0),
            },
            # Clara (frontend specialist) -> Frontend Engineer: SHORTLIST
            {
                "job": job_frontend,
                "candidate": clara,
                "status": ApplicationStatus.SHORTLISTED.value,
                "similarity_score": 0.91,
                "skill_coverage": 1.00,
                "rank": 1,
                "llm_decision": "SHORTLIST",
                "llm_reasoning": (
                    "Clara Muller is an outstanding match for the Frontend Engineer role. "
                    "She covers all 6 required skills with 4 years of Angular experience. "
                    "Semantic similarity of 0.91 is exceptionally high. "
                    "Strong Playwright experience is a direct match for our E2E testing requirements. "
                    "Highly recommend for interview."
                ),
                "applied_at": datetime(2026, 2, 17, 14, 0),
            },
            # David (ML expert) -> ML Engineer: SHORTLIST
            {
                "job": job_ml,
                "candidate": david,
                "status": ApplicationStatus.INTERVIEWING.value,
                "similarity_score": 0.93,
                "skill_coverage": 1.00,
                "rank": 1,
                "llm_decision": "SHORTLIST",
                "llm_reasoning": (
                    "David Park is an exceptional match for the ML Engineer role. "
                    "He covers all 8 required skills and has 5 years of ML engineering experience "
                    "with specific expertise in NLP, pgvector, and LangChain — directly aligned "
                    "with our AI HR tooling. Similarity score 0.93 is the highest seen. "
                    "Interview already scheduled."
                ),
                "applied_at": datetime(2026, 2, 18, 11, 0),
                "interview_date": datetime(2026, 3, 10, 14, 0),
                "interview_notes": "Technical interview — focus on LangChain agent design and pgvector query optimisation",
            },
        ]

        apps_created = 0
        for spec in application_specs:
            job = spec.pop("job")
            candidate = spec.pop("candidate")

            exists = db.query(JobApplication).filter(
                JobApplication.job_id == job.id,
                JobApplication.candidate_id == candidate.id,
            ).first()
            if exists:
                print(f"[SKIP] Application already exists: {candidate.first_name} -> {job.title}")
                continue

            app = JobApplication(
                job_id=job.id,
                candidate_id=candidate.id,
                **spec,
            )
            db.add(app)
            apps_created += 1
            print(f"[OK]   Seeded application: {candidate.first_name} {candidate.last_name} -> {job.title} ({spec['llm_decision']})")

        db.commit()
        print(f"\nApplications: {apps_created} seeded")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
    print("\n[DONE] Hiring seed complete.")
