"""
Seed policy documents: generate 3 PDF policy files using fpdf2, chunk + embed
each one, and store in the policy_documents / policy_chunks tables.

Run once: python seed_policy_data.py
Idempotent: skips documents that already exist (matched by title).
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import openai
from fpdf import FPDF

from agents.tools.policy_parser import parse_and_chunk_policy
from app.database import SessionLocal
from app.models import PolicyChunk, PolicyDocument, User


_BACKEND_DIR = Path(__file__).resolve().parent
_UPLOADS_DIR = _BACKEND_DIR / "uploads" / "policies"


# ---------------------------------------------------------------------------
# PDF generation helpers
# ---------------------------------------------------------------------------

class _PolicyPDF(FPDF):
    def __init__(self, doc_title: str):
        super().__init__()
        self._doc_title = doc_title

    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, self._doc_title, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def add_section(self, title: str, body: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 6, body)
        self.ln(4)


def _build_pdf(doc_title: str, sections: list, file_path: Path) -> None:
    """Render a list of (section_title, body_text) tuples into a PDF file."""
    pdf = _PolicyPDF(doc_title)
    pdf.add_page()
    for section_title, body in sections:
        pdf.add_section(section_title, body)
    pdf.output(str(file_path))


# ---------------------------------------------------------------------------
# Policy content
# ---------------------------------------------------------------------------

_EXPENSE_SECTIONS = [
    (
        "1. Purpose and Scope",
        (
            "This Expense Reimbursement Policy governs the submission, review, and approval of all "
            "employee business expenses. It applies to all permanent and contract employees who incur "
            "expenses on behalf of the company during the course of their duties. The company is "
            "committed to reimbursing legitimate business expenses promptly and fairly while ensuring "
            "responsible use of company funds. All employees are expected to act honestly and in "
            "accordance with this policy when submitting expense claims."
        ),
    ),
    (
        "2. Expense Categories and Approval Thresholds",
        (
            "All expense claims are classified by category. Each category has an auto-approval "
            "limit and a hard maximum. Claims within the auto-approval limit are approved "
            "automatically provided all other conditions are met. Claims between the auto-approval "
            "limit and the hard maximum require manager approval. Claims exceeding the hard maximum "
            "are rejected and will not be reimbursed.\n\n"
            "Meals and Subsistence: Auto-approved up to 25 GBP per claim. Claims between 25 GBP "
            "and 50 GBP require manager approval. Claims exceeding 50 GBP are rejected.\n\n"
            "Travel and Transportation: Auto-approved up to 100 GBP per claim. Claims between "
            "100 GBP and 200 GBP require manager approval. Claims exceeding 200 GBP are rejected.\n\n"
            "Equipment and Hardware: Auto-approved up to 200 GBP per claim. Claims between 200 GBP "
            "and 500 GBP require manager approval. Claims exceeding 500 GBP are rejected.\n\n"
            "Entertainment and Client Events: Auto-approved up to 50 GBP per claim. Claims between "
            "50 GBP and 100 GBP require manager approval. Claims exceeding 100 GBP are rejected.\n\n"
            "Office Supplies and Stationery: Auto-approved up to 50 GBP per claim. Claims between "
            "50 GBP and 100 GBP require manager approval. Claims exceeding 100 GBP are rejected.\n\n"
            "Other Business Expenses: Auto-approved up to 50 GBP per claim. Claims between 50 GBP "
            "and 150 GBP require manager approval. Claims exceeding 150 GBP are rejected."
        ),
    ),
    (
        "3. Receipt Requirements and OCR Validation",
        (
            "A receipt or proof of purchase must be submitted for all claims exceeding 25 GBP. "
            "Receipts must clearly show the vendor name, transaction date, and total amount paid. "
            "Digital receipts and photographs of physical receipts are accepted provided the image "
            "is legible and complete. The automated OCR system will extract and verify key fields "
            "from uploaded receipts. If the OCR-extracted amount differs from the submitted amount "
            "by more than 20 percent, the claim will be flagged for manual review. An overall OCR "
            "confidence score below 0.40 indicates an unreadable receipt and the claim will be "
            "rejected. Confidence between 0.40 and 0.70 will result in escalation for manual "
            "receipt inspection by a manager."
        ),
    ),
    (
        "4. Duplicate Claim Detection",
        (
            "The system automatically checks each new expense claim against previous submissions "
            "within a 7-day window. Any claim from the same employee with a similar amount (within "
            "10 percent) and the same vendor submitted within 7 days will be flagged as a potential "
            "duplicate and rejected. Employees must contact their manager directly if they have a "
            "legitimate reason to submit what appears to be a duplicate claim. Intentional duplicate "
            "submission will be treated as a disciplinary matter."
        ),
    ),
    (
        "5. Monthly Spending Limits",
        (
            "Each employee has a monthly spending limit per category set at three times the hard "
            "maximum for that category. When cumulative approved expenses for a category in a "
            "calendar month reach 80 percent of the monthly limit, subsequent claims in that "
            "category are automatically escalated to the manager for review regardless of the "
            "individual claim amount. Employees approaching their monthly limit will be notified "
            "by the HR system so they can plan spending accordingly."
        ),
    ),
    (
        "6. Non-Reimbursable Expenses",
        (
            "The following types of expenses are not eligible for reimbursement under any "
            "circumstances: personal items unrelated to business duties, alcoholic beverages unless "
            "explicitly pre-approved as part of a client entertainment event, traffic fines and "
            "penalties, personal travel or holiday costs, and expenses for spouses or family members "
            "unless pre-approved in writing for a specific business purpose. Expenses that fall "
            "outside the approved categories must be pre-approved by the employee's manager and HR "
            "before they are incurred."
        ),
    ),
]

_LEAVE_SECTIONS = [
    (
        "1. Purpose and Scope",
        (
            "This Leave Policy sets out the entitlements, procedures, and conditions for all "
            "categories of employee absence. It applies to all permanent employees of the "
            "organisation. The company recognises the importance of a healthy work-life balance "
            "and is committed to administering leave entitlements fairly and consistently. All "
            "leave requests must be submitted through the HR system and are subject to the "
            "approval process described in this policy."
        ),
    ),
    (
        "2. Types of Leave and Annual Entitlements",
        (
            "Annual Vacation Leave: All permanent employees are entitled to 20 working days of "
            "paid vacation leave per calendar year. Vacation leave must be scheduled in advance "
            "wherever possible and is subject to manager approval for requests of 4 days or more.\n\n"
            "Sick Leave: Employees are entitled to 10 days of paid sick leave per calendar year. "
            "Sick leave may only be taken when an employee is genuinely unwell. A medical "
            "certificate may be required for absences exceeding 3 consecutive sick days.\n\n"
            "Personal Leave: Employees are entitled to 5 days of paid personal leave per calendar "
            "year for personal matters that cannot be scheduled outside working hours.\n\n"
            "Maternity Leave: Eligible employees are entitled to up to 26 weeks of maternity leave. "
            "The first 6 weeks are paid at full salary; the remaining 20 weeks are at statutory pay "
            "rates. Employees must provide at least 8 weeks notice of their expected start date.\n\n"
            "Paternity Leave: Eligible employees are entitled to 2 weeks of paternity leave at full "
            "pay, to be taken within 8 weeks of the child's birth or adoption placement.\n\n"
            "Bereavement Leave: 5 days paid leave for the death of an immediate family member "
            "(spouse, child, parent, or sibling); 3 days for other close relatives."
        ),
    ),
    (
        "3. Notice Requirements and Auto-Approval Rules",
        (
            "Minimum Advance Notice: All leave requests must be submitted at least 3 working days "
            "in advance. Requests submitted with less than 3 working days notice may still be "
            "considered but will require manager approval regardless of duration.\n\n"
            "Auto-Approval: Leave requests of 3 working days or fewer will be automatically "
            "approved by the system provided all of the following conditions are met: the employee "
            "has sufficient leave balance remaining for the requested leave type, there are no "
            "conflicts with existing approved leave requests, there are no conflicts with company "
            "blackout periods, and the request meets the 3-day minimum advance notice requirement.\n\n"
            "Manager Approval: Leave requests of 4 to 14 working days require manager approval. "
            "Requests with short notice but otherwise valid will also be escalated to the manager. "
            "The manager must respond within 2 working days of receiving the approval request."
        ),
    ),
    (
        "4. Maximum Consecutive Days and Blackout Periods",
        (
            "Maximum Consecutive Days: Employees may not take more than 14 consecutive working "
            "days of leave without explicit approval from their line manager and HR. Requests "
            "exceeding 14 consecutive working days will be escalated for approval regardless of "
            "other conditions.\n\n"
            "Blackout Periods: The company designates certain dates as blackout periods during "
            "which leave requests will not be approved due to business-critical demands such as "
            "year-end financial closing, major product launches, or peak operational periods. "
            "Blackout period dates are communicated to employees at least 6 weeks in advance. "
            "Leave requested that overlaps with a blackout period is automatically rejected. "
            "Exceptions require written approval from a department head and HR."
        ),
    ),
    (
        "5. Leave Balances, Deductions, and Carry-Forward",
        (
            "Insufficient Balance: If an employee requests leave for which they have insufficient "
            "balance remaining, the request will be automatically rejected. Employees may not take "
            "leave in advance of their accrued entitlement without prior written approval from HR.\n\n"
            "Unpaid Leave: If an employee exhausts their leave balance and requires additional time "
            "off, unpaid leave may be approved at the manager's and HR's discretion. Unpaid leave "
            "days result in a proportional salary deduction calculated as the daily rate multiplied "
            "by the number of unpaid leave days taken in the pay period.\n\n"
            "Annual Carry-Forward: Unused vacation leave of up to 5 days may be carried forward to "
            "the following calendar year and must be used before the end of March. Any carry-forward "
            "leave unused by that date will be forfeited. Sick leave and personal leave do not carry "
            "forward to the following year."
        ),
    ),
]

_GENERAL_SECTIONS = [
    (
        "1. Introduction",
        (
            "This General HR Policy and Code of Conduct establishes the standards of behaviour, "
            "professional conduct, and workplace values expected of all employees. All employees "
            "are required to read, understand, and comply with this policy. Adherence to this "
            "policy is a condition of employment. The company reserves the right to update this "
            "policy and will provide reasonable notice of any material changes to all staff."
        ),
    ),
    (
        "2. Equal Opportunity and Non-Discrimination",
        (
            "The company is committed to equal opportunity in all aspects of employment including "
            "recruitment, training, promotion, and termination. Employment decisions are based "
            "solely on merit, qualifications, and business requirements. Discrimination, harassment, "
            "or victimisation based on age, gender, race, ethnicity, religion, disability, sexual "
            "orientation, or any other protected characteristic is strictly prohibited. Any "
            "employee found to have engaged in discriminatory behaviour will be subject to "
            "disciplinary action up to and including termination of employment. The company also "
            "commits to making reasonable adjustments for employees with disabilities."
        ),
    ),
    (
        "3. Professional Conduct and Workplace Behaviour",
        (
            "All employees are expected to conduct themselves in a professional, respectful, and "
            "collaborative manner at all times. This includes interactions with colleagues, "
            "clients, suppliers, and any other third parties encountered in the course of work. "
            "Employees must avoid behaviour that could bring the company into disrepute, including "
            "inappropriate use of social media, misrepresentation of the company's products or "
            "services, and conduct that could create a hostile work environment. Concerns about "
            "workplace behaviour should be raised through the formal grievance process."
        ),
    ),
    (
        "4. Confidentiality and Data Protection",
        (
            "All employees have a duty to maintain the confidentiality of company information, "
            "client data, and personal data of colleagues. Confidential information must not be "
            "disclosed to unauthorised parties inside or outside the organisation without proper "
            "authorisation. Employees who handle personal data must comply with applicable data "
            "protection laws including the UK GDPR. Personal data must be collected and processed "
            "only for specific, legitimate purposes and must not be retained longer than necessary. "
            "Any data breach or suspected breach must be reported to the Data Protection Officer "
            "within 24 hours of discovery."
        ),
    ),
    (
        "5. Working Hours and Remote Work",
        (
            "Standard working hours are Monday to Friday, 09:00 to 17:30, with a 30-minute "
            "unpaid lunch break. Employees are expected to work their contracted hours. Flexible "
            "working arrangements may be agreed with the line manager and HR subject to business "
            "requirements. Remote working is available for roles where operationally feasible and "
            "must be approved by the employee's manager. Employees working remotely are expected "
            "to maintain the same level of availability, output, and professional standards as "
            "when working on-site."
        ),
    ),
    (
        "6. Performance Reviews and Development",
        (
            "All employees undergo a formal performance review at least once per year. Reviews "
            "are conducted by the employee's line manager and focus on achievement against "
            "objectives, demonstration of company values, and identification of development "
            "opportunities. Employees are encouraged to provide self-assessments and identify "
            "their own learning and development needs. Salary reviews are conducted annually and "
            "are informed by the performance review outcome, market benchmarking, and the "
            "company's financial performance for the year."
        ),
    ),
    (
        "7. Disciplinary Procedures",
        (
            "Where an employee's conduct or performance falls below the expected standard the "
            "company will follow a fair and transparent disciplinary process. The stages are: "
            "informal discussion, formal verbal warning, written warning, final written warning, "
            "and termination of employment. The appropriate starting point depends on the severity "
            "of the issue. Gross misconduct, including but not limited to theft, fraud, violence, "
            "serious breach of confidentiality, and harassment, may result in immediate termination "
            "without prior warning. All employees have the right to be accompanied by a colleague "
            "or trade union representative at any formal disciplinary hearing."
        ),
    ),
    (
        "8. Grievance Process",
        (
            "Employees who have a concern or complaint about their working conditions, treatment, "
            "or any aspect of their employment are encouraged to raise it promptly. In the first "
            "instance employees should attempt to resolve the matter informally with their line "
            "manager. If this is not possible or appropriate, a formal grievance may be submitted "
            "in writing to HR. HR will acknowledge the grievance within 5 working days and arrange "
            "a formal hearing. Employees have the right to be accompanied at the hearing. A written "
            "outcome will be provided within 10 working days of the hearing. Employees who are "
            "dissatisfied with the outcome may appeal to senior management within 10 working days "
            "of receiving the outcome letter."
        ),
    ),
]

_POLICIES = [
    {
        "title": "Employee Expense Reimbursement Policy",
        "description": "Defines expense categories, approval thresholds, receipt requirements, and reimbursement procedures.",
        "category": "expense",
        "sections": _EXPENSE_SECTIONS,
    },
    {
        "title": "Employee Leave and Absence Policy",
        "description": "Covers leave entitlements, notice requirements, auto-approval rules, and blackout periods.",
        "category": "leave",
        "sections": _LEAVE_SECTIONS,
    },
    {
        "title": "General HR Policy and Code of Conduct",
        "description": "Sets out standards of professional conduct, equal opportunity, confidentiality, and disciplinary procedures.",
        "category": "general",
        "sections": _GENERAL_SECTIONS,
    },
]


# ---------------------------------------------------------------------------
# Seed function
# ---------------------------------------------------------------------------

def seed():
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        uploader = db.query(User).first()
        if not uploader:
            print("[ERROR] No users found — start the app and register a user first")
            return

        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        total_docs = 0
        total_chunks = 0

        for policy in _POLICIES:
            title = policy["title"]

            existing = db.query(PolicyDocument).filter(PolicyDocument.title == title).first()
            if existing:
                print(f"[SKIP] '{title}' already exists (id={existing.id})")
                continue

            # Generate PDF
            unique_filename = f"{uuid.uuid4()}.pdf"
            file_path = _UPLOADS_DIR / unique_filename

            print(f"\n[BUILDING] '{title}'...")
            _build_pdf(title, policy["sections"], file_path)
            print(f"  PDF saved: {unique_filename}")

            # Parse and chunk (synchronous)
            chunks_text = parse_and_chunk_policy(str(file_path))
            if not chunks_text:
                print(f"  [WARN] No chunks extracted — skipping '{title}'")
                file_path.unlink(missing_ok=True)
                continue
            print(f"  Chunks extracted: {len(chunks_text)}")

            # Embed all chunks in a single batch API call
            embed_response = openai_client.embeddings.create(
                input=chunks_text,
                model="text-embedding-3-small",
            )
            embeddings = [e.embedding for e in embed_response.data]
            print(f"  Embeddings created: {len(embeddings)}")

            # Store PolicyDocument
            doc = PolicyDocument(
                title=title,
                description=policy["description"],
                category=policy["category"],
                filename=unique_filename,
                file_path=str(file_path),
                uploaded_by=uploader.id,
                is_active=True,
            )
            db.add(doc)
            db.flush()
            db.refresh(doc)

            # Store PolicyChunks
            for idx, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
                chunk = PolicyChunk(
                    document_id=doc.id,
                    content=chunk_text,
                    chunk_index=idx,
                    embedding=embedding,
                    token_count=len(chunk_text.split()),
                )
                db.add(chunk)

            db.commit()
            total_docs += 1
            total_chunks += len(chunks_text)
            print(f"  [OK] Document id={doc.id}, {len(chunks_text)} chunk(s) stored")

        print(f"\n[DONE] Policy seed complete: {total_docs} document(s), {total_chunks} chunk(s) total")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
