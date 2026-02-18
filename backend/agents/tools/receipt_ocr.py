"""
Receipt OCR Tool for Phase 4 Expense Agent.

Wraps the existing EasyOCR infrastructure with receipt-specific parsing:
- Extracts vendor, amount, and date from receipt images
- Returns per-field confidence scores
- Returns OCRExtractionResult schema used by ExpenseAgent tools
"""

import re
import logging
from pathlib import Path
from typing import Optional

from agents.schemas import OCRExtractionResult

logger = logging.getLogger(__name__)

# Lazy-loaded singleton so EasyOCR only initialises once per process
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Initialising EasyOCR reader (CPU mode)...")
            _reader = easyocr.Reader(["en"], gpu=False)
            logger.info("[OK] EasyOCR reader ready")
        except ImportError:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
    return _reader


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_raw_text(file_path: str) -> tuple[str, list[tuple]]:
    """
    Run EasyOCR on an image file.
    Returns (joined_text, raw_results) where raw_results is a list of
    (bbox, text, confidence) tuples from EasyOCR.
    """
    reader = _get_reader()
    raw_results = reader.readtext(file_path)
    lines = [text for (_, text, conf) in raw_results if conf > 0.1]
    return "\n".join(lines), raw_results


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

_AMOUNT_PATTERNS = [
    # Total / Amount Due with currency symbol
    r"(?:total|amount\s*due|grand\s*total|subtotal|balance\s*due)[^\d]*\$?\s*([\d,]+\.\d{2})",
    # Standalone dollar amount (largest match preferred)
    r"\$\s*([\d,]+\.\d{2})",
    # Plain decimal number that looks like a price
    r"\b([\d,]+\.\d{2})\b",
]

_DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",            # 12/31/2024 or 31-12-2024
    r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",              # 2024-12-31
    r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})\b",
    r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",
]


def _parse_amount(text: str) -> tuple[Optional[float], float]:
    """
    Returns (amount, confidence).
    Tries patterns in priority order; later (weaker) patterns get lower confidence.
    """
    lower = text.lower()
    for idx, pattern in enumerate(_AMOUNT_PATTERNS):
        matches = re.findall(pattern, lower, re.IGNORECASE)
        if matches:
            # Clean and take the largest value found (likely the total)
            values = []
            for m in matches:
                try:
                    values.append(float(m.replace(",", "")))
                except ValueError:
                    continue
            if values:
                confidence = max(0.0, 0.9 - idx * 0.2)  # 0.9, 0.7, 0.5
                return max(values), round(confidence, 2)
    return None, 0.0


def _parse_date(text: str) -> tuple[Optional[str], float]:
    """
    Returns (date_string, confidence).
    """
    for idx, pattern in enumerate(_DATE_PATTERNS):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            confidence = max(0.0, 0.9 - idx * 0.15)
            return match.group(1).strip(), round(confidence, 2)
    return None, 0.0


def _parse_vendor(raw_results: list[tuple]) -> tuple[Optional[str], float]:
    """
    Returns (vendor_name, confidence).
    Heuristic: the first high-confidence text line that is not purely numeric
    and is long enough to be a business name.
    """
    for (_, text, conf) in raw_results:
        cleaned = text.strip()
        # Skip lines that are just numbers, dates, or very short
        if len(cleaned) < 3:
            continue
        if re.match(r"^[\d\s\$\.,/:%-]+$", cleaned):
            continue
        # Looks like a name
        return cleaned, round(min(conf, 0.9), 2)
    return None, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_receipt(file_path: str) -> OCRExtractionResult:
    """
    Main entry point. Runs OCR on the receipt image at file_path and
    returns a fully populated OCRExtractionResult.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Receipt file not found: {file_path}")
        return OCRExtractionResult(raw_text="", overall_confidence=0.0)

    try:
        raw_text, raw_results = extract_raw_text(file_path)
    except Exception as exc:
        logger.error(f"OCR extraction failed for {file_path}: {exc}")
        return OCRExtractionResult(raw_text="", overall_confidence=0.0)

    vendor, vendor_conf = _parse_vendor(raw_results)
    amount, amount_conf = _parse_amount(raw_text)
    date_str, date_conf = _parse_date(raw_text)

    # Overall confidence: weighted average (amount matters most for expense validation)
    weights = {"amount": 0.5, "vendor": 0.3, "date": 0.2}
    overall = round(
        amount_conf * weights["amount"]
        + vendor_conf * weights["vendor"]
        + date_conf * weights["date"],
        2,
    )

    logger.info(
        f"OCR result - vendor: {vendor} ({vendor_conf}), "
        f"amount: {amount} ({amount_conf}), "
        f"date: {date_str} ({date_conf}), "
        f"overall: {overall}"
    )

    return OCRExtractionResult(
        raw_text=raw_text,
        vendor=vendor,
        amount=amount,
        date=date_str,
        vendor_confidence=vendor_conf,
        amount_confidence=amount_conf,
        date_confidence=date_conf,
        overall_confidence=overall,
    )
