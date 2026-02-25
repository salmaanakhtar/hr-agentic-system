import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { OCRExtractionResult } from '../../services/expense.service';

export interface SubmittedExpenseData {
  amount: number;
  vendor: string | null;
  date: string;
}

@Component({
  selector: 'app-ocr-result-display',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './ocr-result-display.component.html',
  styleUrl: './ocr-result-display.component.scss'
})
export class OcrResultDisplayComponent {
  @Input() ocrResult: OCRExtractionResult | null = null;
  @Input() submitted: SubmittedExpenseData | null = null;
  @Input() receiptUrl: string | null = null;

  get overallConfidencePct(): number {
    return this.ocrResult ? Math.round(this.ocrResult.overall_confidence * 100) : 0;
  }

  get isLowConfidence(): boolean {
    return this.overallConfidencePct < 70;
  }

  confidencePct(value: number): number {
    return Math.round(value * 100);
  }

  getConfidenceClass(value: number): string {
    const pct = Math.round(value * 100);
    if (pct >= 70) return 'confidence-high';
    if (pct >= 40) return 'confidence-medium';
    return 'confidence-low';
  }

  amountMismatch(): boolean {
    if (!this.ocrResult?.amount || !this.submitted?.amount) return false;
    const diff = Math.abs(this.ocrResult.amount - this.submitted.amount);
    return diff / this.submitted.amount > 0.05; // >5% difference
  }

  vendorMismatch(): boolean {
    if (!this.ocrResult?.vendor || !this.submitted?.vendor) return false;
    return this.ocrResult.vendor.toLowerCase().trim() !== this.submitted.vendor.toLowerCase().trim();
  }

  dateMismatch(): boolean {
    if (!this.ocrResult?.date || !this.submitted?.date) return false;
    return this.ocrResult.date !== this.submitted.date;
  }
}
