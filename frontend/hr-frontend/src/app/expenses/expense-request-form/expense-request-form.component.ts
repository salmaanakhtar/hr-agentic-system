import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { ExpenseService, ExpenseValidationOutput } from '../../services/expense.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';
import { OcrResultDisplayComponent, SubmittedExpenseData } from '../ocr-result-display/ocr-result-display.component';

@Component({
  selector: 'app-expense-request-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, LlmReasoningDisplayComponent, OcrResultDisplayComponent],
  templateUrl: './expense-request-form.component.html',
  styleUrl: './expense-request-form.component.scss'
})
export class ExpenseRequestFormComponent implements OnInit {
  form: FormGroup;
  isSubmitting = false;
  result: ExpenseValidationOutput | null = null;
  error: string | null = null;

  selectedFile: File | null = null;
  fileError: string | null = null;
  previewUrl: string | null = null;
  isDragging = false;

  readonly categories = [
    { value: 'meals', label: 'Meals & Dining' },
    { value: 'travel', label: 'Travel' },
    { value: 'equipment', label: 'Equipment' },
    { value: 'entertainment', label: 'Entertainment' },
    { value: 'office_supplies', label: 'Office Supplies' },
    { value: 'other', label: 'Other' },
  ];

  constructor(
    private fb: FormBuilder,
    private expenseService: ExpenseService,
    private router: Router
  ) {
    this.form = this.fb.group({
      category: ['', Validators.required],
      amount: ['', [Validators.required, Validators.min(0.01)]],
      vendor: [''],
      date: ['', Validators.required],
      description: [''],
    });
  }

  ngOnInit(): void {
    // Default date to today
    const today = new Date().toISOString().split('T')[0];
    this.form.patchValue({ date: today });
  }

  get maxDate(): string {
    return new Date().toISOString().split('T')[0];
  }

  // File handling
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.handleFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(): void {
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    if (event.dataTransfer?.files[0]) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  private handleFile(file: File): void {
    this.fileError = null;
    const allowed = ['image/jpeg', 'image/png', 'application/pdf'];
    if (!allowed.includes(file.type)) {
      this.fileError = 'Invalid file type. Please upload a JPG, PNG, or PDF.';
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      this.fileError = 'File too large. Maximum size is 5MB.';
      return;
    }
    this.selectedFile = file;

    // Preview for images only
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.previewUrl = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    } else {
      this.previewUrl = null;
    }
  }

  removeFile(): void {
    this.selectedFile = null;
    this.previewUrl = null;
    this.fileError = null;
  }

  onSubmit(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    this.isSubmitting = true;
    this.result = null;
    this.error = null;

    const formData = new FormData();
    formData.append('amount', this.form.value.amount);
    formData.append('category', this.form.value.category);
    formData.append('date', this.form.value.date);
    if (this.form.value.vendor) formData.append('vendor', this.form.value.vendor);
    if (this.form.value.description) formData.append('description', this.form.value.description);
    if (this.selectedFile) formData.append('receipt', this.selectedFile);

    this.expenseService.submitExpense(formData).subscribe({
      next: (res) => {
        this.isSubmitting = false;
        this.result = res;
      },
      error: (err) => {
        this.isSubmitting = false;
        this.error = err.error?.detail || 'Failed to submit expense. Please try again.';
      }
    });
  }

  get submittedData(): SubmittedExpenseData | null {
    if (!this.form.value.amount) return null;
    return {
      amount: parseFloat(this.form.value.amount),
      vendor: this.form.value.vendor || null,
      date: this.form.value.date,
    };
  }

  get receiptUrl(): string | null {
    if (!this.result?.ocr_result) return null;
    // The expense_id is set after submission - build receipt URL if available
    return this.previewUrl; // Show local preview if available
  }

  getDecision(decision: string): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (decision === 'AUTO_APPROVE') return 'AUTO_APPROVE';
    if (decision === 'ESCALATE') return 'ESCALATE';
    return 'REJECT';
  }

  goToHistory(): void {
    this.router.navigate(['/expenses/history']);
  }

  isFieldInvalid(field: string): boolean {
    const control = this.form.get(field);
    return !!(control && control.invalid && control.touched);
  }
}
