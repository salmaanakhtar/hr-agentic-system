import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { LeaveService, LeaveValidationOutput } from '../../services/leave.service';
import { LlmReasoningDisplayComponent } from '../llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-leave-request-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, LlmReasoningDisplayComponent],
  templateUrl: './leave-request-form.component.html',
  styleUrl: './leave-request-form.component.scss'
})
export class LeaveRequestFormComponent implements OnInit {
  leaveForm: FormGroup;
  isSubmitting = false;
  submitSuccess = false;
  submitError = '';
  llmResult: LeaveValidationOutput | null = null;

  leaveTypes = [
    { value: 'vacation', label: 'Vacation' },
    { value: 'sick_leave', label: 'Sick Leave' },
    { value: 'personal', label: 'Personal Leave' },
    { value: 'maternity', label: 'Maternity Leave' },
    { value: 'paternity', label: 'Paternity Leave' },
    { value: 'bereavement', label: 'Bereavement Leave' }
  ];

  constructor(
    private fb: FormBuilder,
    private leaveService: LeaveService,
    private router: Router
  ) {
    this.leaveForm = this.fb.group({
      leave_type: ['vacation', Validators.required],
      start_date: ['', Validators.required],
      end_date: ['', Validators.required],
      reason: ['', [Validators.required, Validators.minLength(10)]]
    }, { validators: this.dateRangeValidator });
  }

  ngOnInit(): void {
    // Set minimum date to today
    const today = new Date().toISOString().split('T')[0];
    this.leaveForm.get('start_date')?.setValue(today);
    this.leaveForm.get('end_date')?.setValue(today);
  }

  dateRangeValidator(group: FormGroup): { [key: string]: boolean } | null {
    const start = group.get('start_date')?.value;
    const end = group.get('end_date')?.value;

    if (start && end && new Date(end) < new Date(start)) {
      return { dateRange: true };
    }
    return null;
  }

  get totalDays(): number {
    const start = this.leaveForm.get('start_date')?.value;
    const end = this.leaveForm.get('end_date')?.value;

    if (start && end) {
      const startDate = new Date(start);
      const endDate = new Date(end);
      const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1;
      return diffDays;
    }
    return 0;
  }

  get minDate(): string {
    return new Date().toISOString().split('T')[0];
  }

  get minEndDate(): string {
    const startDate = this.leaveForm.get('start_date')?.value;
    return startDate || this.minDate;
  }

  onSubmit(): void {
    if (this.leaveForm.invalid) {
      this.leaveForm.markAllAsTouched();
      return;
    }

    this.isSubmitting = true;
    this.submitError = '';
    this.submitSuccess = false;
    this.llmResult = null;

    // Add user_id (backend will override from JWT, but schema requires it)
    const requestData = {
      ...this.leaveForm.value,
      user_id: 0 // Placeholder - backend overrides from token
    };

    this.leaveService.submitLeaveRequest(requestData).subscribe({
      next: (result) => {
        this.isSubmitting = false;
        this.submitSuccess = true;
        this.llmResult = result;

        // Log the full result for debugging
        console.log('Leave request result:', result);

        // Scroll to result
        setTimeout(() => {
          const resultElement = document.querySelector('.result-container');
          resultElement?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
      },
      error: (error) => {
        this.isSubmitting = false;

        // Handle different error formats
        if (typeof error.error?.detail === 'string') {
          this.submitError = error.error.detail;
        } else if (typeof error.error?.detail === 'object') {
          // If detail is an object (validation errors), stringify it
          this.submitError = JSON.stringify(error.error.detail, null, 2);
        } else if (error.error?.message) {
          this.submitError = error.error.message;
        } else {
          this.submitError = `Error ${error.status}: Failed to submit leave request`;
        }

        console.error('Leave request error:', error);
        console.error('Error detail:', error.error);
      }
    });
  }

  viewHistory(): void {
    this.router.navigate(['/leave/history']);
  }

  resetForm(): void {
    this.leaveForm.reset();
    this.leaveForm.get('leave_type')?.setValue('vacation');
    const today = new Date().toISOString().split('T')[0];
    this.leaveForm.get('start_date')?.setValue(today);
    this.leaveForm.get('end_date')?.setValue(today);
    this.submitSuccess = false;
    this.submitError = '';
    this.llmResult = null;
  }

  // Map backend fields to display component expectations
  get llmDecision(): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (!this.llmResult) return 'AUTO_APPROVE';

    if (!this.llmResult.is_valid) {
      return 'REJECT';
    }
    if (this.llmResult.auto_approval_eligible) {
      return 'AUTO_APPROVE';
    }
    return 'ESCALATE';
  }

  get llmConfidence(): number {
    if (!this.llmResult) return 0;

    // Calculate confidence based on validation state
    if (!this.llmResult.is_valid) return 0.9; // High confidence rejection
    if (this.llmResult.auto_approval_eligible && this.llmResult.conflicts.length === 0) {
      return 0.95; // Very high confidence approval
    }
    if (this.llmResult.conflicts.length > 0) {
      return 0.6; // Medium confidence with conflicts
    }
    return 0.8; // Default confidence
  }

  get llmFactors(): string[] {
    if (!this.llmResult) return [];
    return this.llmResult.validation_warnings || [];
  }

  get llmRecommendations(): string[] {
    if (!this.llmResult) return [];
    return this.llmResult.recommended_actions || [];
  }
}
