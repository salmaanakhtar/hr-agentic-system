import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { PolicyService, PolicyQueryResult } from '../../services/policy.service';

@Component({
  selector: 'app-policy-query',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './policy-query.component.html',
  styleUrl: './policy-query.component.scss',
})
export class PolicyQueryComponent {
  question = '';
  isLoading = false;
  error: string | null = null;
  result: PolicyQueryResult | null = null;

  constructor(private policyService: PolicyService) {}

  submit(): void {
    if (!this.question.trim()) return;
    this.isLoading = true;
    this.error = null;
    this.result = null;

    this.policyService.queryPolicy(this.question.trim()).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Query failed. Please try again.';
        this.isLoading = false;
      },
    });
  }

  reset(): void {
    this.question = '';
    this.result = null;
    this.error = null;
  }

  confidencePercent(): number {
    return Math.round((this.result?.confidence ?? 0) * 100);
  }

  confidenceClass(): string {
    const pct = this.confidencePercent();
    if (pct >= 70) return 'confidence-high';
    if (pct >= 40) return 'confidence-medium';
    return 'confidence-low';
  }
}
