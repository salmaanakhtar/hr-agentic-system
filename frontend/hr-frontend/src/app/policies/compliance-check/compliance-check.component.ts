import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { PolicyService, ComplianceCheckResult } from '../../services/policy.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-compliance-check',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule, LlmReasoningDisplayComponent],
  templateUrl: './compliance-check.component.html',
  styleUrl: './compliance-check.component.scss',
})
export class ComplianceCheckComponent {
  scenario = '';
  isLoading = false;
  error: string | null = null;
  result: ComplianceCheckResult | null = null;

  constructor(private policyService: PolicyService) {}

  submit(): void {
    if (!this.scenario.trim()) return;
    this.isLoading = true;
    this.error = null;
    this.result = null;

    this.policyService.checkCompliance(this.scenario.trim()).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Compliance check failed. Please try again.';
        this.isLoading = false;
      },
    });
  }

  reset(): void {
    this.scenario = '';
    this.result = null;
    this.error = null;
  }

  // Map policy decisions to LlmReasoningDisplayComponent format
  getDisplayDecision(d: string | null): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (d === 'COMPLIANT') return 'AUTO_APPROVE';
    if (d === 'NON_COMPLIANT') return 'REJECT';
    return 'ESCALATE';
  }

  decisionLabel(d: string | null): string {
    if (d === 'COMPLIANT') return 'Compliant';
    if (d === 'NON_COMPLIANT') return 'Non-Compliant';
    return 'Unclear';
  }

  decisionClass(d: string | null): string {
    if (d === 'COMPLIANT') return 'decision-compliant';
    if (d === 'NON_COMPLIANT') return 'decision-non-compliant';
    return 'decision-unclear';
  }

  confidencePercent(): number {
    return Math.round((this.result?.confidence ?? 0) * 100);
  }
}
