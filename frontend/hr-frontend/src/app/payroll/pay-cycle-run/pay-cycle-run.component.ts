import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { PayrollService, PayCycleRunResult } from '../../services/payroll.service';

@Component({
  selector: 'app-pay-cycle-run',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './pay-cycle-run.component.html',
  styleUrl: './pay-cycle-run.component.scss',
})
export class PayCycleRunComponent {
  periodStart = '';
  periodEnd = '';

  isRunning = false;
  error: string | null = null;
  result: PayCycleRunResult | null = null;

  constructor(
    private payrollService: PayrollService,
    private router: Router,
  ) {}

  runCycle(): void {
    if (!this.periodStart || !this.periodEnd) return;
    this.isRunning = true;
    this.error = null;
    this.result = null;

    this.payrollService.runCycle(this.periodStart, this.periodEnd).subscribe({
      next: (res) => {
        this.result = res;
        this.isRunning = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Pay cycle run failed.';
        this.isRunning = false;
      },
    });
  }

  reset(): void {
    this.result = null;
    this.error = null;
  }

  viewPayslip(payslipId: number | null): void {
    if (payslipId) this.router.navigate(['/payroll/payslips', payslipId]);
  }

  getDecisionClass(decision: string): string {
    if (decision === 'APPROVE') return 'decision-approve';
    if (decision === 'FLAG') return 'decision-flag';
    return 'decision-hold';
  }

  formatCurrency(amount: number | null): string {
    if (amount == null) return '—';
    return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(amount);
  }
}
