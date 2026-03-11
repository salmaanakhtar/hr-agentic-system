import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { PayrollService, Payslip } from '../../services/payroll.service';
import { AuthService, User } from '../../auth.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-payslip',
  standalone: true,
  imports: [CommonModule, RouterModule, LlmReasoningDisplayComponent],
  templateUrl: './payslip.component.html',
  styleUrl: './payslip.component.scss',
})
export class PayslipComponent implements OnInit {
  payslip: Payslip | null = null;
  isLoading = false;
  error: string | null = null;
  approving = false;
  approveSuccess = false;
  user: User | null = null;

  constructor(
    private route: ActivatedRoute,
    private payrollService: PayrollService,
    private authService: AuthService,
  ) {}

  ngOnInit(): void {
    this.user = this.authService.getCurrentUser();
    const id = Number(this.route.snapshot.paramMap.get('id'));
    this.loadPayslip(id);
  }

  loadPayslip(id: number): void {
    this.isLoading = true;
    this.error = null;
    this.payrollService.getPayslip(id).subscribe({
      next: (ps) => {
        this.payslip = ps;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Failed to load payslip.';
        this.isLoading = false;
      },
    });
  }

  approve(): void {
    if (!this.payslip) return;
    this.approving = true;
    this.payrollService.approvePayslip(this.payslip.id).subscribe({
      next: () => {
        this.approveSuccess = true;
        this.approving = false;
        if (this.payslip) this.payslip.status = 'approved';
      },
      error: (err) => {
        this.error = err.error?.detail || 'Approval failed.';
        this.approving = false;
      },
    });
  }

  canApprove(): boolean {
    return (
      (this.user?.role === 'hr' || this.user?.role === 'admin') &&
      this.payslip?.status === 'draft'
    );
  }

  getDisplayDecision(d: string | null): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (d === 'APPROVE') return 'AUTO_APPROVE';
    if (d === 'HOLD') return 'ESCALATE';
    return 'REJECT';
  }

  getStatusClass(status: string): string {
    const map: Record<string, string> = {
      approved: 'status-approved',
      draft: 'status-pending',
      paid: 'status-approved',
    };
    return map[status] || 'status-pending';
  }

  formatCurrency(amount: number | null | undefined): string {
    if (amount == null) return '—';
    return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(amount);
  }

  formatDate(d: string | null | undefined): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
    });
  }
}
