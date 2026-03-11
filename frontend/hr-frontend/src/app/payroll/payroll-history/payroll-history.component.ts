import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { PayrollService, PayCycle, Payslip } from '../../services/payroll.service';
import { AuthService, User } from '../../auth.service';

@Component({
  selector: 'app-payroll-history',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './payroll-history.component.html',
  styleUrl: './payroll-history.component.scss',
})
export class PayrollHistoryComponent implements OnInit {
  user: User | null = null;
  isHrAdmin = false;

  // HR/Admin: cycles list
  cycles: PayCycle[] = [];
  selectedStatus = '';
  expandedCycleId: number | null = null;
  expandedPayslips: Payslip[] = [];
  loadingPayslips = false;

  // Employee: own payslips
  payslips: Payslip[] = [];

  isLoading = false;
  error: string | null = null;

  readonly cycleStatuses = [
    { value: '', label: 'All Statuses' },
    { value: 'completed', label: 'Completed' },
    { value: 'running', label: 'Running' },
    { value: 'failed', label: 'Failed' },
  ];

  constructor(
    private payrollService: PayrollService,
    private authService: AuthService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.user = this.authService.getCurrentUser();
    this.isHrAdmin = this.user?.role === 'hr' || this.user?.role === 'admin';
    this.load();
  }

  load(): void {
    this.isLoading = true;
    this.error = null;

    if (this.isHrAdmin) {
      this.payrollService.getCycles(this.selectedStatus || undefined).subscribe({
        next: (data) => { this.cycles = data; this.isLoading = false; },
        error: (err) => {
          this.error = err.error?.detail || 'Failed to load pay cycles.';
          this.isLoading = false;
        },
      });
    } else {
      this.payrollService.getPayslips().subscribe({
        next: (data) => { this.payslips = data; this.isLoading = false; },
        error: (err) => {
          this.error = err.error?.detail || 'Failed to load payslips.';
          this.isLoading = false;
        },
      });
    }
  }

  onFilterChange(): void {
    this.expandedCycleId = null;
    this.expandedPayslips = [];
    this.load();
  }

  toggleCycle(cycle: PayCycle): void {
    if (this.expandedCycleId === cycle.id) {
      this.expandedCycleId = null;
      this.expandedPayslips = [];
      return;
    }
    this.expandedCycleId = cycle.id;
    this.loadingPayslips = true;
    this.payrollService.getPayslips(cycle.id).subscribe({
      next: (ps) => { this.expandedPayslips = ps; this.loadingPayslips = false; },
      error: () => { this.loadingPayslips = false; },
    });
  }

  viewPayslip(id: number): void {
    this.router.navigate(['/payroll/payslips', id]);
  }

  getStatusClass(status: string): string {
    const map: Record<string, string> = {
      completed: 'status-approved',
      running: 'status-pending',
      failed: 'status-rejected',
      pending: 'status-pending',
      approved: 'status-approved',
      draft: 'status-pending',
      paid: 'status-approved',
    };
    return map[status] || 'status-pending';
  }

  getDecisionClass(decision: string | null): string {
    if (decision === 'APPROVE') return 'decision-approve';
    if (decision === 'FLAG') return 'decision-flag';
    return 'decision-hold';
  }

  formatDate(d: string | null | undefined): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
    });
  }

  formatCurrency(amount: number | null | undefined): string {
    if (amount == null) return '—';
    return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(amount);
  }
}
