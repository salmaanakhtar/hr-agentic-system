import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { AuthService, User } from '../auth.service';
import { LeaveService } from '../services/leave.service';
import { ExpenseService } from '../services/expense.service';
import { PayrollService } from '../services/payroll.service';

interface KpiCard {
  label: string;
  value: string | number;
  sublabel: string;
  colorClass: string;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  user: User | null = null;
  kpiCards: KpiCard[] = [];
  kpiLoading = true;

  constructor(
    private authService: AuthService,
    private router: Router,
    private leaveService: LeaveService,
    private expenseService: ExpenseService,
    private payrollService: PayrollService
  ) {}

  ngOnInit(): void {
    if (!this.authService.isLoggedIn()) {
      this.router.navigate(['/login']);
      return;
    }

    this.user = this.authService.getCurrentUser();
    this.authService.currentUser$.subscribe(user => {
      this.user = user;
    });

    if (this.user?.role.toLowerCase() === 'employee') {
      this.loadEmployeeKpis();
    } else {
      this.kpiLoading = false;
    }
  }

  private loadEmployeeKpis(): void {
    const leaveColors = ['kpi-indigo', 'kpi-blue', 'kpi-green'];

    forkJoin({
      leaveBalances: this.leaveService.getLeaveBalances().pipe(catchError(() => of([]))),
      expenses: this.expenseService.getExpenseHistory().pipe(catchError(() => of([]))),
      payslips: this.payrollService.getPayslips().pipe(catchError(() => of([])))
    }).subscribe(({ leaveBalances, expenses, payslips }) => {
      const cards: KpiCard[] = [];

      leaveBalances.forEach((b, i) => {
        cards.push({
          label: this.formatLeaveType(b.leave_type),
          value: b.remaining_days,
          sublabel: `${b.used_days} used of ${b.total_days} days`,
          colorClass: leaveColors[i % leaveColors.length]
        });
      });

      const pending = expenses.filter(e => e.status === 'submitted').length;
      const approved = expenses.filter(e => e.status === 'approved').length;

      cards.push({
        label: 'Pending Expenses',
        value: pending,
        sublabel: 'awaiting review',
        colorClass: 'kpi-amber'
      });
      cards.push({
        label: 'Approved Expenses',
        value: approved,
        sublabel: 'total approved',
        colorClass: 'kpi-green'
      });

      const sorted = [...payslips].sort((a, b) => {
        const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
        const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
        return tb - ta;
      });
      const last = sorted[0] ?? null;

      cards.push({
        label: 'Last Net Pay',
        value: last ? `£${last.net_pay.toFixed(2)}` : 'N/A',
        sublabel: last?.period_end ? `Period to ${last.period_end}` : 'No payslip on record',
        colorClass: 'kpi-purple'
      });

      this.kpiCards = cards;
      this.kpiLoading = false;
    });
  }

  private formatLeaveType(type: string): string {
    return type
      .split('_')
      .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(' ');
  }

  getRoleMessage(): string {
    if (!this.user) return '';
    switch (this.user.role.toLowerCase()) {
      case 'admin':    return 'Full system access — manage employees, payroll, and configurations.';
      case 'hr':       return 'Manage leave, expenses, hiring pipelines, and payroll cycles.';
      case 'manager':  return 'Oversee your team — approve requests, review candidates, and track expenses.';
      case 'employee': return 'Access your leave balances, expense claims, and payslips.';
      default:         return `Logged in as ${this.user.role}.`;
    }
  }
}
