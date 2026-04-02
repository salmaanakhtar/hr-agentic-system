import { Component, OnInit, OnDestroy, ElementRef, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Chart } from 'chart.js/auto';
import { AuthService, User } from '../auth.service';
import { LeaveService, LeaveReport } from '../services/leave.service';
import { ExpenseService, ExpenseReport } from '../services/expense.service';
import { PayrollService, PayrollReport } from '../services/payroll.service';
import { HiringService, HiringReport } from '../services/hiring.service';

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
export class DashboardComponent implements OnInit, OnDestroy {
  user: User | null = null;
  kpiCards: KpiCard[] = [];
  kpiLoading = true;
  showCharts = false;
  isHrOrAdmin = false;

  @ViewChild('applicationsCanvas') applicationsCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('expenseCanvas') expenseCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('leaveCanvas') leaveCanvas!: ElementRef<HTMLCanvasElement>;

  private charts: Chart[] = [];
  private expenseReport: ExpenseReport | null = null;
  private hiringReport: HiringReport | null = null;
  private leaveReport: LeaveReport | null = null;

  constructor(
    private authService: AuthService,
    private router: Router,
    private leaveService: LeaveService,
    private expenseService: ExpenseService,
    private payrollService: PayrollService,
    private hiringService: HiringService
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

    const role = this.user?.role.toLowerCase() ?? '';
    if (role === 'employee') {
      this.loadEmployeeKpis();
    } else if (['manager', 'hr', 'admin'].includes(role)) {
      this.isHrOrAdmin = ['hr', 'admin'].includes(role);
      this.loadOrgKpis(role);
    } else {
      this.kpiLoading = false;
    }
  }

  ngOnDestroy(): void {
    this.charts.forEach(c => c.destroy());
  }

  private loadOrgKpis(role: string): void {
    const hrAdmin = ['hr', 'admin'].includes(role);

    const expense$ = this.expenseService.getReports().pipe(catchError(() => of(null)));
    const hiring$ = this.hiringService.getReports().pipe(catchError(() => of(null)));
    const leave$ = hrAdmin ? this.leaveService.getLeaveReports().pipe(catchError(() => of(null))) : of(null);
    const payroll$ = hrAdmin ? this.payrollService.getReports().pipe(catchError(() => of(null))) : of(null);

    forkJoin({ expense: expense$, hiring: hiring$, leave: leave$, payroll: payroll$ })
      .subscribe(({ expense, hiring, leave, payroll }) => {
        const cards: KpiCard[] = [];

        if (expense) {
          cards.push({
            label: 'Total Claims',
            value: expense.summary.total_claims,
            sublabel: `${expense.summary.approved} approved`,
            colorClass: 'kpi-amber'
          });
          cards.push({
            label: 'Approval Rate',
            value: `${expense.summary.approval_rate}%`,
            sublabel: `${expense.summary.pending} pending`,
            colorClass: 'kpi-green'
          });
        }

        if (hiring) {
          cards.push({
            label: 'Open Jobs',
            value: hiring.summary.open_jobs,
            sublabel: `${hiring.summary.total_applications} applications`,
            colorClass: 'kpi-indigo'
          });
          cards.push({
            label: 'Shortlist Rate',
            value: `${hiring.llm_decisions.shortlist_rate}%`,
            sublabel: `${hiring.summary.total_candidates} candidates`,
            colorClass: 'kpi-blue'
          });
        }

        if (leave) {
          cards.push({
            label: 'Leave Requests',
            value: leave.total_requests,
            sublabel: leave.average_approval_time_hours != null
              ? `avg ${leave.average_approval_time_hours.toFixed(1)}h approval`
              : 'no approved requests',
            colorClass: 'kpi-purple'
          });
        }

        if (payroll) {
          const netPay = payroll.financials?.total_net_pay_approved;
          cards.push({
            label: 'Total Net Pay',
            value: netPay != null
              ? `\u00a3${netPay.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : 'N/A',
            sublabel: `${payroll.financials?.approved_payslip_count ?? 0} approved payslips`,
            colorClass: 'kpi-green'
          });
        }

        this.kpiCards = cards;
        this.expenseReport = expense;
        this.hiringReport = hiring;
        this.leaveReport = leave;
        this.kpiLoading = false;
        this.showCharts = true;

        setTimeout(() => this.createCharts(), 0);
      });
  }

  private createCharts(): void {
    this.charts.forEach(c => c.destroy());
    this.charts = [];

    if (this.hiringReport && this.applicationsCanvas?.nativeElement) {
      const appsByStatus = this.hiringReport.applications_by_status ?? {};
      const labels = Object.keys(appsByStatus);
      const data = labels.map(k => appsByStatus[k]);
      this.charts.push(new Chart(this.applicationsCanvas.nativeElement, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: ['#6366f1', '#10b981', '#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6'],
            borderWidth: 2,
            borderColor: 'transparent'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#6b7280', font: { size: 11 }, padding: 12 } }
          }
        }
      }));
    }

    if (this.expenseReport && this.expenseCanvas?.nativeElement) {
      const byCategory = this.expenseReport.by_category ?? [];
      this.charts.push(new Chart(this.expenseCanvas.nativeElement, {
        type: 'bar',
        data: {
          labels: byCategory.map(c => c.category),
          datasets: [{
            label: 'Approved (\u00a3)',
            data: byCategory.map(c => c.total_approved),
            backgroundColor: 'rgba(245,158,11,0.75)',
            borderColor: '#f59e0b',
            borderWidth: 1,
            borderRadius: 5
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(107,114,128,0.1)' } },
            y: { ticks: { color: '#6b7280' }, grid: { display: false } }
          }
        }
      }));
    }

    if (this.leaveReport && this.leaveCanvas?.nativeElement) {
      const totals = this.leaveReport.leave_type_totals ?? {};
      const labels = Object.keys(totals).map(k => this.formatLeaveType(k));
      const data = Object.values(totals) as number[];
      this.charts.push(new Chart(this.leaveCanvas.nativeElement, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Days',
            data,
            backgroundColor: ['rgba(99,102,241,0.75)', 'rgba(16,185,129,0.75)', 'rgba(59,130,246,0.75)'],
            borderRadius: 5
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(107,114,128,0.1)' } },
            x: { ticks: { color: '#6b7280' }, grid: { display: false } }
          }
        }
      }));
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
        value: last ? `\u00a3${last.net_pay.toFixed(2)}` : 'N/A',
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
