import { Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';
import { AppShellComponent } from './shared/app-shell/app-shell.component';
import { authGuard } from './auth.guard';
import { managerGuard, hrAdminGuard } from './role.guard';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LeaveRequestFormComponent } from './leave/leave-request-form/leave-request-form.component';
import { LeaveHistoryComponent } from './leave/leave-history/leave-history.component';
import { LeaveBalanceCardComponent } from './leave/leave-balance-card/leave-balance-card.component';
import { PendingApprovalsComponent } from './leave/pending-approvals/pending-approvals.component';
import { TeamCalendarComponent } from './leave/team-calendar/team-calendar.component';
import { ExpenseRequestFormComponent } from './expenses/expense-request-form/expense-request-form.component';
import { ExpenseHistoryComponent } from './expenses/expense-history/expense-history.component';
import { PendingExpenseApprovalsComponent } from './expenses/pending-expense-approvals/pending-expense-approvals.component';
import { JobListComponent } from './hiring/job-list/job-list.component';
import { JobPostingFormComponent } from './hiring/job-posting-form/job-posting-form.component';
import { CandidateUploadComponent } from './hiring/candidate-upload/candidate-upload.component';
import { CandidateProfileComponent } from './hiring/candidate-profile/candidate-profile.component';
import { HiringPipelineComponent } from './hiring/hiring-pipeline/hiring-pipeline.component';
import { InterviewScheduleComponent } from './hiring/interview-schedule/interview-schedule.component';
import { PayCycleRunComponent } from './payroll/pay-cycle-run/pay-cycle-run.component';
import { PayrollHistoryComponent } from './payroll/payroll-history/payroll-history.component';
import { PayslipComponent } from './payroll/payslip/payslip.component';
import { PolicyListComponent } from './policies/policy-list/policy-list.component';
import { PolicyUploadComponent } from './policies/policy-upload/policy-upload.component';
import { PolicyQueryComponent } from './policies/policy-query/policy-query.component';
import { ComplianceCheckComponent } from './policies/compliance-check/compliance-check.component';

export const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'signup', component: SignupComponent },
  {
    path: '',
    component: AppShellComponent,
    canActivate: [authGuard],
    children: [
      { path: 'dashboard', component: DashboardComponent },

      // Leave Management
      { path: 'leave/request', component: LeaveRequestFormComponent },
      { path: 'leave/history', component: LeaveHistoryComponent },
      { path: 'leave/balance', component: LeaveBalanceCardComponent },
      { path: 'leave/approvals', component: PendingApprovalsComponent, canActivate: [managerGuard] },
      { path: 'leave/calendar', component: TeamCalendarComponent, canActivate: [managerGuard] },

      // Expense Management
      { path: 'expenses/submit', component: ExpenseRequestFormComponent },
      { path: 'expenses/history', component: ExpenseHistoryComponent },
      { path: 'expenses/approvals', component: PendingExpenseApprovalsComponent, canActivate: [managerGuard] },

      // Hiring Pipeline (manager+ only)
      { path: 'hiring/jobs', component: JobListComponent, canActivate: [managerGuard] },
      { path: 'hiring/jobs/new', component: JobPostingFormComponent, canActivate: [managerGuard] },
      { path: 'hiring/jobs/:id/edit', component: JobPostingFormComponent, canActivate: [managerGuard] },
      { path: 'hiring/pipeline/:jobId', component: HiringPipelineComponent, canActivate: [managerGuard] },
      { path: 'hiring/candidates/upload', component: CandidateUploadComponent, canActivate: [managerGuard] },
      { path: 'hiring/candidates/:id', component: CandidateProfileComponent, canActivate: [managerGuard] },
      { path: 'hiring/interview/:applicationId', component: InterviewScheduleComponent, canActivate: [managerGuard] },

      // Payroll
      { path: 'payroll/run', component: PayCycleRunComponent, canActivate: [hrAdminGuard] },
      { path: 'payroll/history', component: PayrollHistoryComponent },
      { path: 'payroll/payslips/:id', component: PayslipComponent },

      // Policy
      { path: 'policies/documents', component: PolicyListComponent },
      { path: 'policies/upload', component: PolicyUploadComponent, canActivate: [hrAdminGuard] },
      { path: 'policies/query', component: PolicyQueryComponent },
      { path: 'policies/compliance', component: ComplianceCheckComponent },
    ]
  },
  { path: '**', redirectTo: '/login' }
];
