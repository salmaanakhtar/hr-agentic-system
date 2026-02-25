import { Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LeaveRequestFormComponent } from './leave/leave-request-form/leave-request-form.component';
import { LeaveHistoryComponent } from './leave/leave-history/leave-history.component';
import { LeaveBalanceCardComponent } from './leave/leave-balance-card/leave-balance-card.component';
import { PendingApprovalsComponent } from './leave/pending-approvals/pending-approvals.component';
import { TeamCalendarComponent } from './leave/team-calendar/team-calendar.component';
import { ExpenseRequestFormComponent } from './expenses/expense-request-form/expense-request-form.component';
import { ExpenseHistoryComponent } from './expenses/expense-history/expense-history.component';
import { PendingExpenseApprovalsComponent } from './expenses/pending-expense-approvals/pending-expense-approvals.component';

export const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'signup', component: SignupComponent },
  { path: 'dashboard', component: DashboardComponent },

  // Leave Management Routes
  { path: 'leave/request', component: LeaveRequestFormComponent },
  { path: 'leave/history', component: LeaveHistoryComponent },
  { path: 'leave/balance', component: LeaveBalanceCardComponent },
  { path: 'leave/approvals', component: PendingApprovalsComponent },
  { path: 'leave/calendar', component: TeamCalendarComponent },

  // Expense Management Routes
  { path: 'expenses/submit', component: ExpenseRequestFormComponent },
  { path: 'expenses/history', component: ExpenseHistoryComponent },
  { path: 'expenses/approvals', component: PendingExpenseApprovalsComponent },

  { path: '**', redirectTo: '/login' }
];
