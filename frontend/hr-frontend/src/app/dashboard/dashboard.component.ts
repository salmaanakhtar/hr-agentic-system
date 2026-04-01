import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService, User } from '../auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  user: User | null = null;

  constructor(
    private authService: AuthService,
    private router: Router
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
  }

  getRoleMessage(): string {
    if (!this.user) return '';
    switch (this.user.role.toLowerCase()) {
      case 'admin':   return 'Full system access — manage employees, payroll, and configurations.';
      case 'hr':      return 'Manage leave, expenses, hiring pipelines, and payroll cycles.';
      case 'manager': return 'Oversee your team — approve requests, review candidates, and track expenses.';
      case 'employee': return 'Access your leave balances, expense claims, and payslips.';
      default:        return `Logged in as ${this.user.role}.`;
    }
  }
}
