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
    if (!this.user) {
      this.authService.currentUser$.subscribe(user => {
        this.user = user;
      });
    }
  }

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }

  getRoleMessage(): string {
    if (!this.user) return '';

    switch (this.user.role.toLowerCase()) {
      case 'admin':
        return 'Welcome Admin! You have full access to the HR system.';
      case 'hr':
        return 'Welcome HR Manager! Manage employees and processes.';
      case 'manager':
        return 'Welcome Manager! Oversee your team and projects.';
      case 'employee':
        return 'Welcome Employee! Access your HR information.';
      default:
        return `Hello ${this.user.role}!`;
    }
  }
}
