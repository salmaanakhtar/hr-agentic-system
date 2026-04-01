import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService, User } from '../../auth.service';
import { ThemeToggleComponent } from '../theme-toggle/theme-toggle.component';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule, ThemeToggleComponent],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss'
})
export class SidebarComponent implements OnInit {
  user: User | null = null;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.user = this.authService.getCurrentUser();
    this.authService.currentUser$.subscribe(user => {
      this.user = user;
    });
  }

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }

  isManagerOrAbove(): boolean {
    return ['manager', 'hr', 'admin'].includes(this.user?.role?.toLowerCase() ?? '');
  }

  isHrOrAdmin(): boolean {
    return ['hr', 'admin'].includes(this.user?.role?.toLowerCase() ?? '');
  }

  isEmployeeOrManager(): boolean {
    return ['employee', 'manager'].includes(this.user?.role?.toLowerCase() ?? '');
  }
}
