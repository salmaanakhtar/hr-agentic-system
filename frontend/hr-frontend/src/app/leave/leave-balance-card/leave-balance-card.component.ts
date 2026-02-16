import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LeaveService, LeaveBalance } from '../../services/leave.service';

@Component({
  selector: 'app-leave-balance-card',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './leave-balance-card.component.html',
  styleUrl: './leave-balance-card.component.scss'
})
export class LeaveBalanceCardComponent implements OnInit {
  balances: LeaveBalance[] = [];
  isLoading = false;
  error = '';
  selectedYear: number;

  leaveTypeLabels: { [key: string]: string } = {
    'vacation': 'Vacation',
    'sick_leave': 'Sick Leave',
    'personal': 'Personal Leave',
    'maternity': 'Maternity Leave',
    'paternity': 'Paternity Leave',
    'bereavement': 'Bereavement Leave'
  };

  constructor(private leaveService: LeaveService) {
    this.selectedYear = new Date().getFullYear();
  }

  ngOnInit(): void {
    this.loadBalances();
  }

  loadBalances(): void {
    this.isLoading = true;
    this.error = '';

    this.leaveService.getLeaveBalances(undefined, this.selectedYear).subscribe({
      next: (balances) => {
        this.balances = balances;
        this.isLoading = false;
      },
      error: (error) => {
        this.error = 'Failed to load leave balances';
        this.isLoading = false;
        console.error('Error loading balances:', error);
      }
    });
  }

  onYearChange(): void {
    this.loadBalances();
  }

  getUsagePercentage(balance: LeaveBalance): number {
    if (balance.total_days === 0) return 0;
    return (balance.used_days / balance.total_days) * 100;
  }

  getProgressBarClass(percentage: number): string {
    if (percentage >= 90) return 'progress-critical';
    if (percentage >= 70) return 'progress-warning';
    return 'progress-good';
  }

  getLeaveTypeLabel(leaveType: string): string {
    return this.leaveTypeLabels[leaveType] || leaveType;
  }

  getAvailableYears(): number[] {
    const currentYear = new Date().getFullYear();
    return [currentYear - 1, currentYear, currentYear + 1];
  }
}
