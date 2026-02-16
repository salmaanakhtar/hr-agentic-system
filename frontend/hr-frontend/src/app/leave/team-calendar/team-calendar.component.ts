import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LeaveService, TeamCalendarEntry, PriorityPeriod } from '../../services/leave.service';

export interface CalendarDay {
  date: number;
  isCurrentMonth: boolean;
  isPast: boolean;
  isToday: boolean;
  leaves: TeamCalendarEntry[];
  priorityPeriods: PriorityPeriod[];
}

@Component({
  selector: 'app-team-calendar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './team-calendar.component.html',
  styleUrl: './team-calendar.component.scss'
})
export class TeamCalendarComponent implements OnInit {
  leaveRequests: TeamCalendarEntry[] = [];
  priorityPeriods: PriorityPeriod[] = [];
  calendarDays: CalendarDay[] = [];
  isLoading = false;
  error = '';

  selectedMonth: number;
  selectedYear: number;

  months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  constructor(private leaveService: LeaveService) {
    const now = new Date();
    this.selectedMonth = now.getMonth() + 1; // 1-12
    this.selectedYear = now.getFullYear();
  }

  ngOnInit(): void {
    this.loadCalendar();
  }

  loadCalendar(): void {
    this.isLoading = true;
    this.error = '';

    // Format month as YYYY-MM for backend
    const monthStr = this.selectedMonth.toString().padStart(2, '0');
    const monthParam = `${this.selectedYear}-${monthStr}`;

    this.leaveService.getTeamCalendar(undefined, monthParam).subscribe({
      next: (data) => {
        this.leaveRequests = data.leave_requests;
        this.priorityPeriods = data.priority_periods;
        this.generateCalendarDays();
        this.isLoading = false;
      },
      error: (error) => {
        this.error = 'Failed to load team calendar';
        this.isLoading = false;
        console.error('Error loading calendar:', error);
      }
    });
  }

  generateCalendarDays(): void {
    const year = this.selectedYear;
    const month = this.selectedMonth - 1; // 0-indexed for Date

    // Get first day of month
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);

    // Get day of week for first day (0 = Sunday)
    const firstDayOfWeek = firstDay.getDay();

    // Calculate days to show from previous month
    const daysFromPrevMonth = firstDayOfWeek;

    // Calculate days to show from next month
    const daysInMonth = lastDay.getDate();
    const totalCells = Math.ceil((daysFromPrevMonth + daysInMonth) / 7) * 7;
    const daysFromNextMonth = totalCells - (daysFromPrevMonth + daysInMonth);

    const days: CalendarDay[] = [];
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Previous month days
    const prevMonthLastDay = new Date(year, month, 0).getDate();
    for (let i = daysFromPrevMonth - 1; i >= 0; i--) {
      const date = prevMonthLastDay - i;
      days.push({
        date,
        isCurrentMonth: false,
        isPast: true,
        isToday: false,
        leaves: [],
        priorityPeriods: []
      });
    }

    // Current month days
    for (let date = 1; date <= daysInMonth; date++) {
      const currentDate = new Date(year, month, date);
      currentDate.setHours(0, 0, 0, 0);

      const isPast = currentDate < today;
      const isToday = currentDate.getTime() === today.getTime();

      // Find leaves for this day
      const leavesForDay = this.leaveRequests.filter(leave => {
        const start = new Date(leave.start_date);
        const end = new Date(leave.end_date);
        start.setHours(0, 0, 0, 0);
        end.setHours(0, 0, 0, 0);
        return currentDate >= start && currentDate <= end;
      });

      // Find priority periods for this day
      const periodsForDay = this.priorityPeriods.filter(period => {
        const start = new Date(period.start_date);
        const end = new Date(period.end_date);
        start.setHours(0, 0, 0, 0);
        end.setHours(0, 0, 0, 0);
        return currentDate >= start && currentDate <= end;
      });

      days.push({
        date,
        isCurrentMonth: true,
        isPast,
        isToday,
        leaves: leavesForDay,
        priorityPeriods: periodsForDay
      });
    }

    // Next month days
    for (let date = 1; date <= daysFromNextMonth; date++) {
      days.push({
        date,
        isCurrentMonth: false,
        isPast: false,
        isToday: false,
        leaves: [],
        priorityPeriods: []
      });
    }

    this.calendarDays = days;
  }

  onMonthChange(): void {
    this.loadCalendar();
  }

  onYearChange(): void {
    this.loadCalendar();
  }

  previousMonth(): void {
    this.selectedMonth--;
    if (this.selectedMonth < 1) {
      this.selectedMonth = 12;
      this.selectedYear--;
    }
    this.loadCalendar();
  }

  nextMonth(): void {
    this.selectedMonth++;
    if (this.selectedMonth > 12) {
      this.selectedMonth = 1;
      this.selectedYear++;
    }
    this.loadCalendar();
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  }

  getLeaveTypeColor(leaveType: string): string {
    const colors: { [key: string]: string } = {
      'vacation': '#6366f1',
      'sick_leave': '#f59e0b',
      'personal': '#8b5cf6',
      'maternity': '#ec4899',
      'paternity': '#3b82f6',
      'bereavement': '#6b7280'
    };
    return colors[leaveType] || '#6366f1';
  }

  getStatusBadgeClass(status: string): string {
    switch (status) {
      case 'approved':
        return 'status-approved';
      case 'pending':
        return 'status-pending';
      default:
        return '';
    }
  }

  getAvailableYears(): number[] {
    const currentYear = new Date().getFullYear();
    return [currentYear - 1, currentYear, currentYear + 1];
  }

  isPriorityPeriod(period: PriorityPeriod): boolean {
    return !period.is_blackout;
  }

  isBlackoutPeriod(period: PriorityPeriod): boolean {
    return period.is_blackout;
  }
}
