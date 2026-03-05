import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { HiringService, JobPosting } from '../../services/hiring.service';

@Component({
  selector: 'app-job-list',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './job-list.component.html',
  styleUrl: './job-list.component.scss'
})
export class JobListComponent implements OnInit {
  jobs: JobPosting[] = [];
  isLoading = true;
  error: string | null = null;
  statusFilter = '';
  departmentFilter = '';

  statuses = [
    { value: '', label: 'All Statuses' },
    { value: 'open', label: 'Open' },
    { value: 'draft', label: 'Draft' },
    { value: 'closed', label: 'Closed' },
    { value: 'on_hold', label: 'On Hold' },
  ];

  constructor(private hiringService: HiringService) {}

  ngOnInit(): void {
    this.loadJobs();
  }

  loadJobs(): void {
    this.isLoading = true;
    this.error = null;
    this.hiringService.getJobs(
      this.statusFilter || undefined,
      this.departmentFilter || undefined,
    ).subscribe({
      next: (jobs) => { this.jobs = jobs; this.isLoading = false; },
      error: () => { this.error = 'Failed to load job postings.'; this.isLoading = false; }
    });
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'open': return 'status-open';
      case 'draft': return 'status-draft';
      case 'closed': return 'status-closed';
      case 'on_hold': return 'status-hold';
      default: return '';
    }
  }

  formatEmploymentType(t: string): string {
    return t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  formatSalary(min: number | null, max: number | null): string {
    if (!min && !max) return '—';
    const fmt = (n: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
    if (min && max) return `${fmt(min)} – ${fmt(max)}`;
    if (min) return `From ${fmt(min)}`;
    return `Up to ${fmt(max!)}`;
  }
}
