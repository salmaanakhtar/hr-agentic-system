import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, ActivatedRoute } from '@angular/router';
import { HiringService, JobPosting, Application } from '../../services/hiring.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-hiring-pipeline',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule, LlmReasoningDisplayComponent],
  templateUrl: './hiring-pipeline.component.html',
  styleUrl: './hiring-pipeline.component.scss'
})
export class HiringPipelineComponent implements OnInit {
  job: JobPosting | null = null;
  applications: Application[] = [];
  isLoading = true;
  error: string | null = null;

  // Status update modal
  showStatusModal = false;
  selectedApp: Application | null = null;
  newStatus = '';
  statusNotes = '';
  isUpdating = false;

  // Expanded reasoning per application id
  expandedReasoning: Set<number> = new Set();

  statusOptions = [
    { value: 'applied', label: 'Applied' },
    { value: 'shortlisted', label: 'Shortlisted' },
    { value: 'interviewing', label: 'Interviewing' },
    { value: 'offered', label: 'Offered' },
    { value: 'rejected', label: 'Rejected' },
    { value: 'passed', label: 'Passed' },
  ];

  constructor(
    private hiringService: HiringService,
    private route: ActivatedRoute,
  ) {}

  ngOnInit(): void {
    const jobId = +this.route.snapshot.params['jobId'];
    this.loadJob(jobId);
    this.loadApplications(jobId);
  }

  loadJob(jobId: number): void {
    this.hiringService.getJobById(jobId).subscribe({
      next: (job) => { this.job = job; },
      error: () => { this.error = 'Failed to load job details.'; }
    });
  }

  loadApplications(jobId: number): void {
    this.isLoading = true;
    this.hiringService.getApplications(jobId).subscribe({
      next: (apps) => { this.applications = apps; this.isLoading = false; },
      error: () => { this.error = 'Failed to load applications.'; this.isLoading = false; }
    });
  }

  toggleReasoning(appId: number): void {
    if (this.expandedReasoning.has(appId)) {
      this.expandedReasoning.delete(appId);
    } else {
      this.expandedReasoning.add(appId);
    }
  }

  isReasoningExpanded(appId: number): boolean {
    return this.expandedReasoning.has(appId);
  }

  openStatusModal(app: Application): void {
    this.selectedApp = app;
    this.newStatus = app.status;
    this.statusNotes = '';
    this.showStatusModal = true;
  }

  closeStatusModal(): void {
    this.showStatusModal = false;
    this.selectedApp = null;
    this.newStatus = '';
    this.statusNotes = '';
  }

  confirmStatusUpdate(): void {
    if (!this.selectedApp || !this.newStatus) return;
    this.isUpdating = true;
    const id = this.selectedApp.id;
    this.hiringService.updateApplicationStatus(id, this.newStatus, this.statusNotes || undefined).subscribe({
      next: () => {
        const app = this.applications.find(a => a.id === id);
        if (app) app.status = this.newStatus as Application['status'];
        this.isUpdating = false;
        this.closeStatusModal();
      },
      error: (err) => {
        alert(err.error?.detail || 'Failed to update status.');
        this.isUpdating = false;
      }
    });
  }

  // Map hiring decision to LlmReasoningDisplay decision type
  getDisplayDecision(d: string | null): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (d === 'SHORTLIST') return 'AUTO_APPROVE';
    if (d === 'REVIEW') return 'ESCALATE';
    return 'REJECT';
  }

  getDecisionClass(d: string | null): string {
    if (d === 'SHORTLIST') return 'decision-shortlist';
    if (d === 'REVIEW') return 'decision-review';
    return 'decision-pass';
  }

  getDecisionLabel(d: string | null): string {
    if (d === 'SHORTLIST') return 'Shortlist';
    if (d === 'REVIEW') return 'Review';
    if (d === 'PASS') return 'Pass';
    return 'Pending';
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'shortlisted': return 'status-shortlisted';
      case 'interviewing': return 'status-interviewing';
      case 'offered': return 'status-offered';
      case 'rejected': return 'status-rejected';
      case 'passed': return 'status-passed';
      default: return 'status-applied';
    }
  }

  formatScore(score: number | null): string {
    if (score == null) return '—';
    return `${(score * 100).toFixed(0)}%`;
  }

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  formatEmploymentType(t: string): string {
    return t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}
