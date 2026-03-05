import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute } from '@angular/router';
import { HiringService, Candidate, Application } from '../../services/hiring.service';

@Component({
  selector: 'app-candidate-profile',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './candidate-profile.component.html',
  styleUrl: './candidate-profile.component.scss'
})
export class CandidateProfileComponent implements OnInit {
  candidate: Candidate | null = null;
  applications: Application[] = [];
  isLoading = true;
  error: string | null = null;

  constructor(
    private hiringService: HiringService,
    private route: ActivatedRoute,
  ) {}

  ngOnInit(): void {
    const id = +this.route.snapshot.params['id'];
    this.loadCandidate(id);
    this.loadApplications(id);
  }

  loadCandidate(id: number): void {
    this.hiringService.getCandidateById(id).subscribe({
      next: (c) => { this.candidate = c; this.isLoading = false; },
      error: () => { this.error = 'Failed to load candidate profile.'; this.isLoading = false; }
    });
  }

  loadApplications(candidateId: number): void {
    this.hiringService.getApplications(undefined, candidateId).subscribe({
      next: (apps) => { this.applications = apps; },
      error: () => {}
    });
  }

  getDecisionClass(decision: string | null): string {
    if (decision === 'SHORTLIST') return 'decision-shortlist';
    if (decision === 'REVIEW') return 'decision-review';
    return 'decision-pass';
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

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  formatScore(score: number | null): string {
    if (score == null) return '—';
    return `${(score * 100).toFixed(0)}%`;
  }
}
