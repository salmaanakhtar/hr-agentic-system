import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router, ActivatedRoute, RouterModule } from '@angular/router';
import { HiringService, Application } from '../../services/hiring.service';

@Component({
  selector: 'app-interview-schedule',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  templateUrl: './interview-schedule.component.html',
  styleUrl: './interview-schedule.component.scss'
})
export class InterviewScheduleComponent implements OnInit {
  form: FormGroup;
  application: Application | null = null;
  applicationId: number | null = null;
  isLoading = true;
  isSubmitting = false;
  error: string | null = null;
  success = false;
  scheduledDate: string | null = null;

  constructor(
    private fb: FormBuilder,
    private hiringService: HiringService,
    private router: Router,
    private route: ActivatedRoute,
  ) {
    this.form = this.fb.group({
      interview_date: ['', Validators.required],
      interview_notes: [''],
    });
  }

  ngOnInit(): void {
    this.applicationId = +this.route.snapshot.params['applicationId'];
    this.loadApplication(this.applicationId);
  }

  loadApplication(id: number): void {
    // Load all applications to find the one with this id
    this.hiringService.getApplications().subscribe({
      next: (apps) => {
        this.application = apps.find(a => a.id === id) ?? null;
        if (!this.application) {
          this.error = 'Application not found.';
        }
        this.isLoading = false;
      },
      error: () => {
        this.error = 'Failed to load application details.';
        this.isLoading = false;
      }
    });
  }

  isFieldInvalid(field: string): boolean {
    const ctrl = this.form.get(field);
    return !!(ctrl && ctrl.invalid && ctrl.touched);
  }

  onSubmit(): void {
    if (this.form.invalid || !this.applicationId) {
      this.form.markAllAsTouched();
      return;
    }

    this.isSubmitting = true;
    this.error = null;

    const rawDate = this.form.value.interview_date;
    // Convert datetime-local value to ISO string
    const isoDate = new Date(rawDate).toISOString();

    this.hiringService.scheduleInterview(
      this.applicationId,
      isoDate,
      this.form.value.interview_notes || undefined
    ).subscribe({
      next: (res) => {
        this.isSubmitting = false;
        this.success = true;
        this.scheduledDate = res.interview_date;
      },
      error: (err) => {
        this.isSubmitting = false;
        this.error = err.error?.detail || 'Failed to schedule interview.';
      }
    });
  }

  goToPipeline(): void {
    if (this.application) {
      this.router.navigate(['/hiring/pipeline', this.application.job_id]);
    } else {
      this.router.navigate(['/hiring/jobs']);
    }
  }

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  }
}
