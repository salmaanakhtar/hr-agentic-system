import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { FormsModule } from '@angular/forms';
import { Router, ActivatedRoute, RouterModule } from '@angular/router';
import { HiringService } from '../../services/hiring.service';

@Component({
  selector: 'app-job-posting-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, FormsModule, RouterModule],
  templateUrl: './job-posting-form.component.html',
  styleUrl: './job-posting-form.component.scss'
})
export class JobPostingFormComponent implements OnInit {
  form!: FormGroup;
  isSubmitting = false;
  error: string | null = null;
  isEditMode = false;
  jobId: number | null = null;
  newSkill = '';
  skills: string[] = [];

  employmentTypes = [
    { value: 'full_time', label: 'Full Time' },
    { value: 'part_time', label: 'Part Time' },
    { value: 'contract', label: 'Contract' },
  ];

  constructor(
    private fb: FormBuilder,
    private hiringService: HiringService,
    private router: Router,
    private route: ActivatedRoute,
  ) {}

  ngOnInit(): void {
    this.form = this.fb.group({
      title: ['', Validators.required],
      department: ['', Validators.required],
      description: ['', Validators.required],
      requirements: ['', Validators.required],
      experience_years: [null],
      employment_type: ['full_time', Validators.required],
      location: [''],
      salary_min: [null],
      salary_max: [null],
    });

    const idParam = this.route.snapshot.params['id'];
    if (idParam) {
      this.jobId = +idParam;
      this.isEditMode = true;
      this.loadJob(this.jobId);
    }
  }

  loadJob(id: number): void {
    this.hiringService.getJobById(id).subscribe({
      next: (job) => {
        this.skills = [...(job.required_skills || [])];
        this.form.patchValue({
          title: job.title,
          department: job.department,
          description: job.description,
          requirements: job.requirements,
          experience_years: job.experience_years,
          employment_type: job.employment_type,
          location: job.location ?? '',
          salary_min: job.salary_min,
          salary_max: job.salary_max,
        });
      },
      error: () => { this.error = 'Failed to load job posting.'; }
    });
  }

  addSkill(): void {
    const s = this.newSkill.trim();
    if (s && !this.skills.includes(s)) {
      this.skills.push(s);
    }
    this.newSkill = '';
  }

  removeSkill(skill: string): void {
    this.skills = this.skills.filter(s => s !== skill);
  }

  onSkillKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.addSkill();
    }
  }

  isFieldInvalid(field: string): boolean {
    const ctrl = this.form.get(field);
    return !!(ctrl && ctrl.invalid && ctrl.touched);
  }

  onSubmit(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    this.isSubmitting = true;
    this.error = null;
    const payload = { ...this.form.value, required_skills: this.skills };

    const obs = this.isEditMode && this.jobId
      ? this.hiringService.updateJob(this.jobId, payload)
      : this.hiringService.createJob(payload);

    obs.subscribe({
      next: () => { this.isSubmitting = false; this.router.navigate(['/hiring/jobs']); },
      error: (err) => {
        this.isSubmitting = false;
        this.error = err.error?.detail || 'Failed to save job posting.';
      }
    });
  }
}
