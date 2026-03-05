import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { HiringService, Candidate } from '../../services/hiring.service';

@Component({
  selector: 'app-candidate-upload',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  templateUrl: './candidate-upload.component.html',
  styleUrl: './candidate-upload.component.scss'
})
export class CandidateUploadComponent {
  form: FormGroup;
  isSubmitting = false;
  error: string | null = null;
  result: Candidate | null = null;

  selectedFile: File | null = null;
  fileError: string | null = null;
  isDragging = false;

  constructor(
    private fb: FormBuilder,
    private hiringService: HiringService,
    private router: Router,
  ) {
    this.form = this.fb.group({
      first_name: ['', Validators.required],
      last_name: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]],
      phone: [''],
      linkedin_url: [''],
    });
  }

  isFieldInvalid(field: string): boolean {
    const ctrl = this.form.get(field);
    return !!(ctrl && ctrl.invalid && ctrl.touched);
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(): void {
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    const file = event.dataTransfer?.files[0];
    if (file) this.handleFile(file);
  }

  onFileChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.handleFile(file);
  }

  private handleFile(file: File): void {
    this.fileError = null;
    if (file.type !== 'application/pdf') {
      this.fileError = 'Only PDF files are accepted.';
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      this.fileError = `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum is 10MB.`;
      return;
    }
    this.selectedFile = file;
  }

  removeFile(): void {
    this.selectedFile = null;
    this.fileError = null;
  }

  onSubmit(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    if (!this.selectedFile) {
      this.fileError = 'Please upload a CV (PDF).';
      return;
    }

    this.isSubmitting = true;
    this.error = null;

    const formData = new FormData();
    formData.append('first_name', this.form.value.first_name);
    formData.append('last_name', this.form.value.last_name);
    formData.append('email', this.form.value.email);
    if (this.form.value.phone) formData.append('phone', this.form.value.phone);
    if (this.form.value.linkedin_url) formData.append('linkedin_url', this.form.value.linkedin_url);
    formData.append('cv', this.selectedFile);

    this.hiringService.uploadCandidate(formData).subscribe({
      next: (res) => {
        this.isSubmitting = false;
        this.result = res.candidate;
      },
      error: (err) => {
        this.isSubmitting = false;
        this.error = err.error?.detail || 'Failed to upload candidate.';
      }
    });
  }

  viewCandidate(): void {
    if (this.result) {
      this.router.navigate(['/hiring/candidates', this.result.id]);
    }
  }

  uploadAnother(): void {
    this.result = null;
    this.error = null;
    this.selectedFile = null;
    this.fileError = null;
    this.form.reset();
  }
}
