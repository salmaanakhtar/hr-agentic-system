import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { PolicyService } from '../../services/policy.service';
import { AuthService } from '../../auth.service';
import { Router } from '@angular/router';

const CATEGORIES = ['general', 'leave', 'expense', 'hiring', 'payroll'];

@Component({
  selector: 'app-policy-upload',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './policy-upload.component.html',
  styleUrl: './policy-upload.component.scss',
})
export class PolicyUploadComponent {
  title = '';
  category = '';
  description = '';
  selectedFile: File | null = null;
  categories = CATEGORIES;

  isUploading = false;
  error: string | null = null;
  successResult: { title: string; chunk_count: number; document_id: number } | null = null;

  constructor(
    private policyService: PolicyService,
    private authService: AuthService,
    private router: Router,
  ) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.error = null;
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type === 'application/pdf') {
      this.selectedFile = file;
      this.error = null;
    } else if (file) {
      this.error = 'Only PDF files are accepted.';
    }
  }

  isValid(): boolean {
    return !!this.title.trim() && !!this.category && !!this.selectedFile;
  }

  submit(): void {
    if (!this.isValid()) return;

    const formData = new FormData();
    formData.append('title', this.title.trim());
    formData.append('category', this.category);
    if (this.description.trim()) formData.append('description', this.description.trim());
    formData.append('file', this.selectedFile!);

    this.isUploading = true;
    this.error = null;
    this.successResult = null;

    this.policyService.uploadDocument(formData).subscribe({
      next: (res) => {
        this.isUploading = false;
        this.successResult = { title: res.title, chunk_count: res.chunk_count, document_id: res.document_id };
      },
      error: (err) => {
        this.error = err.error?.detail || 'Upload failed. Please try again.';
        this.isUploading = false;
      },
    });
  }

  uploadAnother(): void {
    this.title = '';
    this.category = '';
    this.description = '';
    this.selectedFile = null;
    this.successResult = null;
    this.error = null;
  }

  categoryLabel(cat: string): string {
    return cat.charAt(0).toUpperCase() + cat.slice(1);
  }
}
