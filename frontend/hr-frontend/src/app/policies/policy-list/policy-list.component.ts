import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { PolicyService, PolicyDocument } from '../../services/policy.service';
import { AuthService, User } from '../../auth.service';

const CATEGORIES = ['general', 'leave', 'expense', 'hiring', 'payroll'];

@Component({
  selector: 'app-policy-list',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './policy-list.component.html',
  styleUrl: './policy-list.component.scss',
})
export class PolicyListComponent implements OnInit {
  documents: PolicyDocument[] = [];
  isLoading = false;
  error: string | null = null;
  user: User | null = null;

  selectedCategory = '';
  categories = CATEGORIES;

  deletingId: number | null = null;
  deleteError: string | null = null;

  constructor(
    private policyService: PolicyService,
    private authService: AuthService,
  ) {}

  ngOnInit(): void {
    this.user = this.authService.getCurrentUser();
    this.load();
  }

  load(): void {
    this.isLoading = true;
    this.error = null;
    const cat = this.selectedCategory || undefined;
    this.policyService.getDocuments(cat).subscribe({
      next: (docs) => {
        this.documents = docs;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Failed to load policy documents.';
        this.isLoading = false;
      },
    });
  }

  onCategoryChange(): void {
    this.load();
  }

  canManage(): boolean {
    return this.user?.role === 'hr' || this.user?.role === 'admin';
  }

  delete(doc: PolicyDocument): void {
    if (!confirm(`Deactivate "${doc.title}"? It will no longer appear in policy searches.`)) return;
    this.deletingId = doc.id;
    this.deleteError = null;
    this.policyService.deleteDocument(doc.id).subscribe({
      next: () => {
        this.deletingId = null;
        this.documents = this.documents.filter(d => d.id !== doc.id);
      },
      error: (err) => {
        this.deleteError = err.error?.detail || 'Failed to deactivate document.';
        this.deletingId = null;
      },
    });
  }

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  categoryLabel(cat: string): string {
    return cat.charAt(0).toUpperCase() + cat.slice(1);
  }
}
