import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ExpenseService, Expense } from '../../services/expense.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';
import { OcrResultDisplayComponent, SubmittedExpenseData } from '../ocr-result-display/ocr-result-display.component';

@Component({
  selector: 'app-expense-history',
  standalone: true,
  imports: [CommonModule, FormsModule, LlmReasoningDisplayComponent, OcrResultDisplayComponent],
  templateUrl: './expense-history.component.html',
  styleUrl: './expense-history.component.scss'
})
export class ExpenseHistoryComponent implements OnInit {
  expenses: Expense[] = [];
  isLoading = false;
  error: string | null = null;

  // Filters
  selectedStatus = '';
  selectedCategory = '';

  // Modal
  selectedExpense: Expense | null = null;
  showModal = false;

  // Cancel state
  cancellingId: number | null = null;

  readonly statuses = [
    { value: '', label: 'All' },
    { value: 'submitted', label: 'Pending' },
    { value: 'approved', label: 'Approved' },
    { value: 'rejected', label: 'Rejected' },
    { value: 'cancelled', label: 'Cancelled' },
  ];

  readonly categories = [
    { value: '', label: 'All Categories' },
    { value: 'meals', label: 'Meals' },
    { value: 'travel', label: 'Travel' },
    { value: 'equipment', label: 'Equipment' },
    { value: 'entertainment', label: 'Entertainment' },
    { value: 'office_supplies', label: 'Office Supplies' },
    { value: 'other', label: 'Other' },
  ];

  constructor(private expenseService: ExpenseService) {}

  ngOnInit(): void {
    this.loadExpenses();
  }

  loadExpenses(): void {
    this.isLoading = true;
    this.error = null;

    this.expenseService.getExpenseHistory(
      this.selectedStatus || undefined,
      this.selectedCategory || undefined
    ).subscribe({
      next: (data) => {
        this.expenses = data;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Failed to load expense history.';
        this.isLoading = false;
      }
    });
  }

  onFilterChange(): void {
    this.loadExpenses();
  }

  cancelExpense(expense: Expense): void {
    if (this.cancellingId) return;
    this.cancellingId = expense.id;

    this.expenseService.cancelExpense(expense.id).subscribe({
      next: () => {
        expense.status = 'cancelled';
        this.cancellingId = null;
      },
      error: (err) => {
        alert(err.error?.detail || 'Failed to cancel expense.');
        this.cancellingId = null;
      }
    });
  }

  openModal(expense: Expense): void {
    this.selectedExpense = expense;
    this.showModal = true;
  }

  closeModal(): void {
    this.showModal = false;
    this.selectedExpense = null;
  }

  getStatusClass(status: string): string {
    const map: Record<string, string> = {
      approved: 'status-approved',
      submitted: 'status-pending',
      rejected: 'status-rejected',
      cancelled: 'status-cancelled',
    };
    return map[status] || 'status-cancelled';
  }

  getStatusLabel(status: string): string {
    return status === 'submitted' ? 'Pending' : status.charAt(0).toUpperCase() + status.slice(1);
  }

  getStatusIcon(status: string): string {
    const map: Record<string, string> = {
      approved: '✓',
      submitted: '⏳',
      rejected: '✗',
      cancelled: '—',
    };
    return map[status] || '—';
  }

  formatDate(dateStr: string | null): string {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric'
    });
  }

  formatCurrency(amount: number): string {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
  }

  formatCategory(cat: string): string {
    return cat.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  getDecision(decision: string | null): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (decision === 'AUTO_APPROVE') return 'AUTO_APPROVE';
    if (decision === 'ESCALATE') return 'ESCALATE';
    return 'REJECT';
  }

  getReceiptUrl(expense: Expense): string | null {
    if (!expense.receipt_filename) return null;
    return `http://127.0.0.1:8000/uploads/receipts/${expense.receipt_filename}`;
  }

  getSubmittedData(expense: Expense): SubmittedExpenseData {
    return {
      amount: expense.amount,
      vendor: expense.vendor,
      date: expense.date,
    };
  }
}
