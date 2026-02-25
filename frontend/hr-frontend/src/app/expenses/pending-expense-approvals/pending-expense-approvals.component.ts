import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ExpenseService, Expense } from '../../services/expense.service';
import { LlmReasoningDisplayComponent } from '../../leave/llm-reasoning-display/llm-reasoning-display.component';
import { OcrResultDisplayComponent, SubmittedExpenseData } from '../ocr-result-display/ocr-result-display.component';

@Component({
  selector: 'app-pending-expense-approvals',
  standalone: true,
  imports: [CommonModule, FormsModule, LlmReasoningDisplayComponent, OcrResultDisplayComponent],
  templateUrl: './pending-expense-approvals.component.html',
  styleUrl: './pending-expense-approvals.component.scss'
})
export class PendingExpenseApprovalsComponent implements OnInit {
  expenses: Expense[] = [];
  isLoading = false;
  error: string | null = null;

  // Per-card comment inputs
  approvalComments: Record<number, string> = {};

  // Reject modal state
  showRejectModal = false;
  selectedExpense: Expense | null = null;
  rejectionReason = '';
  isActioning = false;
  actioningId: number | null = null;

  constructor(private expenseService: ExpenseService) {}

  ngOnInit(): void {
    this.loadPending();
  }

  loadPending(): void {
    this.isLoading = true;
    this.error = null;

    this.expenseService.getPendingApprovals().subscribe({
      next: (data) => {
        this.expenses = data;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Failed to load pending approvals.';
        this.isLoading = false;
      }
    });
  }

  approveExpense(expense: Expense): void {
    this.actioningId = expense.id;
    const comments = this.approvalComments[expense.id];

    this.expenseService.approveExpense(expense.id, comments).subscribe({
      next: () => {
        this.expenses = this.expenses.filter(e => e.id !== expense.id);
        this.actioningId = null;
      },
      error: (err) => {
        alert(err.error?.detail || 'Failed to approve expense.');
        this.actioningId = null;
      }
    });
  }

  openRejectModal(expense: Expense): void {
    this.selectedExpense = expense;
    this.rejectionReason = '';
    this.showRejectModal = true;
  }

  closeRejectModal(): void {
    this.showRejectModal = false;
    this.selectedExpense = null;
    this.rejectionReason = '';
  }

  confirmReject(): void {
    if (!this.selectedExpense || !this.rejectionReason.trim()) return;

    this.actioningId = this.selectedExpense.id;
    const id = this.selectedExpense.id;

    this.expenseService.rejectExpense(id, this.rejectionReason.trim()).subscribe({
      next: () => {
        this.expenses = this.expenses.filter(e => e.id !== id);
        this.actioningId = null;
        this.closeRejectModal();
      },
      error: (err) => {
        alert(err.error?.detail || 'Failed to reject expense.');
        this.actioningId = null;
      }
    });
  }

  getDecision(decision: string | null): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (decision === 'AUTO_APPROVE') return 'AUTO_APPROVE';
    if (decision === 'ESCALATE') return 'ESCALATE';
    return 'REJECT';
  }

  getLlmBadgeClass(decision: string | null): string {
    if (decision === 'AUTO_APPROVE') return 'badge-auto';
    if (decision === 'ESCALATE') return 'badge-escalate';
    if (decision === 'REJECT') return 'badge-reject';
    return 'badge-neutral';
  }

  getLlmBadgeLabel(decision: string | null): string {
    if (decision === 'AUTO_APPROVE') return 'AI: Auto-Approve';
    if (decision === 'ESCALATE') return 'AI: Escalated';
    if (decision === 'REJECT') return 'AI: Reject';
    return 'AI: Unknown';
  }

  formatCurrency(amount: number): string {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
  }

  formatDate(dateStr: string | null): string {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric'
    });
  }

  formatCategory(cat: string): string {
    return cat.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
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

  get isLowConfidence(): ((expense: Expense) => boolean) {
    return (expense: Expense) =>
      expense.ocr_extracted != null &&
      expense.ocr_extracted.overall_confidence < 0.7;
  }
}
