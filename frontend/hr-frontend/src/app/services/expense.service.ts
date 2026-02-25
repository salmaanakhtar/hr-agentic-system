import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

// --- Interfaces matching backend schemas ---

export interface OCRExtractionResult {
  raw_text: string;
  vendor: string | null;
  amount: number | null;
  date: string | null;
  vendor_confidence: number;
  amount_confidence: number;
  date_confidence: number;
  overall_confidence: number;
}

export interface ExpenseValidationOutput {
  success: boolean;
  message: string;
  reasoning: string;
  decision: 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT';
  confidence: number;
  expense_id: number | null;
  validated_amount: number | null;
  ocr_result: OCRExtractionResult | null;
  ocr_match: boolean;
  policy_violations: string[];
  factors: string[];
  recommendations: string[];
}

export interface Expense {
  id: number;
  user_id: number;
  employee_name: string | null;
  amount: number;
  category: string;
  vendor: string | null;
  date: string;
  description: string | null;
  receipt_filename: string | null;
  receipt_url: string | null;
  ocr_confidence: number | null;
  ocr_extracted: OCRExtractionResult | null;
  status: 'draft' | 'submitted' | 'approved' | 'rejected' | 'cancelled';
  llm_decision: string | null;
  llm_reasoning: string | null;
  submitted_at: string | null;
  reviewed_at: string | null;
  rejection_reason: string | null;
  created_at: string | null;
}

export interface ExpenseApprovalOutput {
  success: boolean;
  message: string;
  decision: string;
  expense_id: number;
  comments: string | null;
  rejection_reason: string | null;
  next_steps: string[];
}

export interface ExpensePolicy {
  id: number;
  category: string;
  max_amount: number;
  approval_threshold: number;
  requires_receipt: boolean;
  description: string;
}

@Injectable({
  providedIn: 'root'
})
export class ExpenseService {
  private apiUrl = 'http://127.0.0.1:8000/api/expenses';

  constructor(private http: HttpClient) {}

  // Submit expense with optional receipt (multipart form)
  submitExpense(formData: FormData): Observable<ExpenseValidationOutput> {
    return this.http.post<ExpenseValidationOutput>(`${this.apiUrl}/submit`, formData);
  }

  // Get expense history for current user
  getExpenseHistory(
    status?: string,
    category?: string,
    startDate?: string,
    endDate?: string,
    offset: number = 0,
    limit: number = 50
  ): Observable<Expense[]> {
    let params = new HttpParams()
      .set('offset', offset.toString())
      .set('limit', limit.toString());

    if (status) params = params.set('status', status);
    if (category) params = params.set('category', category);
    if (startDate) params = params.set('start_date', startDate);
    if (endDate) params = params.set('end_date', endDate);

    return this.http.get<{ expenses: Expense[]; total: number }>(
      `${this.apiUrl}/requests`,
      { params }
    ).pipe(map(r => r.expenses));
  }

  // Get single expense
  getExpenseById(id: number): Observable<Expense> {
    return this.http.get<Expense>(`${this.apiUrl}/requests/${id}`);
  }

  // Cancel a submitted expense
  cancelExpense(id: number): Observable<{ message: string; expense_id: number }> {
    return this.http.put<{ message: string; expense_id: number }>(
      `${this.apiUrl}/requests/${id}/cancel`,
      {}
    );
  }

  // Get pending approvals (manager/hr only)
  getPendingApprovals(): Observable<Expense[]> {
    return this.http.get<{ pending_approvals: Expense[]; total: number }>(
      `${this.apiUrl}/pending-approvals`
    ).pipe(map(r => r.pending_approvals));
  }

  // Approve expense (manager/hr only)
  approveExpense(id: number, comments?: string): Observable<ExpenseApprovalOutput> {
    let params = new HttpParams();
    if (comments) params = params.set('comments', comments);
    return this.http.post<ExpenseApprovalOutput>(
      `${this.apiUrl}/requests/${id}/approve`,
      null,
      { params }
    );
  }

  // Reject expense (manager/hr only)
  rejectExpense(id: number, reason: string): Observable<ExpenseApprovalOutput> {
    let params = new HttpParams().set('reason', reason);
    return this.http.post<ExpenseApprovalOutput>(
      `${this.apiUrl}/requests/${id}/reject`,
      null,
      { params }
    );
  }

  // Get all expense policies
  getPolicies(): Observable<ExpensePolicy[]> {
    return this.http.get<{ policies: ExpensePolicy[] }>(
      `${this.apiUrl}/policies`
    ).pipe(map(r => r.policies));
  }
}
