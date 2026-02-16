import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

// Interfaces matching backend schemas
export interface LeaveRequestInput {
  user_id?: number; // Optional - backend overrides from JWT
  leave_type: string;
  start_date: string; // ISO format: YYYY-MM-DD
  end_date: string;
  reason: string;
}

export interface LeaveValidationOutput {
  success: boolean;
  message: string;
  reasoning: string;
  is_valid: boolean;
  conflicts: string[];
  auto_approval_eligible: boolean;
  required_approvals: string[];
  validation_warnings: string[];
  recommended_actions: string[];
}

export interface LeaveRequest {
  id: number;
  user_id: number;
  leave_type: string;
  start_date: string;
  end_date: string;
  days_requested: number;
  status: 'draft' | 'submitted' | 'approved' | 'rejected' | 'cancelled';
  reason: string;
  submitted_at: string;
  approved_at?: string;
  approved_by?: number;
  rejected_at?: string;
  rejected_by?: number;
  rejection_reason?: string;
  llm_decision?: string;
  llm_reasoning?: string;
  created_at: string;
  updated_at: string;
}

export interface LeaveBalance {
  id: number;
  user_id: number;
  leave_type: string;
  total_days: number;
  used_days: number;
  carried_forward: number;
  year: number;
  remaining_days: number;
}

export interface PendingApproval {
  id: number;
  leave_request: LeaveRequest;
  submitted_at: string;
  priority: string;
  employee_name?: string;
  department?: string;
}

export interface TeamCalendarEntry {
  id: number;
  user_id: number;
  user_name: string;
  leave_type: string;
  start_date: string;
  end_date: string;
  days_requested: number;
  status: string;
}

export interface PriorityPeriod {
  id: number;
  name: string;
  start_date: string;
  end_date: string;
  description: string;
  is_blackout: boolean;
}

export interface LeaveReport {
  total_requests: number;
  approved_requests: number;
  pending_requests: number;
  rejected_requests: number;
  most_used_leave_type: string;
  average_days_per_request: number;
}

@Injectable({
  providedIn: 'root'
})
export class LeaveService {
  private apiUrl = 'http://127.0.0.1:8000/api/leave';

  constructor(private http: HttpClient) {}

  // Submit leave request (calls Leave Agent)
  submitLeaveRequest(data: LeaveRequestInput): Observable<LeaveValidationOutput> {
    return this.http.post<LeaveValidationOutput>(`${this.apiUrl}/request`, data);
  }

  // Get leave requests with filters
  getLeaveHistory(
    userId?: number,
    status?: string,
    skip: number = 0,
    limit: number = 50
  ): Observable<LeaveRequest[]> {
    let params = new HttpParams()
      .set('skip', skip.toString())
      .set('limit', limit.toString());

    if (userId) {
      params = params.set('user_id', userId.toString());
    }
    if (status) {
      params = params.set('status', status);
    }

    return this.http.get<{ total: number; limit: number; offset: number; requests: LeaveRequest[] }>(
      `${this.apiUrl}/requests`,
      { params }
    ).pipe(
      map(response => response.requests)
    );
  }

  // Get single request details
  getLeaveRequest(requestId: number): Observable<LeaveRequest> {
    return this.http.get<LeaveRequest>(`${this.apiUrl}/requests/${requestId}`);
  }

  // Get leave balances
  getLeaveBalances(userId?: number, year?: number): Observable<LeaveBalance[]> {
    let params = new HttpParams();
    if (userId) {
      params = params.set('user_id', userId.toString());
    }
    if (year) {
      params = params.set('year', year.toString());
    }

    return this.http.get<{ user_id: number; year: number; balances: any[] }>(
      `${this.apiUrl}/balance`,
      { params }
    ).pipe(
      map(response => response.balances.map(b => ({
        id: 0, // Not provided by backend
        user_id: response.user_id,
        leave_type: b.leave_type,
        total_days: b.total_days,
        used_days: b.used_days,
        carried_forward: b.carried_forward,
        year: response.year,
        remaining_days: b.remaining_days
      })))
    );
  }

  // Cancel pending request
  cancelLeaveRequest(requestId: number): Observable<{ success: boolean; message: string; request_id: number }> {
    return this.http.put<{ success: boolean; message: string; request_id: number }>(
      `${this.apiUrl}/requests/${requestId}/cancel`,
      {}
    );
  }

  // Get pending approvals (manager only)
  getPendingApprovals(
    department?: string,
    skip: number = 0,
    limit: number = 50
  ): Observable<PendingApproval[]> {
    let params = new HttpParams()
      .set('skip', skip.toString())
      .set('limit', limit.toString());

    if (department) {
      params = params.set('department', department);
    }

    return this.http.get<{ total: number; pending_approvals: any[] }>(
      `${this.apiUrl}/pending-approvals`,
      { params }
    ).pipe(
      map(response => response.pending_approvals.map(req => ({
        id: req.id,
        leave_request: {
          id: req.id,
          user_id: req.user_id,
          leave_type: req.leave_type,
          start_date: req.start_date,
          end_date: req.end_date,
          days_requested: req.days_requested,
          status: 'submitted' as any,
          reason: req.reason,
          submitted_at: req.submitted_at,
          created_at: req.submitted_at,
          updated_at: req.submitted_at,
          llm_decision: req.llm_decision,
          llm_reasoning: req.llm_reasoning
        },
        submitted_at: req.submitted_at,
        priority: 'normal',
        employee_name: req.employee_name,
        department: undefined
      })))
    );
  }

  // Approve leave request (manager only)
  approveLeave(
    requestId: number,
    comments?: string
  ): Observable<{ message: string; request: LeaveRequest }> {
    return this.http.post<{ message: string; request: LeaveRequest }>(
      `${this.apiUrl}/requests/${requestId}/approve`,
      { comments }
    );
  }

  // Reject leave request (manager only)
  rejectLeave(
    requestId: number,
    reason: string
  ): Observable<{ message: string; request: LeaveRequest }> {
    return this.http.post<{ message: string; request: LeaveRequest }>(
      `${this.apiUrl}/requests/${requestId}/reject`,
      { reason }
    );
  }

  // Get team calendar
  getTeamCalendar(
    department?: string,
    month?: string  // Format: YYYY-MM
  ): Observable<{
    leave_requests: TeamCalendarEntry[];
    priority_periods: PriorityPeriod[];
  }> {
    let params = new HttpParams();
    if (department) {
      params = params.set('department', department);
    }
    if (month) {
      params = params.set('month', month);
    }

    return this.http.get<{
      leave_requests: TeamCalendarEntry[];
      priority_periods: PriorityPeriod[];
    }>(`${this.apiUrl}/team-calendar`, { params });
  }

  // Adjust leave balance (HR only)
  adjustLeaveBalance(
    userId: number,
    leaveType: string,
    adjustment: number,
    reason: string
  ): Observable<{ message: string; balance: LeaveBalance }> {
    return this.http.post<{ message: string; balance: LeaveBalance }>(
      `${this.apiUrl}/balance/adjust`,
      {
        user_id: userId,
        leave_type: leaveType,
        adjustment,
        reason
      }
    );
  }

  // Get leave usage reports (HR/Manager)
  getLeaveReports(
    startDate?: string,
    endDate?: string,
    department?: string
  ): Observable<LeaveReport> {
    let params = new HttpParams();
    if (startDate) {
      params = params.set('start_date', startDate);
    }
    if (endDate) {
      params = params.set('end_date', endDate);
    }
    if (department) {
      params = params.set('department', department);
    }

    return this.http.get<LeaveReport>(`${this.apiUrl}/reports`, { params });
  }
}
