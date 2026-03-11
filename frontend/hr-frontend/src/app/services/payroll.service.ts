import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface Payslip {
  id: number;
  employee_id: number;
  employee_name: string | null;
  pay_cycle_id: number;
  gross_pay: number;
  deductions_leave: number;
  deductions_tax: number;
  net_pay: number;
  days_worked: number;
  leave_days_taken: number;
  llm_decision: string | null;   // APPROVE | HOLD | FLAG
  llm_reasoning: string | null;
  status: 'draft' | 'approved' | 'paid';
  approved_by: number | null;
  approved_at: string | null;
  created_at: string | null;
  updated_at: string | null;
  period_start?: string | null;
  period_end?: string | null;
}

export interface PayCycle {
  id: number;
  period_start: string | null;
  period_end: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed';
  run_by: number | null;
  run_at: string | null;
  payslip_count: number;
  created_at: string | null;
  updated_at: string | null;
  payslips?: Payslip[];
}

export interface PayCycleRunEmployeeResult {
  employee_id: number;
  employee_name: string;
  success: boolean;
  decision: string;
  payslip_id: number | null;
  net_pay: number | null;
  message: string;
}

export interface PayCycleRunResult {
  pay_cycle_id: number;
  period_start: string;
  period_end: string;
  status: string;
  total_employees: number;
  failed_count: number;
  results: PayCycleRunEmployeeResult[];
}

@Injectable({ providedIn: 'root' })
export class PayrollService {
  private apiUrl = 'http://127.0.0.1:8000/api/payroll';

  constructor(private http: HttpClient) {}

  runCycle(periodStart: string, periodEnd: string): Observable<PayCycleRunResult> {
    return this.http.post<PayCycleRunResult>(`${this.apiUrl}/run-cycle`, {
      period_start: periodStart,
      period_end: periodEnd,
    });
  }

  getCycles(status?: string): Observable<PayCycle[]> {
    let params = new HttpParams();
    if (status) params = params.set('status', status);
    return this.http.get<{ pay_cycles: PayCycle[]; total: number }>(
      `${this.apiUrl}/cycles`, { params }
    ).pipe(map(r => r.pay_cycles));
  }

  getCycle(id: number): Observable<PayCycle & { payslips: Payslip[] }> {
    return this.http.get<PayCycle & { payslips: Payslip[] }>(`${this.apiUrl}/cycles/${id}`);
  }

  getPayslips(cycleId?: number, employeeId?: number, status?: string): Observable<Payslip[]> {
    let params = new HttpParams();
    if (cycleId != null) params = params.set('cycle_id', cycleId.toString());
    if (employeeId != null) params = params.set('employee_id', employeeId.toString());
    if (status) params = params.set('status', status);
    return this.http.get<{ payslips: Payslip[]; total: number }>(
      `${this.apiUrl}/payslips`, { params }
    ).pipe(map(r => r.payslips));
  }

  getPayslip(id: number): Observable<Payslip> {
    return this.http.get<Payslip>(`${this.apiUrl}/payslips/${id}`);
  }

  approvePayslip(id: number, notes?: string): Observable<{ message: string; payslip_id: number; status: string }> {
    return this.http.put<{ message: string; payslip_id: number; status: string }>(
      `${this.apiUrl}/payslips/${id}/approve`,
      { notes: notes ?? null }
    );
  }

  getReports(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/reports`);
  }
}
