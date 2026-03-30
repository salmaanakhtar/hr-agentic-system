import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface PolicyDocument {
  id: number;
  title: string;
  description: string | null;
  category: string;
  filename: string;
  uploaded_by: number;
  is_active: boolean;
  chunk_count: number;
  created_at: string | null;
}

export interface PolicyQueryResult {
  success: boolean;
  message: string;
  reasoning: string;
  answer: string;
  citations: { document_title?: string; chunk_content?: string; [key: string]: any }[];
  confidence: number;
  sources: string[];
}

export interface ComplianceCheckResult {
  success: boolean;
  message: string;
  reasoning: string;
  decision: string;  // COMPLIANT | NON_COMPLIANT | UNCLEAR
  confidence: number;
  citations: { document_title?: string; chunk_content?: string; [key: string]: any }[];
  policy_references: string[];
  recommendations: string[];
}

export interface PolicyReports {
  summary: {
    total_active_documents: number;
    total_inactive_documents: number;
    total_documents: number;
    total_embedded_chunks: number;
  };
  documents_by_category: Record<string, number>;
  most_recent_document: PolicyDocument | null;
}

@Injectable({ providedIn: 'root' })
export class PolicyService {
  private apiUrl = 'http://127.0.0.1:8000/api/policies';

  constructor(private http: HttpClient) {}

  getDocuments(category?: string, activeOnly = true): Observable<PolicyDocument[]> {
    let params = new HttpParams().set('active_only', String(activeOnly));
    if (category) params = params.set('category', category);
    return this.http.get<{ documents: PolicyDocument[]; total: number }>(
      `${this.apiUrl}/documents`, { params }
    ).pipe(map(r => r.documents));
  }

  getDocument(id: number): Observable<PolicyDocument> {
    return this.http.get<PolicyDocument>(`${this.apiUrl}/documents/${id}`);
  }

  uploadDocument(formData: FormData): Observable<{
    message: string;
    document_id: number;
    title: string;
    category: string;
    filename: string;
    chunk_count: number;
  }> {
    return this.http.post<any>(`${this.apiUrl}/documents`, formData);
  }

  deleteDocument(id: number): Observable<{ message: string; document_id: number; title: string; is_active: boolean }> {
    return this.http.delete<any>(`${this.apiUrl}/documents/${id}`);
  }

  queryPolicy(question: string): Observable<PolicyQueryResult> {
    return this.http.post<PolicyQueryResult>(`${this.apiUrl}/query`, { question });
  }

  checkCompliance(scenario: string, context?: Record<string, any>): Observable<ComplianceCheckResult> {
    return this.http.post<ComplianceCheckResult>(`${this.apiUrl}/compliance-check`, {
      scenario,
      context: context ?? null,
    });
  }

  getReports(): Observable<PolicyReports> {
    return this.http.get<PolicyReports>(`${this.apiUrl}/reports`);
  }
}
