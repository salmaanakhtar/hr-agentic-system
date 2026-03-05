import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

// --- Interfaces ---

export interface JobPosting {
  id: number;
  title: string;
  department: string;
  description: string;
  requirements: string;
  required_skills: string[];
  experience_years: number | null;
  employment_type: 'full_time' | 'part_time' | 'contract';
  location: string | null;
  salary_min: number | null;
  salary_max: number | null;
  status: 'draft' | 'open' | 'closed' | 'on_hold';
  created_by: number;
  created_at: string | null;
  updated_at: string | null;
  has_embedding: boolean;
}

export interface JobPostingCreate {
  title: string;
  department: string;
  description: string;
  requirements: string;
  required_skills: string[];
  experience_years?: number | null;
  employment_type: 'full_time' | 'part_time' | 'contract';
  location?: string | null;
  salary_min?: number | null;
  salary_max?: number | null;
}

export interface JobPostingUpdate {
  title?: string;
  department?: string;
  description?: string;
  requirements?: string;
  required_skills?: string[];
  experience_years?: number | null;
  employment_type?: 'full_time' | 'part_time' | 'contract';
  location?: string | null;
  salary_min?: number | null;
  salary_max?: number | null;
  status?: 'draft' | 'open' | 'closed' | 'on_hold';
}

export interface Candidate {
  id: number;
  first_name: string;
  last_name: string;
  full_name: string;
  email: string;
  phone: string | null;
  cv_filename: string | null;
  cv_url: string | null;
  skills: string[];
  experience_years: number | null;
  education: string[];
  current_title: string | null;
  linkedin_url: string | null;
  has_embedding: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface ApplicationJob {
  id: number;
  title: string;
  department: string;
}

export interface ApplicationCandidate {
  id: number;
  full_name: string;
  email: string;
}

export interface Application {
  id: number;
  job_id: number;
  candidate_id: number;
  status: 'applied' | 'shortlisted' | 'interviewing' | 'offered' | 'rejected' | 'passed';
  similarity_score: number | null;
  skill_coverage: number | null;
  rank: number | null;
  llm_decision: 'SHORTLIST' | 'REVIEW' | 'PASS' | null;
  llm_reasoning: string | null;
  interview_date: string | null;
  interview_notes: string | null;
  reviewed_by: number | null;
  reviewed_at: string | null;
  applied_at: string | null;
  updated_at: string | null;
  job: ApplicationJob | null;
  candidate: ApplicationCandidate | null;
}

export interface HiringApplicationOutput {
  success: boolean;
  message: string;
  application_id: number;
  job_id: number;
  candidate_id: number;
  decision: 'SHORTLIST' | 'REVIEW' | 'PASS';
  reasoning: string;
  similarity_score: number | null;
  skill_coverage: number | null;
  rank: number | null;
  status: string;
  factors: string[];
  recommendations: string[];
}

export interface HiringReportSummary {
  total_jobs: number;
  open_jobs: number;
  total_candidates: number;
  total_applications: number;
}

export interface HiringReport {
  summary: HiringReportSummary;
  jobs_by_status: Record<string, number>;
  applications_by_status: Record<string, number>;
  llm_decisions: {
    shortlisted: number;
    reviewed: number;
    passed: number;
    shortlist_rate: number;
  };
  performance: {
    avg_similarity_shortlisted: number | null;
  };
}

@Injectable({
  providedIn: 'root'
})
export class HiringService {
  private apiUrl = 'http://127.0.0.1:8000/api/hiring';

  constructor(private http: HttpClient) {}

  // --- Job Postings ---

  createJob(data: JobPostingCreate): Observable<{ message: string; job: JobPosting }> {
    return this.http.post<{ message: string; job: JobPosting }>(`${this.apiUrl}/jobs`, data);
  }

  getJobs(
    status?: string,
    department?: string,
    offset: number = 0,
    limit: number = 50
  ): Observable<JobPosting[]> {
    let params = new HttpParams()
      .set('offset', offset.toString())
      .set('limit', limit.toString());

    if (status) params = params.set('status', status);
    if (department) params = params.set('department', department);

    return this.http.get<{ jobs: JobPosting[]; total: number; limit: number; offset: number }>(
      `${this.apiUrl}/jobs`,
      { params }
    ).pipe(map(r => r.jobs));
  }

  getJobById(id: number): Observable<JobPosting> {
    return this.http.get<JobPosting>(`${this.apiUrl}/jobs/${id}`);
  }

  updateJob(id: number, data: JobPostingUpdate): Observable<{ message: string; job: JobPosting }> {
    return this.http.put<{ message: string; job: JobPosting }>(`${this.apiUrl}/jobs/${id}`, data);
  }

  // --- Candidates ---

  uploadCandidate(formData: FormData): Observable<{ message: string; candidate: Candidate }> {
    return this.http.post<{ message: string; candidate: Candidate }>(
      `${this.apiUrl}/candidates`,
      formData
    );
  }

  getCandidates(offset: number = 0, limit: number = 50): Observable<Candidate[]> {
    const params = new HttpParams()
      .set('offset', offset.toString())
      .set('limit', limit.toString());

    return this.http.get<{ candidates: Candidate[]; total: number }>(
      `${this.apiUrl}/candidates`,
      { params }
    ).pipe(map(r => r.candidates));
  }

  getCandidateById(id: number): Observable<Candidate> {
    return this.http.get<Candidate>(`${this.apiUrl}/candidates/${id}`);
  }

  // --- Applications ---

  createApplication(jobId: number, candidateId: number): Observable<HiringApplicationOutput> {
    return this.http.post<HiringApplicationOutput>(`${this.apiUrl}/applications`, {
      job_id: jobId,
      candidate_id: candidateId,
    });
  }

  getApplications(
    jobId?: number,
    candidateId?: number,
    status?: string,
    offset: number = 0,
    limit: number = 50
  ): Observable<Application[]> {
    let params = new HttpParams()
      .set('offset', offset.toString())
      .set('limit', limit.toString());

    if (jobId != null) params = params.set('job_id', jobId.toString());
    if (candidateId != null) params = params.set('candidate_id', candidateId.toString());
    if (status) params = params.set('status', status);

    return this.http.get<{ applications: Application[]; total: number }>(
      `${this.apiUrl}/applications`,
      { params }
    ).pipe(map(r => r.applications));
  }

  updateApplicationStatus(
    id: number,
    status: string,
    notes?: string
  ): Observable<{ message: string; application_id: number; new_status: string }> {
    const body: { status: string; notes?: string } = { status };
    if (notes) body['notes'] = notes;
    return this.http.put<{ message: string; application_id: number; new_status: string }>(
      `${this.apiUrl}/applications/${id}/status`,
      body
    );
  }

  scheduleInterview(
    id: number,
    interviewDate: string,
    interviewNotes?: string
  ): Observable<{ message: string; application_id: number; interview_date: string; status: string }> {
    return this.http.post<{ message: string; application_id: number; interview_date: string; status: string }>(
      `${this.apiUrl}/applications/${id}/schedule-interview`,
      { interview_date: interviewDate, interview_notes: interviewNotes ?? null }
    );
  }

  // --- Reports ---

  getReports(): Observable<HiringReport> {
    return this.http.get<HiringReport>(`${this.apiUrl}/reports`);
  }
}
