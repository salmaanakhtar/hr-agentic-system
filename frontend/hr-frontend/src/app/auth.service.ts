import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { tap } from 'rxjs/operators';

export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface SignupRequest {
  username: string;
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  role: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://127.0.0.1:8000';
  private tokenKey = 'auth_token';
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(private http: HttpClient) {
    const token = localStorage.getItem(this.tokenKey);
    if (token) {
      this.loadUserFromToken(token);
    }
  }

  signup(signupData: SignupRequest): Observable<any> {
    return this.http.post(`${this.apiUrl}/signup`, signupData);
  }

  login(loginData: LoginRequest): Observable<any> {
    return this.http.post(`${this.apiUrl}/login`, loginData).pipe(
      tap((response: any) => {
        if (response.access_token) {
          localStorage.setItem(this.tokenKey, response.access_token);
          this.loadUserFromToken(response.access_token);
        }
      })
    );
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    this.currentUserSubject.next(null);
  }

  isLoggedIn(): boolean {
    return !!localStorage.getItem(this.tokenKey);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  private loadUserFromToken(token: string): void {
    const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
    this.http.get<User>(`${this.apiUrl}/me`, { headers }).subscribe({
      next: (user) => this.currentUserSubject.next(user),
      error: () => {
        this.logout(); // Invalid token
      }
    });
  }

  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }
}
