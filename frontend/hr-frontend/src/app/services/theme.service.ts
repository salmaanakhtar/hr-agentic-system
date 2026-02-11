import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private isDarkMode = new BehaviorSubject<boolean>(false);
  public isDarkMode$ = this.isDarkMode.asObservable();

  constructor() {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      this.setTheme(savedTheme === 'dark');
    } else {
      // Check system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      this.setTheme(prefersDark);
    }
  }

  toggleTheme(): void {
    this.setTheme(!this.isDarkMode.value);
  }

  setTheme(isDark: boolean): void {
    this.isDarkMode.next(isDark);
    const theme = isDark ? 'dark' : 'light';

    // Update document class
    document.documentElement.setAttribute('data-theme', theme);

    // Save preference
    localStorage.setItem('theme', theme);
  }

  getCurrentTheme(): boolean {
    return this.isDarkMode.value;
  }
}
