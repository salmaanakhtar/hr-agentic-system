import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { filter, map, take } from 'rxjs/operators';
import { AuthService } from './auth.service';

const MANAGER_ROLES = ['manager', 'hr', 'admin'];
const HR_ADMIN_ROLES = ['hr', 'admin'];

function roleGuard(allowedRoles: string[]): CanActivateFn {
  return () => {
    const auth = inject(AuthService);
    const router = inject(Router);
    if (!auth.isLoggedIn()) return router.createUrlTree(['/login']);
    return auth.currentUser$.pipe(
      filter(u => u !== null),
      take(1),
      map(u =>
        allowedRoles.includes(u!.role.toLowerCase())
          ? true
          : router.createUrlTree(['/dashboard'])
      )
    );
  };
}

export const managerGuard: CanActivateFn = roleGuard(MANAGER_ROLES);
export const hrAdminGuard: CanActivateFn = roleGuard(HR_ADMIN_ROLES);
