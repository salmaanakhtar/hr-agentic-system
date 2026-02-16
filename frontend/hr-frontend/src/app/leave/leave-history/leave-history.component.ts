import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LeaveService, LeaveRequest } from '../../services/leave.service';
import { LlmReasoningDisplayComponent } from '../llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-leave-history',
  standalone: true,
  imports: [CommonModule, FormsModule, LlmReasoningDisplayComponent],
  templateUrl: './leave-history.component.html',
  styleUrl: './leave-history.component.scss'
})
export class LeaveHistoryComponent implements OnInit {
  leaveRequests: LeaveRequest[] = [];
  filteredRequests: LeaveRequest[] = [];
  isLoading = false;
  error = '';

  // Filters
  statusFilter: string = 'all';
  searchTerm: string = '';

  // Pagination
  currentPage = 0;
  pageSize = 50;

  // Modal
  selectedRequest: LeaveRequest | null = null;
  showReasoningModal = false;

  constructor(private leaveService: LeaveService) {}

  ngOnInit(): void {
    this.loadLeaveHistory();
  }

  loadLeaveHistory(): void {
    this.isLoading = true;
    this.error = '';

    this.leaveService.getLeaveHistory(undefined, undefined, this.currentPage * this.pageSize, this.pageSize)
      .subscribe({
        next: (requests) => {
          console.log('Raw requests response:', requests);
          console.log('Is array?', Array.isArray(requests));

          // Handle both array and single object responses
          this.leaveRequests = Array.isArray(requests) ? requests : [];
          console.log('Leave requests assigned:', this.leaveRequests.length, 'items');

          this.applyFilters();
          console.log('Filtered requests:', this.filteredRequests.length, 'items');
          console.log('Is loading?', this.isLoading);

          this.isLoading = false;
          console.log('Is loading after set to false?', this.isLoading);
        },
        error: (error) => {
          this.error = 'Failed to load leave history';
          this.isLoading = false;
          console.error('Error loading leave history:', error);
        }
      });
  }

  applyFilters(): void {
    this.filteredRequests = this.leaveRequests.filter(request => {
      // Map "pending" filter to "submitted" status in database
      let matchesStatus = this.statusFilter === 'all';
      if (!matchesStatus) {
        if (this.statusFilter === 'pending') {
          matchesStatus = request.status === 'submitted';
        } else {
          matchesStatus = request.status === this.statusFilter;
        }
      }

      const matchesSearch = !this.searchTerm ||
        request.leave_type.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
        request.reason.toLowerCase().includes(this.searchTerm.toLowerCase());

      return matchesStatus && matchesSearch;
    });
  }

  onStatusFilterChange(): void {
    this.applyFilters();
  }

  onSearchChange(): void {
    this.applyFilters();
  }

  cancelRequest(request: LeaveRequest): void {
    if (!confirm('Are you sure you want to cancel this leave request?')) {
      return;
    }

    this.leaveService.cancelLeaveRequest(request.id).subscribe({
      next: (response) => {
        // Update the request status locally
        const index = this.leaveRequests.findIndex(r => r.id === request.id);
        if (index !== -1) {
          this.leaveRequests[index].status = 'cancelled';
        }
        this.applyFilters();
        alert(response.message || 'Leave request cancelled successfully');
      },
      error: (error) => {
        alert('Failed to cancel request: ' + (error.error?.detail || 'Unknown error'));
        console.error('Error cancelling request:', error);
      }
    });
  }

  viewReasoning(request: LeaveRequest): void {
    this.selectedRequest = request;
    this.showReasoningModal = true;
  }

  closeModal(): void {
    this.showReasoningModal = false;
    this.selectedRequest = null;
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'approved':
        return 'status-approved';
      case 'pending':
      case 'submitted':
        return 'status-pending';
      case 'rejected':
        return 'status-rejected';
      case 'cancelled':
        return 'status-cancelled';
      default:
        return '';
    }
  }

  getStatusIcon(status: string): string {
    switch (status) {
      case 'approved':
        return '✓';
      case 'pending':
      case 'submitted':
        return '⏳';
      case 'rejected':
        return '✗';
      case 'cancelled':
        return '⊘';
      default:
        return '';
    }
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  canCancel(request: LeaveRequest): boolean {
    return request.status === 'submitted';
  }

  getDecision(decision: string | undefined): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (decision === 'AUTO_APPROVE' || decision === 'ESCALATE' || decision === 'REJECT') {
      return decision;
    }
    return 'AUTO_APPROVE';  // Default
  }
}
