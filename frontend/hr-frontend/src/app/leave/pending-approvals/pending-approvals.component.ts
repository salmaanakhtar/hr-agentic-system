import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LeaveService, PendingApproval } from '../../services/leave.service';
import { LlmReasoningDisplayComponent } from '../llm-reasoning-display/llm-reasoning-display.component';

@Component({
  selector: 'app-pending-approvals',
  standalone: true,
  imports: [CommonModule, FormsModule, LlmReasoningDisplayComponent],
  templateUrl: './pending-approvals.component.html',
  styleUrl: './pending-approvals.component.scss'
})
export class PendingApprovalsComponent implements OnInit {
  pendingApprovals: PendingApproval[] = [];
  isLoading = false;
  error = '';

  // Rejection modal
  showRejectModal = false;
  selectedApproval: PendingApproval | null = null;
  rejectionReason = '';

  // Approval comments
  approvalComments: { [key: number]: string } = {};

  constructor(private leaveService: LeaveService) {}

  ngOnInit(): void {
    this.loadPendingApprovals();
  }

  loadPendingApprovals(): void {
    this.isLoading = true;
    this.error = '';

    this.leaveService.getPendingApprovals().subscribe({
      next: (approvals) => {
        this.pendingApprovals = approvals;
        this.isLoading = false;
      },
      error: (error) => {
        this.error = 'Failed to load pending approvals';
        this.isLoading = false;
        console.error('Error loading pending approvals:', error);
      }
    });
  }

  approveRequest(approval: PendingApproval): void {
    if (!confirm('Are you sure you want to approve this leave request?')) {
      return;
    }

    const comments = this.approvalComments[approval.id] || undefined;

    this.leaveService.approveLeave(approval.leave_request.id, comments).subscribe({
      next: (response) => {
        // Remove from list
        this.pendingApprovals = this.pendingApprovals.filter(a => a.id !== approval.id);
        alert('Leave request approved successfully');
      },
      error: (error) => {
        alert('Failed to approve request: ' + (error.error?.detail || 'Unknown error'));
        console.error('Error approving request:', error);
      }
    });
  }

  openRejectModal(approval: PendingApproval): void {
    this.selectedApproval = approval;
    this.showRejectModal = true;
    this.rejectionReason = '';
  }

  closeRejectModal(): void {
    this.showRejectModal = false;
    this.selectedApproval = null;
    this.rejectionReason = '';
  }

  confirmReject(): void {
    if (!this.selectedApproval || !this.rejectionReason.trim()) {
      alert('Please provide a reason for rejection');
      return;
    }

    this.leaveService.rejectLeave(this.selectedApproval.leave_request.id, this.rejectionReason).subscribe({
      next: (response) => {
        // Remove from list
        this.pendingApprovals = this.pendingApprovals.filter(a => a.id !== this.selectedApproval?.id);
        this.closeRejectModal();
        alert('Leave request rejected');
      },
      error: (error) => {
        alert('Failed to reject request: ' + (error.error?.detail || 'Unknown error'));
        console.error('Error rejecting request:', error);
      }
    });
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  getPriorityClass(priority: string): string {
    switch (priority.toLowerCase()) {
      case 'high':
        return 'priority-high';
      case 'normal':
        return 'priority-normal';
      case 'low':
        return 'priority-low';
      default:
        return '';
    }
  }

  getPriorityIcon(priority: string): string {
    switch (priority.toLowerCase()) {
      case 'high':
        return '🔴';
      case 'normal':
        return '🟡';
      case 'low':
        return '🟢';
      default:
        return '';
    }
  }

  getDecision(decision: string | undefined): 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' {
    if (decision === 'AUTO_APPROVE' || decision === 'ESCALATE' || decision === 'REJECT') {
      return decision;
    }
    return 'ESCALATE';  // Default
  }
}
