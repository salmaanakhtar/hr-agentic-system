import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-llm-reasoning-display',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './llm-reasoning-display.component.html',
  styleUrl: './llm-reasoning-display.component.scss'
})
export class LlmReasoningDisplayComponent {
  @Input() decision: 'AUTO_APPROVE' | 'ESCALATE' | 'REJECT' = 'AUTO_APPROVE';
  @Input() reasoning: string = '';
  @Input() confidence: number = 0;
  @Input() factors: string[] = [];
  @Input() recommendations: string[] = [];

  isExpanded: boolean = false;

  toggleExpanded(): void {
    this.isExpanded = !this.isExpanded;
  }

  copyToClipboard(): void {
    const factorsText = this.factors && this.factors.length > 0 ? this.factors.join('\n') : 'None';
    const recommendationsText = this.recommendations && this.recommendations.length > 0 ? this.recommendations.join('\n') : 'None';

    const text = `Decision: ${this.decision}\n\nReasoning:\n${this.reasoning}\n\nConfidence: ${(this.confidence * 100).toFixed(0)}%\n\nFactors Considered:\n${factorsText}\n\nRecommendations:\n${recommendationsText}`;

    navigator.clipboard.writeText(text).then(() => {
      // Show temporary success message (could add a toast notification)
      console.log('Reasoning copied to clipboard');
    });
  }

  getDecisionIcon(): string {
    switch (this.decision) {
      case 'AUTO_APPROVE':
        return '✓';
      case 'ESCALATE':
        return '↑';
      case 'REJECT':
        return '✗';
      default:
        return '?';
    }
  }

  getDecisionClass(): string {
    switch (this.decision) {
      case 'AUTO_APPROVE':
        return 'decision-approved';
      case 'ESCALATE':
        return 'decision-escalate';
      case 'REJECT':
        return 'decision-rejected';
      default:
        return '';
    }
  }

  getConfidenceClass(): string {
    if (this.confidence >= 0.8) return 'confidence-high';
    if (this.confidence >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  }
}
