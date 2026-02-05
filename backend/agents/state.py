from typing import Any, Dict, Optional, List, Set
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from datetime import datetime, timedelta
import json
import logging


class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class WorkflowStep(BaseModel):
    step_id: str
    step_name: str
    agent_name: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = Field(default_factory=list)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Override dict to handle datetime serialization for JSON storage."""
        data = super().dict(*args, **kwargs)
        # Convert datetime objects to ISO strings for JSON serialization
        if data.get('started_at'):
            data['started_at'] = data['started_at'].isoformat()
        if data.get('completed_at'):
            data['completed_at'] = data['completed_at'].isoformat()
        return data


class WorkflowState(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: str
    user_id: int
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    completed_at: Optional[datetime] = None

    # Workflow data
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Step tracking
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)

    # Validation and consistency
    version: int = Field(default=1)
    checksum: Optional[str] = None
    is_valid: bool = Field(default=True)

    # Lifecycle management
    ttl_hours: int = Field(default=24)
    cleanup_scheduled: bool = Field(default=False)

    @validator('expires_at', pre=True, always=True)
    def set_expires_at(cls, v, values):
        if v is None and 'ttl_hours' in values:
            ttl_hours = values.get('ttl_hours', 24)
            return datetime.now() + timedelta(hours=ttl_hours)
        return v

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def update(self, data: Dict[str, Any]) -> None:
        self.data.update(data)
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            self.updated_at = datetime.now()
            self.version += 1
            self._update_checksum()

    def clear(self) -> None:
        self.data.clear()
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def add_step(self, step: WorkflowStep) -> None:
        self.steps.append(step)
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def update_step_status(self, step_id: str, status: str,
                          error_message: Optional[str] = None) -> None:
        for step in self.steps:
            if step.step_id == step_id:
                step.status = status
                if status == "running" and not step.started_at:
                    step.started_at = datetime.now()
                elif status in ["completed", "failed", "cancelled"]:
                    step.completed_at = datetime.now()
                    if error_message:
                        step.error_message = error_message
                break
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        return next((step for step in self.steps if step.step_id == step_id), None)

    def get_pending_steps(self) -> List[WorkflowStep]:
        return [step for step in self.steps if step.status == "pending"]

    def get_completed_steps(self) -> List[WorkflowStep]:
        return [step for step in self.steps if step.status == "completed"]

    def get_failed_steps(self) -> List[WorkflowStep]:
        return [step for step in self.steps if step.status == "failed"]

    def can_proceed_to_step(self, step_id: str) -> bool:
        step = self.get_step(step_id)
        if not step:
            return False

        # Check if all dependencies are completed
        for dep_id in step.dependencies:
            dep_step = self.get_step(dep_id)
            if not dep_step or dep_step.status != "completed":
                return False

        return True

    def mark_workflow_completed(self) -> None:
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def mark_workflow_failed(self, error_message: str) -> None:
        self.status = WorkflowStatus.FAILED
        self.metadata["error_message"] = error_message
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.version += 1
        self._update_checksum()

    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def should_cleanup(self) -> bool:
        if self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            return True
        if self.is_expired():
            return True
        return False

    def _update_checksum(self) -> None:
        import hashlib
        data_str = json.dumps({
            "workflow_id": self.workflow_id,
            "version": self.version,
            "data": self.data,
            "steps": [step.dict() for step in self.steps]
        }, sort_keys=True, default=str)
        self.checksum = hashlib.md5(data_str.encode()).hexdigest()

    def validate_integrity(self) -> bool:
        if not self.checksum:
            return True

        import hashlib
        data_str = json.dumps({
            "workflow_id": self.workflow_id,
            "version": self.version,
            "data": self.data,
            "steps": [step.dict() for step in self.steps]
        }, sort_keys=True, default=str)
        expected_checksum = hashlib.md5(data_str.encode()).hexdigest()
        return self.checksum == expected_checksum

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "data": self.data,
            "metadata": self.metadata,
            "steps": [step.dict() for step in self.steps],
            "current_step": self.current_step,
            "next_steps": self.next_steps,
            "version": self.version,
            "checksum": self.checksum,
            "is_valid": self.is_valid,
            "ttl_hours": self.ttl_hours,
            "cleanup_scheduled": self.cleanup_scheduled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        steps_data = data.get("steps", [])
        steps = []
        for step_data in steps_data:
            # Convert ISO strings back to datetime objects for WorkflowStep
            if step_data.get("started_at"):
                step_data["started_at"] = datetime.fromisoformat(step_data["started_at"])
            if step_data.get("completed_at"):
                step_data["completed_at"] = datetime.fromisoformat(step_data["completed_at"])
            steps.append(WorkflowStep(**step_data))

        return cls(
            workflow_id=data["workflow_id"],
            workflow_type=data["workflow_type"],
            user_id=data["user_id"],
            status=WorkflowStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            steps=steps,
            current_step=data.get("current_step"),
            next_steps=data.get("next_steps", []),
            version=data.get("version", 1),
            checksum=data.get("checksum"),
            is_valid=data.get("is_valid", True),
            ttl_hours=data.get("ttl_hours", 24),
            cleanup_scheduled=data.get("cleanup_scheduled", False)
        )