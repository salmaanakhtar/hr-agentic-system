from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class AgentState(BaseModel):

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.updated_at = datetime.now()

    def update(self, data: Dict[str, Any]) -> None:
        self.data.update(data)
        self.updated_at = datetime.now()

    def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            self.updated_at = datetime.now()

    def clear(self) -> None:
        self.data.clear()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "data": self.data,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":

        return cls(
            workflow_id=data["workflow_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            data=data["data"],
            metadata=data.get("metadata", {})
        )