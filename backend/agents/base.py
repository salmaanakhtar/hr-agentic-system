from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from pydantic import BaseModel
import uuid
import logging
from datetime import datetime

TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)


class AgentInput(BaseModel):
    user_id: int
    workflow_id: Optional[str] = None
    data: Dict[str, Any] = {}


class AgentOutput(BaseModel):
    success: bool
    message: str
    reasoning: str
    data: Dict[str, Any] = {}
    workflow_id: Optional[str] = None


class Agent(ABC, Generic[TInput, TOutput]):

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(self, input_data: TInput) -> TOutput:

        pass

    @abstractmethod
    def get_input_schema(self) -> type[TInput]:
        pass

    @abstractmethod
    def get_output_schema(self) -> type[TOutput]:
        pass

    def validate_input(self, data: Dict[str, Any]) -> TInput:
        try:
            return self.get_input_schema()(**data)
        except Exception as e:
            raise ValueError(f"Invalid input for agent {self.name}: {e}")

    def create_output(self, success: bool, message: str, reasoning: str,
                     data: Optional[Dict[str, Any]] = None,
                     workflow_id: Optional[str] = None) -> TOutput:
        return self.get_output_schema()(
            success=success,
            message=message,
            reasoning=reasoning,
            data=data or {},
            workflow_id=workflow_id
        )