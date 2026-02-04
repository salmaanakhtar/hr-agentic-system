from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)


class ToolResult(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    data: Optional[Any] = Field(None, description="The result data if successful")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(ABC, Generic[TInput, TOutput]):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(self, input_data: TInput) -> ToolResult:
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
            raise ValueError(f"Invalid input for tool {self.name}: {e}")

    def create_result(self, success: bool, data: Any = None,
                     error: str = None, **metadata) -> ToolResult:
        return ToolResult(
            success=success,
            data=data,
            error=error,
            metadata=metadata
        )


class DatabaseTool(BaseTool[TInput, TOutput]):

    def __init__(self, name: str, description: str, connection_string: str):
        super().__init__(name, description)
        self.connection_string = connection_string

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def execute_statement(self, statement: str, params: Optional[Dict[str, Any]] = None) -> int:
        pass


class VectorTool(BaseTool[TInput, TOutput]):

    def __init__(self, name: str, description: str, vector_db_url: str):
        super().__init__(name, description)
        self.vector_db_url = vector_db_url

    @abstractmethod
    async def search_similar(self, query_embedding: List[float],
                           limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def store_embedding(self, document_id: str, embedding: List[float],
                            metadata: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def delete_embedding(self, document_id: str) -> bool:
        pass


class OCRTool(BaseTool[TInput, TOutput]):

    def __init__(self, name: str, description: str, api_key: Optional[str] = None):
        super().__init__(name, description)
        self.api_key = api_key

    @abstractmethod
    async def extract_text(self, image_data: bytes, image_format: str) -> str:
        pass

    @abstractmethod
    async def extract_text_from_file(self, file_path: str) -> str:
        pass


class AuditTool(BaseTool[TInput, TOutput]):

    def __init__(self, name: str, description: str, log_storage_path: Optional[str] = None):
        super().__init__(name, description)
        self.log_storage_path = log_storage_path

    @abstractmethod
    async def log_action(self, user_id: int, action: str, details: Dict[str, Any],
                        ip_address: Optional[str] = None) -> str:
        pass

    @abstractmethod
    async def get_audit_trail(self, user_id: Optional[int] = None,
                            action: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def log_agent_decision(self, agent_name: str, workflow_id: str,
                               decision: str, reasoning: str,
                               confidence: Optional[float] = None) -> str:
        pass


class PolicyTool(BaseTool[TInput, TOutput]):

    def __init__(self, name: str, description: str, policy_store_path: Optional[str] = None):
        super().__init__(name, description)
        self.policy_store_path = policy_store_path

    @abstractmethod
    async def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def search_policies(self, query: str, category: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_policy_categories(self) -> List[str]:
        pass

    @abstractmethod
    async def validate_against_policy(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        tool = self.get_tool(name)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.get_input_schema().__name__,
            "output_schema": tool.get_output_schema().__name__,
        }


tool_registry = ToolRegistry()

try:
    from .database import SQLAlchemyDatabaseTool, HRDatabaseTool
    from .vector import PgVectorTool, HRVectorTool
    from .ocr import EasyOCR, HROCRTool
    from .audit import DatabaseAuditTool, HRAuditTool
    from .policy import DatabasePolicyTool, HRPolicyTool
except ImportError as e:
    logger.warning(f"Some concrete tool implementations could not be imported: {e}")
    logger.info("Abstract base classes are still available for use")

__all__ = [

    "BaseTool",
    "DatabaseTool",
    "VectorTool",
    "OCRTool",
    "AuditTool",
    "PolicyTool",


    "ToolResult",


    "ToolRegistry",
    "tool_registry",

    "SQLAlchemyDatabaseTool",
    "HRDatabaseTool",
    "PgVectorTool",
    "HRVectorTool",
    "EasyOCR",
    "HROCRTool",
    "DatabaseAuditTool",
    "HRAuditTool",
    "DatabasePolicyTool",
    "HRPolicyTool",
]