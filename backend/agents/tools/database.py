from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from . import DatabaseTool, ToolResult


class DatabaseQueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Query parameters")


class DatabaseQueryOutput(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Query result rows")
    row_count: int = Field(..., description="Number of rows returned")


class DatabaseExecuteInput(BaseModel):
    statement: str = Field(..., description="SQL statement to execute")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Statement parameters")


class DatabaseExecuteOutput(BaseModel):
    affected_rows: int = Field(..., description="Number of rows affected")


class SQLAlchemyDatabaseTool(DatabaseTool[BaseModel, BaseModel]):

    def __init__(self, name: str, description: str, database_url: str):
        super().__init__(name, description, database_url)
        self.engine = None
        self.session_factory = None

    async def connect(self) -> None:
        try:
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True
            )
            self.session_factory = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self) -> None:
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connection closed")

    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                result = await session.execute(text(query), params or {})
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise

    async def execute_statement(self, statement: str, params: Optional[Dict[str, Any]] = None) -> int:
        async with self.session_factory() as session:
            try:
                result = await session.execute(text(statement), params or {})
                await session.commit()
                return result.rowcount
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Statement execution failed: {e}")
                raise

    async def execute(self, input_data: BaseModel) -> ToolResult:
        try:
            if isinstance(input_data, DatabaseQueryInput):
                rows = await self.query(input_data.query, input_data.params)
                output = DatabaseQueryOutput(rows=rows, row_count=len(rows))
                return self.create_result(True, output.dict())
            elif isinstance(input_data, DatabaseExecuteInput):
                affected_rows = await self.execute_statement(input_data.statement, input_data.params)
                output = DatabaseExecuteOutput(affected_rows=affected_rows)
                return self.create_result(True, output.dict())
            else:
                return self.create_result(False, error="Unsupported input type")
        except Exception as e:
            return self.create_result(False, error=str(e))

    def get_input_schema(self):
        return BaseModel

    def get_output_schema(self):
        return BaseModel 


class HRDatabaseTool(SQLAlchemyDatabaseTool):

    def __init__(self, database_url: Optional[str] = None):
        db_url = database_url or os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("Database URL must be provided or set in DATABASE_URL environment variable")

        super().__init__(
            name="hr_database",
            description="Database tool for HR operations",
            database_url=db_url
        )

    async def get_employee_by_id(self, employee_id: int) -> Optional[Dict[str, Any]]:
        query = """
        SELECT e.*, u.email, u.role
        FROM employees e
        JOIN users u ON e.user_id = u.id
        WHERE e.id = :employee_id
        """
        rows = await self.query(query, {"employee_id": employee_id})
        return rows[0] if rows else None

    async def get_pending_approvals(self, manager_id: int) -> List[Dict[str, Any]]:
        query = """
        SELECT * FROM approvals
        WHERE approver_id = :manager_id
        AND status = 'pending'
        ORDER BY created_at DESC
        """
        return await self.query(query, {"manager_id": manager_id})

    async def create_approval_request(self, request_data: Dict[str, Any]) -> int:
        statement = """
        INSERT INTO approvals (requester_id, approver_id, request_type, request_data, status)
        VALUES (:requester_id, :approver_id, :request_type, :request_data, :status)
        """
        params = {
            **request_data,
            "status": "pending"
        }
        return await self.execute_statement(statement, params)

    async def update_approval_status(self, approval_id: int, status: str,
                                   comments: Optional[str] = None) -> bool:
        statement = """
        UPDATE approvals
        SET status = :status, comments = :comments, updated_at = NOW()
        WHERE id = :approval_id
        """
        params = {
            "approval_id": approval_id,
            "status": status,
            "comments": comments
        }
        affected = await self.execute_statement(statement, params)
        return affected > 0