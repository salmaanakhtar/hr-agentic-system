from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging
import json
import os
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from . import AuditTool, ToolResult


class AuditLogInput(BaseModel):
    action: str = Field(..., description="Action being performed")
    user_id: int = Field(..., description="ID of the user performing the action")
    resource_type: str = Field(..., description="Type of resource being acted upon")
    resource_id: Optional[int] = Field(default=None, description="ID of the resource")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional action details")
    ip_address: Optional[str] = Field(default=None, description="IP address of the user")
    user_agent: Optional[str] = Field(default=None, description="User agent string")


class AuditLogOutput(BaseModel):
    log_id: int = Field(..., description="ID of the created audit log entry")
    timestamp: datetime = Field(..., description="Timestamp of the log entry")


class AuditQueryInput(BaseModel):
    user_id: Optional[int] = Field(default=None, description="Filter by user ID")
    resource_type: Optional[str] = Field(default=None, description="Filter by resource type")
    resource_id: Optional[int] = Field(default=None, description="Filter by resource ID")
    action: Optional[str] = Field(default=None, description="Filter by action")
    start_date: Optional[datetime] = Field(default=None, description="Start date for query")
    end_date: Optional[datetime] = Field(default=None, description="End date for query")
    limit: int = Field(default=100, description="Maximum number of results")


class AuditQueryOutput(BaseModel):
    logs: List[Dict[str, Any]] = Field(..., description="Audit log entries")
    total_count: int = Field(..., description="Total number of matching logs")


class DatabaseAuditTool(AuditTool[BaseModel, BaseModel]):

    def __init__(self, name: str, description: str, database_url: str):
        super().__init__(name, description, None)
        self.connection_string = database_url
        self.engine = None
        self.session_factory = None

    async def connect(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True
            )
            self.session_factory = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            self.logger.info("Audit database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to audit database: {e}")
            raise

    async def disconnect(self) -> None:
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Audit database connection closed")

    async def log_action(self, user_id: int, action: str, details: Dict[str, Any],
                        ip_address: Optional[str] = None) -> str:
        async with self.session_factory() as session:
            try:
                statement = """
                INSERT INTO audit_logs (action, user_id, details, ip_address, timestamp)
                VALUES (:user_id, :action, :details, :ip_address, :timestamp)
                RETURNING id
                """

                params = {
                    "user_id": user_id,
                    "action": action,
                    "details": json.dumps(details) if details else None,
                    "ip_address": ip_address,
                    "timestamp": datetime.utcnow()
                }

                result = await session.execute(text(statement), params)
                await session.commit()

                log_id = result.scalar()
                self.logger.info(f"Audit log created: {log_id} - {action}")
                return str(log_id)

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Audit logging failed: {e}")
                raise

    async def get_audit_trail(self, user_id: Optional[int] = None,
                            action: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.query_logs(user_id=user_id, action=action,
                                   start_date=start_date, end_date=end_date, limit=100)

    async def log_agent_decision(self, agent_name: str, workflow_id: str,
                               decision: str, reasoning: str,
                               confidence: Optional[float] = None) -> str:
        async with self.session_factory() as session:
            try:
                statement = """
                INSERT INTO agent_decisions (agent_name, workflow_id, decision,
                                           reasoning, confidence, timestamp)
                VALUES (:agent_name, :workflow_id, :decision, :reasoning,
                       :confidence, :timestamp)
                RETURNING id
                """

                params = {
                    "agent_name": agent_name,
                    "workflow_id": workflow_id,
                    "decision": decision,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow()
                }

                result = await session.execute(text(statement), params)
                await session.commit()

                decision_id = result.scalar()
                self.logger.info(f"Agent decision logged: {decision_id} - {agent_name}")
                return str(decision_id)

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Agent decision logging failed: {e}")
                raise

    async def query_logs(self, user_id: Optional[int] = None,
                        resource_type: Optional[str] = None,
                        resource_id: Optional[int] = None,
                        action: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                query = """
                SELECT id, action, user_id, resource_type, resource_id,
                       details, ip_address, user_agent, timestamp
                FROM audit_logs
                WHERE 1=1
                """

                params = {}

                if user_id is not None:
                    query += " AND user_id = :user_id"
                    params["user_id"] = user_id

                if resource_type is not None:
                    query += " AND resource_type = :resource_type"
                    params["resource_type"] = resource_type

                if resource_id is not None:
                    query += " AND resource_id = :resource_id"
                    params["resource_id"] = resource_id

                if action is not None:
                    query += " AND action = :action"
                    params["action"] = action

                if start_date is not None:
                    query += " AND timestamp >= :start_date"
                    params["start_date"] = start_date

                if end_date is not None:
                    query += " AND timestamp <= :end_date"
                    params["end_date"] = end_date

                query += " ORDER BY timestamp DESC LIMIT :limit"
                params["limit"] = limit

                result = await session.execute(text(query), params)
                rows = result.fetchall()

                logs = []
                for row in rows:
                    log_dict = dict(row._mapping)

                    if log_dict.get('details'):
                        try:
                            log_dict['details'] = json.loads(log_dict['details'])
                        except:
                            pass 
                    logs.append(log_dict)

                return logs

            except Exception as e:
                self.logger.error(f"Audit query failed: {e}")
                raise

    async def get_log_count(self, user_id: Optional[int] = None,
                           resource_type: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> int:
        async with self.session_factory() as session:
            try:
                query = "SELECT COUNT(*) FROM audit_logs WHERE 1=1"
                params = {}

                if user_id is not None:
                    query += " AND user_id = :user_id"
                    params["user_id"] = user_id

                if resource_type is not None:
                    query += " AND resource_type = :resource_type"
                    params["resource_type"] = resource_type

                if start_date is not None:
                    query += " AND timestamp >= :start_date"
                    params["start_date"] = start_date

                if end_date is not None:
                    query += " AND timestamp <= :end_date"
                    params["end_date"] = end_date

                result = await session.execute(text(query), params)
                return result.scalar()

            except Exception as e:
                self.logger.error(f"Audit count query failed: {e}")
                raise

    async def execute(self, input_data: BaseModel) -> ToolResult:
        try:
            if isinstance(input_data, AuditLogInput):
                log_id = await self.log_action(
                    input_data.action,
                    input_data.user_id,
                    input_data.resource_type,
                    input_data.resource_id,
                    input_data.details,
                    input_data.ip_address,
                    input_data.user_agent
                )
                output = AuditLogOutput(log_id=log_id, timestamp=datetime.utcnow())
                return self.create_result(True, output.dict())

            elif isinstance(input_data, AuditQueryInput):
                logs = await self.query_logs(
                    input_data.user_id,
                    input_data.resource_type,
                    input_data.resource_id,
                    input_data.action,
                    input_data.start_date,
                    input_data.end_date,
                    input_data.limit
                )
                total_count = await self.get_log_count(
                    input_data.user_id,
                    input_data.resource_type,
                    input_data.start_date,
                    input_data.end_date
                )
                output = AuditQueryOutput(logs=logs, total_count=total_count)
                return self.create_result(True, output.dict())

            else:
                return self.create_result(False, error="Unsupported input type")

        except Exception as e:
            return self.create_result(False, error=str(e))

    def get_input_schema(self):
        return BaseModel

    def get_output_schema(self):
        return BaseModel

class HRAuditTool(DatabaseAuditTool):

    def __init__(self, database_url: Optional[str] = None):
        db_url = database_url or os.getenv('AUDIT_DB_URL') or os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("Audit database URL must be provided or set in AUDIT_DB_URL/DATABASE_URL environment variable")

        super().__init__(
            name="hr_audit",
            description="Audit tool for HR operations and compliance",
            database_url=db_url
        )

    async def log_hr_action(self, action: str, user_id: int, employee_id: int,
                           details: Optional[Dict[str, Any]] = None,
                           ip_address: Optional[str] = None) -> int:
        return await self.log_action(
            action=action,
            user_id=user_id,
            resource_type="employee",
            resource_id=employee_id,
            details=details,
            ip_address=ip_address
        )

    async def log_leave_request(self, user_id: int, employee_id: int,
                               leave_type: str, start_date: str, end_date: str) -> int:
        details = {
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date
        }
        return await self.log_hr_action(
            action="leave_request_created",
            user_id=user_id,
            employee_id=employee_id,
            details=details
        )

    async def log_approval_action(self, user_id: int, approval_id: int,
                                 action: str, comments: Optional[str] = None) -> int:
        details = {"approval_id": approval_id, "comments": comments}
        return await self.log_action(
            action=f"approval_{action}",
            user_id=user_id,
            resource_type="approval",
            resource_id=approval_id,
            details=details
        )

    async def get_employee_audit_trail(self, employee_id: int,
                                      days_back: int = 90) -> List[Dict[str, Any]]:
        start_date = datetime.utcnow() - timedelta(days=days_back)
        return await self.query_logs(
            resource_type="employee",
            resource_id=employee_id,
            start_date=start_date
        )

    async def get_compliance_report(self, start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        logs = await self.query_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000 
        )


        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_actions": len(logs),
            "actions_by_type": {},
            "sensitive_actions": [],
            "compliance_flags": []
        }

        sensitive_actions = ["data_access", "salary_change", "termination", "leave_denied"]

        for log in logs:
            action = log.get("action", "")
            report["actions_by_type"][action] = report["actions_by_type"].get(action, 0) + 1

            if action in sensitive_actions:
                report["sensitive_actions"].append(log)


        if report["actions_by_type"].get("unauthorized_access", 0) > 0:
            report["compliance_flags"].append("Unauthorized access detected")

        return report