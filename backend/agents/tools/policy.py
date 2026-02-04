from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging
import os
from pathlib import Path
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from . import PolicyTool, ToolResult


class PolicyQueryInput(BaseModel):
    query: str = Field(..., description="Search query for policies")
    category: Optional[str] = Field(default=None, description="Policy category filter")
    limit: int = Field(default=10, description="Maximum number of results")


class PolicyQueryOutput(BaseModel):
    policies: List[Dict[str, Any]] = Field(..., description="Matching policy documents")
    total_results: int = Field(..., description="Total number of matching policies")


class PolicyRetrieveInput(BaseModel):
    policy_id: int = Field(..., description="ID of the policy to retrieve")


class PolicyRetrieveOutput(BaseModel):
    policy: Dict[str, Any] = Field(..., description="Policy document details")


class PolicyValidateInput(BaseModel):
    action: str = Field(..., description="Action to validate against policies")
    context: Dict[str, Any] = Field(..., description="Context for validation")


class PolicyValidateOutput(BaseModel):
    is_compliant: bool = Field(..., description="Whether the action is compliant")
    violations: List[str] = Field(..., description="List of policy violations")
    recommendations: List[str] = Field(..., description="Policy recommendations")


class DatabasePolicyTool(PolicyTool[BaseModel, BaseModel]):

    def __init__(self, name: str, description: str, database_url: str):
        super().__init__(name, description, None)
        self.connection_string = database_url
        self.engine = None
        self.session_factory = None
        self._policies_cache = {} 
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
            self.logger.info("Policy database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to policy database: {e}")
            raise

    async def disconnect(self) -> None:
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Policy database connection closed")

    async def search_policies(self, query: str, category: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                sql_query = """
                SELECT id, title, category, content, tags, effective_date, version
                FROM policies
                WHERE (title ILIKE :query OR content ILIKE :query OR tags::text ILIKE :query)
                """

                params = {"query": f"%{query}%"}

                if category:
                    sql_query += " AND category = :category"
                    params["category"] = category

                sql_query += " ORDER BY effective_date DESC LIMIT :limit"
                params["limit"] = limit

                result = await session.execute(text(sql_query), params)
                rows = result.fetchall()

                return [dict(row._mapping) for row in rows]

            except Exception as e:
                self.logger.error(f"Policy search failed: {e}")
                raise

    async def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        try:
            policy_id_int = int(policy_id)
            return await self.get_policy_by_id(policy_id_int)
        except ValueError:
            return None

    async def get_policy_categories(self) -> List[str]:
        async with self.session_factory() as session:
            try:
                query = "SELECT DISTINCT category FROM policies ORDER BY category"

                result = await session.execute(text(query))
                rows = result.fetchall()

                return [row[0] for row in rows if row[0]]

            except Exception as e:
                self.logger.error(f"Policy categories retrieval failed: {e}")
                return []

    async def validate_against_policy(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.validate_action(action, context)

    async def get_policy_by_id(self, policy_id: int) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                query = """
                SELECT id, title, category, content, tags, effective_date, version,
                       created_at, updated_at
                FROM policies
                WHERE id = :policy_id
                """

                result = await session.execute(text(query), {"policy_id": policy_id})
                row = result.fetchone()

                return dict(row._mapping) if row else None

            except Exception as e:
                self.logger.error(f"Policy retrieval failed: {e}")
                raise

    async def get_policies_by_category(self, category: str) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                query = """
                SELECT id, title, category, content, tags, effective_date, version
                FROM policies
                WHERE category = :category
                ORDER BY effective_date DESC
                """

                result = await session.execute(text(query), {"category": category})
                rows = result.fetchall()

                return [dict(row._mapping) for row in rows]

            except Exception as e:
                self.logger.error(f"Category policy retrieval failed: {e}")
                raise

    async def validate_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            relevant_policies = await self.search_policies(action, limit=5)

            violations = []
            recommendations = []


            if action == "leave_request":
                leave_type = context.get("leave_type", "")
                duration = context.get("duration", 0)

                for policy in relevant_policies:
                    if "annual leave" in policy.get("title", "").lower():
                        content = policy.get("content", "").lower()
                        if "maximum" in content and duration > 25:
                            violations.append("Leave duration exceeds maximum allowed")
                        if "notice" in content:
                            recommendations.append("Ensure proper notice period is given")

            elif action == "salary_change":
                percentage = context.get("percentage_change", 0)

                if abs(percentage) > 20:
                    violations.append("Salary change exceeds normal range")

            elif action == "termination":
                has_documentation = context.get("documentation_complete", False)

                if not has_documentation:
                    violations.append("Termination requires complete documentation")

            is_compliant = len(violations) == 0

            return {
                "is_compliant": is_compliant,
                "violations": violations,
                "recommendations": recommendations,
                "relevant_policies": [p["title"] for p in relevant_policies]
            }

        except Exception as e:
            self.logger.error(f"Policy validation failed: {e}")
            return {
                "is_compliant": False,
                "violations": [f"Validation error: {str(e)}"],
                "recommendations": [],
                "relevant_policies": []
            }

    async def execute(self, input_data: BaseModel) -> ToolResult:
        try:
            if isinstance(input_data, PolicyQueryInput):
                policies = await self.search_policies(
                    input_data.query,
                    input_data.category,
                    input_data.limit
                )
                output = PolicyQueryOutput(policies=policies, total_results=len(policies))
                return self.create_result(True, output.dict())

            elif isinstance(input_data, PolicyRetrieveInput):
                policy = await self.get_policy_by_id(input_data.policy_id)
                if policy:
                    output = PolicyRetrieveOutput(policy=policy)
                    return self.create_result(True, output.dict())
                else:
                    return self.create_result(False, error="Policy not found")

            elif isinstance(input_data, PolicyValidateInput):
                validation_result = await self.validate_action(
                    input_data.action,
                    input_data.context
                )
                output = PolicyValidateOutput(**validation_result)
                return self.create_result(True, output.dict())

            else:
                return self.create_result(False, error="Unsupported input type")

        except Exception as e:
            return self.create_result(False, error=str(e))

    def get_input_schema(self):
        return BaseModel

    def get_output_schema(self):
        return BaseModel 


class HRPolicyTool(DatabasePolicyTool):

    def __init__(self, database_url: Optional[str] = None):
        db_url = database_url or os.getenv('POLICY_DB_URL') or os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("Policy database URL must be provided or set in POLICY_DB_URL/DATABASE_URL environment variable")

        super().__init__(
            name="hr_policy",
            description="Policy tool for HR compliance and document access",
            database_url=db_url
        )

    async def validate_leave_request(self, leave_type: str, duration: int,
                                   employee_seniority: int) -> Dict[str, Any]:
        context = {
            "leave_type": leave_type,
            "duration": duration,
            "employee_seniority": employee_seniority
        }
        return await self.validate_action("leave_request", context)

    async def validate_salary_change(self, current_salary: float,
                                   proposed_salary: float,
                                   employee_role: str) -> Dict[str, Any]:
        percentage_change = ((proposed_salary - current_salary) / current_salary) * 100
        context = {
            "current_salary": current_salary,
            "proposed_salary": proposed_salary,
            "percentage_change": percentage_change,
            "employee_role": employee_role
        }
        return await self.validate_action("salary_change", context)

    async def validate_termination(self, employee_id: int, reason: str,
                                 has_documentation: bool) -> Dict[str, Any]:
        context = {
            "employee_id": employee_id,
            "reason": reason,
            "documentation_complete": has_documentation
        }
        return await self.validate_action("termination", context)

    async def get_leave_policies(self) -> List[Dict[str, Any]]:
        return await self.get_policies_by_category("leave")

    async def get_compensation_policies(self) -> List[Dict[str, Any]]:
        return await self.get_policies_by_category("compensation")

    async def get_disciplinary_policies(self) -> List[Dict[str, Any]]:
        return await self.get_policies_by_category("disciplinary")

    async def search_hr_policies(self, query: str) -> List[Dict[str, Any]]:
        hr_categories = ["leave", "compensation", "disciplinary", "code_of_conduct", "benefits"]
        all_results = []

        for category in hr_categories:
            try:
                results = await self.search_policies(query, category, limit=5)
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(f"Search in category {category} failed: {e}")

        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])

        return unique_results[:10]