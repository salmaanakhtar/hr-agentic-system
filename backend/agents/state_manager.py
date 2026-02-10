from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from datetime import datetime, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager

from .state import WorkflowState, WorkflowStatus, WorkflowStep
from app.database import get_async_db, AsyncSessionLocal
from app.models import WorkflowState as WorkflowStateModel


class StateManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def get_db_session(self):
        session = AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_workflow_state(self, workflow_type: str, user_id: int,
                                  initial_data: Optional[Dict[str, Any]] = None,
                                  ttl_hours: int = 24) -> WorkflowState:
        workflow_state = WorkflowState(
            workflow_type=workflow_type,
            user_id=user_id,
            data=initial_data or {},
            ttl_hours=ttl_hours
        )

        async with self.get_db_session() as session:
            db_state = WorkflowStateModel(
                workflow_id=workflow_state.workflow_id,
                workflow_type=workflow_state.workflow_type,
                user_id=workflow_state.user_id,
                status=workflow_state.status.value,
                expires_at=workflow_state.expires_at,
                data=workflow_state.data,
                workflow_metadata=workflow_state.metadata,
                steps=[step.dict() for step in workflow_state.steps],
                current_step=workflow_state.current_step,
                next_steps=workflow_state.next_steps,
                version=workflow_state.version,
                checksum=workflow_state.checksum,
                is_valid=workflow_state.is_valid,
                ttl_hours=workflow_state.ttl_hours,
                cleanup_scheduled=workflow_state.cleanup_scheduled
            )
            session.add(db_state)

        self.logger.info(f"Created workflow state: {workflow_state.workflow_id}")
        return workflow_state

    async def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        async with self.get_db_session() as session:
            result = await session.execute(
                select(WorkflowStateModel).where(WorkflowStateModel.workflow_id == workflow_id)
            )
            db_state = result.scalar_one_or_none()

            if not db_state:
                return None

            return WorkflowState.from_dict(db_state.to_dict())

    async def update_workflow_state(self, workflow_state: WorkflowState) -> None:
        
        if not workflow_state.validate_integrity():
            self.logger.warning(f"State integrity check failed for workflow {workflow_state.workflow_id}")
            workflow_state.is_valid = False

        async with self.get_db_session() as session:
            result = await session.execute(
                select(WorkflowStateModel).where(WorkflowStateModel.workflow_id == workflow_state.workflow_id)
            )
            db_state = result.scalar_one_or_none()

            if not db_state:
                raise ValueError(f"Workflow state not found: {workflow_state.workflow_id}")

            
            db_state.status = workflow_state.status.value
            db_state.updated_at = workflow_state.updated_at
            db_state.completed_at = workflow_state.completed_at
            db_state.data = workflow_state.data
            db_state.workflow_metadata = workflow_state.metadata
            db_state.steps = [step.dict() for step in workflow_state.steps]
            db_state.current_step = workflow_state.current_step
            db_state.next_steps = workflow_state.next_steps
            db_state.version = workflow_state.version
            db_state.checksum = workflow_state.checksum
            db_state.is_valid = workflow_state.is_valid

        self.logger.debug(f"Updated workflow state: {workflow_state.workflow_id}")

    async def delete_workflow_state(self, workflow_id: str) -> bool:
        async with self.get_db_session() as session:
            result = await session.execute(
                delete(WorkflowStateModel).where(WorkflowStateModel.workflow_id == workflow_id)
            )
            return result.rowcount > 0

    async def get_user_workflow_states(self, user_id: int,
                                     status: Optional[WorkflowStatus] = None,
                                     limit: int = 50) -> List[WorkflowState]:
        async with self.get_db_session() as session:
            query = select(WorkflowStateModel).where(WorkflowStateModel.user_id == user_id)

            if status:
                query = query.where(WorkflowStateModel.status == status.value)

            query = query.order_by(WorkflowStateModel.updated_at.desc()).limit(limit)

            result = await session.execute(query)
            db_states = result.scalars().all()

            return [WorkflowState.from_dict(db_state.to_dict()) for db_state in db_states]

    async def get_expired_workflow_states(self) -> List[WorkflowState]:
        async with self.get_db_session() as session:
            now = datetime.utcnow()
            result = await session.execute(
                select(WorkflowStateModel).where(
                    and_(
                        WorkflowStateModel.expires_at < now,
                        WorkflowStateModel.status.in_([
                            WorkflowStatus.CREATED.value,
                            WorkflowStatus.RUNNING.value,
                            WorkflowStatus.PAUSED.value
                        ])
                    )
                )
            )
            db_states = result.scalars().all()
            return [WorkflowState.from_dict(db_state.to_dict()) for db_state in db_states]

    async def get_completed_workflow_states(self, days_old: int = 7) -> List[WorkflowState]:
        async with self.get_db_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            result = await session.execute(
                select(WorkflowStateModel).where(
                    and_(
                        WorkflowStateModel.status.in_([
                            WorkflowStatus.COMPLETED.value,
                            WorkflowStatus.FAILED.value,
                            WorkflowStatus.CANCELLED.value
                        ]),
                        WorkflowStateModel.completed_at < cutoff_date
                    )
                )
            )
            db_states = result.scalars().all()
            return [WorkflowState.from_dict(db_state.to_dict()) for db_state in db_states]

    async def cleanup_expired_states(self) -> int:
        async with self.get_db_session() as session:
            now = datetime.utcnow()
            result = await session.execute(
                delete(WorkflowStateModel).where(
                    and_(
                        WorkflowStateModel.expires_at < now,
                        WorkflowStateModel.status.in_([
                            WorkflowStatus.CREATED.value,
                            WorkflowStatus.RUNNING.value,
                            WorkflowStatus.PAUSED.value
                        ])
                    )
                )
            )
            count = result.rowcount
            if count > 0:
                self.logger.info(f"Cleaned up {count} expired workflow states")
            return count

    async def cleanup_old_completed_states(self, days_old: int = 30) -> int:
        async with self.get_db_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            result = await session.execute(
                delete(WorkflowStateModel).where(
                    and_(
                        WorkflowStateModel.status.in_([
                            WorkflowStatus.COMPLETED.value,
                            WorkflowStatus.FAILED.value,
                            WorkflowStatus.CANCELLED.value
                        ]),
                        WorkflowStateModel.completed_at < cutoff_date
                    )
                )
            )
            count = result.rowcount
            if count > 0:
                self.logger.info(f"Cleaned up {count} old completed workflow states")
            return count

    async def validate_workflow_state(self, workflow_id: str) -> bool:
        workflow_state = await self.get_workflow_state(workflow_id)
        if not workflow_state:
            return False

        is_valid = workflow_state.validate_integrity()

        if not is_valid:
            async with self.get_db_session() as session:
                await session.execute(
                    update(WorkflowStateModel)
                    .where(WorkflowStateModel.workflow_id == workflow_id)
                    .values(is_valid=False)
                )

        return is_valid

    async def get_workflow_stats(self) -> Dict[str, Any]:
        async with self.get_db_session() as session:
           
            status_counts = {}
            for status in WorkflowStatus:
                result = await session.execute(
                    select(func.count()).select_from(WorkflowStateModel)
                    .where(WorkflowStateModel.status == status.value)
                )
                status_counts[status.value] = result.scalar()

            
            now = datetime.utcnow()
            result = await session.execute(
                select(func.count()).select_from(WorkflowStateModel)
                .where(WorkflowStateModel.expires_at < now)
            )
            expired_count = result.scalar()

            
            result = await session.execute(
                select(func.count()).select_from(WorkflowStateModel)
                .where(WorkflowStateModel.cleanup_scheduled == True)
            )
            cleanup_count = result.scalar()

            return {
                "total_workflows": sum(status_counts.values()),
                "status_counts": status_counts,
                "expired_count": expired_count,
                "cleanup_scheduled": cleanup_count
            }

    async def schedule_cleanup(self, workflow_id: str) -> None:
        async with self.get_db_session() as session:
            await session.execute(
                update(WorkflowStateModel)
                .where(WorkflowStateModel.workflow_id == workflow_id)
                .values(cleanup_scheduled=True)
            )

    async def run_periodic_cleanup(self) -> Dict[str, int]:
        expired_cleaned = await self.cleanup_expired_states()
        old_completed_cleaned = await self.cleanup_old_completed_states()

        return {
            "expired_states_cleaned": expired_cleaned,
            "old_completed_states_cleaned": old_completed_cleaned
        }



state_manager = StateManager()