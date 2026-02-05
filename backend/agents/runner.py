from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .base import Agent, AgentInput, AgentOutput
from .registry import registry, get_agent
from .state import WorkflowState, WorkflowStatus
from .state_manager import state_manager


@dataclass
class ExecutionMetrics:
    """Metrics collected during agent execution."""
    agent_name: str
    execution_time: float
    success: bool
    retry_count: int
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    memory_usage: Optional[int] = None
    workflow_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        return self.execution_time * 1000


@dataclass
class ExecutionResult:
    """Result of an agent execution."""
    agent_name: str
    success: bool
    output: Optional[AgentOutput] = None
    error: Optional[str] = None
    metrics: Optional[ExecutionMetrics] = None
    workflow_state: Optional[WorkflowState] = None


class AgentRunner:
    """
    Core agent execution framework that handles agent discovery,
    instantiation, execution, and monitoring.
    """

    def __init__(self,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_metrics: bool = True,
                 enable_workflow_persistence: bool = True):
        """
        Initialize the agent runner.

        Args:
            max_retries: Maximum number of retry attempts for failed executions
            retry_delay: Delay between retry attempts in seconds
            enable_metrics: Whether to collect execution metrics
            enable_workflow_persistence: Whether to persist workflow state
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_metrics = enable_metrics
        self.enable_workflow_persistence = enable_workflow_persistence

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._execution_history: List[ExecutionResult] = []

    def discover_agents(self) -> List[str]:
        """
        Discover all registered agents.

        Returns:
            List of agent names
        """
        agents = registry.list_agents()
        self.logger.info(f"Discovered {len(agents)} agents: {agents}")
        return agents

    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        Get information about a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent information dictionary
        """
        try:
            info = registry.get_agent_info(agent_name)
            self.logger.debug(f"Retrieved info for agent '{agent_name}': {info}")
            return info
        except ValueError as e:
            self.logger.error(f"Failed to get info for agent '{agent_name}': {e}")
            raise

    def instantiate_agent(self, agent_name: str) -> Agent:
        """
        Instantiate an agent by name.

        Args:
            agent_name: Name of the agent to instantiate

        Returns:
            Instantiated agent instance
        """
        try:
            agent = get_agent(agent_name)
            self.logger.info(f"Instantiated agent '{agent_name}'")
            return agent
        except ValueError as e:
            self.logger.error(f"Failed to instantiate agent '{agent_name}': {e}")
            raise

    async def execute_agent(self,
                           agent_name: str,
                           input_data: Union[Dict[str, Any], AgentInput],
                           workflow_id: Optional[str] = None,
                           create_workflow: bool = True) -> ExecutionResult:
        """
        Execute an agent with the given input data.

        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent (dict or AgentInput instance)
            workflow_id: Optional workflow ID to associate with execution
            create_workflow: Whether to create workflow state if none exists

        Returns:
            ExecutionResult containing the outcome
        """
        start_time = time.time()
        metrics = None

        try:
            # Get agent instance
            agent = self.instantiate_agent(agent_name)

            # Prepare input data
            if isinstance(input_data, dict):
                # Convert dict to appropriate AgentInput type
                input_schema = agent.get_input_schema()
                if hasattr(input_data, 'user_id'):
                    # Already an AgentInput instance
                    processed_input = input_data
                else:
                    # Create AgentInput instance
                    processed_input = input_schema(**input_data)
            else:
                processed_input = input_data

            # Set workflow ID if provided
            if workflow_id and hasattr(processed_input, 'workflow_id'):
                processed_input.workflow_id = workflow_id

            # Create or get workflow state
            workflow_state = None
            if self.enable_workflow_persistence:
                if workflow_id:
                    try:
                        workflow_state = await state_manager.get_workflow_state(workflow_id)
                    except ValueError:
                        if create_workflow:
                            workflow_state = await state_manager.create_workflow_state(
                                workflow_type=f"agent_execution_{agent_name}",
                                user_id=getattr(processed_input, 'user_id', 0),
                                initial_data={"execution_context": {"agent_name": agent_name}}
                            )
                            workflow_id = workflow_state.workflow_id
                            self.logger.info(f"Created workflow state: {workflow_id}")
                elif create_workflow:
                    workflow_state = await state_manager.create_workflow_state(
                        workflow_type=f"agent_execution_{agent_name}",
                        user_id=getattr(processed_input, 'user_id', 0),
                        initial_data={"execution_context": {"agent_name": agent_name}}
                    )
                    workflow_id = workflow_state.workflow_id
                    self.logger.info(f"Created workflow state: {workflow_id}")

            # Execute agent with retry logic
            result = await self._execute_with_retry(agent, processed_input, workflow_id)

            # Update workflow state if it exists
            if workflow_state and result.success:
                workflow_state.set("last_execution_result", result.output.dict() if result.output else {})
                workflow_state.set("last_execution_success", True)
                await state_manager.update_workflow_state(workflow_state)

            # Create metrics
            if self.enable_metrics:
                execution_time = time.time() - start_time
                metrics = ExecutionMetrics(
                    agent_name=agent_name,
                    execution_time=execution_time,
                    success=result.success,
                    retry_count=result.metrics.retry_count if result.metrics else 0,
                    error_message=result.error,
                    start_time=datetime.fromtimestamp(start_time),
                    end_time=datetime.now(),
                    workflow_id=workflow_id
                )

            execution_result = ExecutionResult(
                agent_name=agent_name,
                success=result.success,
                output=result.output,
                error=result.error,
                metrics=metrics,
                workflow_state=workflow_state
            )

            # Log execution result
            self._log_execution_result(execution_result)

            # Store in history
            self._execution_history.append(execution_result)

            return execution_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error executing agent '{agent_name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            if self.enable_metrics:
                metrics = ExecutionMetrics(
                    agent_name=agent_name,
                    execution_time=execution_time,
                    success=False,
                    retry_count=0,
                    error_message=error_msg,
                    start_time=datetime.fromtimestamp(start_time),
                    end_time=datetime.now(),
                    workflow_id=workflow_id
                )

            execution_result = ExecutionResult(
                agent_name=agent_name,
                success=False,
                error=error_msg,
                metrics=metrics
            )

            self._execution_history.append(execution_result)
            return execution_result

    async def _execute_with_retry(self,
                                 agent: Agent,
                                 input_data: AgentInput,
                                 workflow_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute an agent with retry logic.

        Args:
            agent: Agent instance to execute
            input_data: Input data for the agent
            workflow_id: Optional workflow ID

        Returns:
            ExecutionResult
        """
        last_error = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Executing agent '{agent.name}' (attempt {attempt + 1}/{self.max_retries + 1})")

                # Execute the agent
                output = await agent.execute(input_data)

                # Validate output
                if not isinstance(output, agent.get_output_schema()):
                    raise ValueError(f"Agent returned invalid output type: {type(output)}")

                self.logger.info(f"Agent '{agent.name}' executed successfully")
                return ExecutionResult(
                    agent_name=agent.name,
                    success=True,
                    output=output,
                    metrics=ExecutionMetrics(
                        agent_name=agent.name,
                        execution_time=0.0,  # Will be set by caller
                        success=True,
                        retry_count=retry_count,
                        workflow_id=workflow_id
                    )
                )

            except Exception as e:
                last_error = str(e)
                retry_count = attempt

                self.logger.warning(f"Agent '{agent.name}' execution failed (attempt {attempt + 1}): {last_error}")

                # Don't retry on the last attempt
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break

        # All retries exhausted
        error_msg = f"Agent '{agent.name}' failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        self.logger.error(error_msg)

        return ExecutionResult(
            agent_name=agent.name,
            success=False,
            error=error_msg,
            metrics=ExecutionMetrics(
                agent_name=agent.name,
                execution_time=0.0,  # Will be set by caller
                success=False,
                retry_count=retry_count,
                error_message=error_msg,
                workflow_id=workflow_id
            )
        )

    def _log_execution_result(self, result: ExecutionResult) -> None:
        """Log the execution result with appropriate level."""
        if result.success:
            self.logger.info(
                f"Agent '{result.agent_name}' execution completed successfully "
                f"(duration: {result.metrics.duration_ms:.2f}ms, retries: {result.metrics.retry_count})"
            )
        else:
            self.logger.error(
                f"Agent '{result.agent_name}' execution failed: {result.error}"
            )

    def get_execution_history(self,
                            agent_name: Optional[str] = None,
                            limit: Optional[int] = None) -> List[ExecutionResult]:
        """
        Get execution history, optionally filtered by agent name.

        Args:
            agent_name: Optional agent name filter
            limit: Optional limit on number of results

        Returns:
            List of execution results
        """
        history = self._execution_history

        if agent_name:
            history = [r for r in history if r.agent_name == agent_name]

        if limit:
            history = history[-limit:]

        return history

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        if not self._execution_history:
            return {"total_executions": 0}

        total_executions = len(self._execution_history)
        successful_executions = len([r for r in self._execution_history if r.success])
        failed_executions = total_executions - successful_executions

        success_rate = (successful_executions / total_executions) * 100 if total_executions > 0 else 0

        total_retries = sum(r.metrics.retry_count for r in self._execution_history if r.metrics)
        avg_execution_time = sum(r.metrics.execution_time for r in self._execution_history if r.metrics) / total_executions

        agent_stats = {}
        for result in self._execution_history:
            if result.agent_name not in agent_stats:
                agent_stats[result.agent_name] = {"executions": 0, "successes": 0}
            agent_stats[result.agent_name]["executions"] += 1
            if result.success:
                agent_stats[result.agent_name]["successes"] += 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "total_retries": total_retries,
            "average_execution_time": avg_execution_time,
            "agent_stats": agent_stats
        }

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
        self.logger.info("Cleared execution history")


# Global agent runner instance
agent_runner = AgentRunner()