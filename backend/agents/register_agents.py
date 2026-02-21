"""
Agent Registration Module

Registers all available agents with the agent registry.
This should be called during application startup.
"""

from agents.registry import registry
from agents.examples import HelloWorldAgent
from agents.leave_agent import LeaveAgent
from agents.expense_agent import ExpenseAgent
from agents.orchestrator import OrchestratorAgent
import logging

logger = logging.getLogger(__name__)


def register_all_agents():
    """
    Register all available agents with the agent registry.

    This function should be called during application startup to ensure
    all agents are available for execution.
    """
    logger.info("Registering agents...")

    # Register example agents
    registry.register(HelloWorldAgent())
    logger.info("Registered: HelloWorldAgent")

    # Register business logic agents
    registry.register(LeaveAgent())
    logger.info("Registered: LeaveAgent")

    registry.register(ExpenseAgent())
    logger.info("Registered: ExpenseAgent")

    # Register orchestrator
    registry.register(OrchestratorAgent())
    logger.info("Registered: OrchestratorAgent")

    # Log all registered agents
    registered = registry.list_agents()
    logger.info(f"Total agents registered: {len(registered)}")
    logger.info(f"Available agents: {', '.join(registered)}")

    return registered


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    registered = register_all_agents()
    print(f"\nSuccessfully registered {len(registered)} agents:")
    for agent_name in registered:
        info = registry.get_agent_info(agent_name)
        print(f"  - {agent_name}: {info['description']}")
