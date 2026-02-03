from typing import Dict, List, Optional, Type
import logging
from .base import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:

    def __init__(self):
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get(self, name: str) -> Agent:
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found")
        return self._agents[name]

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def get_agent_info(self, name: str) -> Dict[str, str]:
        agent = self.get(name)
        return {
            "name": agent.name,
            "description": agent.description,
            "input_schema": agent.get_input_schema().__name__,
            "output_schema": agent.get_output_schema().__name__,
        }

    def unregister(self, name: str) -> None:
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")

    def clear(self) -> None:
        self._agents.clear()
        logger.info("Cleared all agents from registry")



registry = AgentRegistry()


def register_agent(agent: Agent) -> None:
    registry.register(agent)


def get_agent(name: str) -> Agent:
    return registry.get(name)