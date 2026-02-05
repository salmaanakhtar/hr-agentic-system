from .base import Agent
from .schemas import BasicAgentInput, BasicAgentOutput
import logging


class HelloWorldAgent(Agent[BasicAgentInput, BasicAgentOutput]):


    def __init__(self):
        super().__init__(
            name="hello_world",
            description="A simple agent that returns a greeting"
        )

    async def execute(self, input_data: BasicAgentInput) -> BasicAgentOutput:
        name = input_data.data.get("name", "World")

        greeting = f"Hello, {name}! This is the hello_world agent."

        if input_data.workflow_id:
            self.logger.info(f"Processing workflow {input_data.workflow_id}")

        return self.create_output(
            success=True,
            message=greeting,
            reasoning="Generated a personalized greeting based on input data",
            data={
                "greeting": greeting,
                "agent_name": self.name,
                "workflow_id": input_data.workflow_id
            },
            workflow_id=input_data.workflow_id
        )

    def get_input_schema(self):
        return BasicAgentInput

    def get_output_schema(self):
        return BasicAgentOutput