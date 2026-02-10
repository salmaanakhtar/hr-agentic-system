import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents import (
    agent_runner,
    register_agent,
    HelloWorldAgent,
    BasicAgentInput
)


async def demonstrate_agent_execution():
    print("Agent Runner Demonstration")
    print("=" * 50)


    hello_agent = HelloWorldAgent()
    register_agent(hello_agent)

    print("Available Agents:")
    agents = agent_runner.discover_agents()
    for agent in agents:
        info = agent_runner.get_agent_info(agent)
        print(f"  • {info['name']}: {info['description']}")

    print("\n Executing Hello World Agent...")


    input_data = {
        "user_id": 1,
        "data": {
            "name": "Demo User",
            "message": "Welcome to the HR Agentic System!"
        }
    }

    result = await agent_runner.execute_agent("hello_world", input_data, create_workflow=True)

    print("\n Execution Results:")
    print(f"  Agent: {result.agent_name}")
    print(f"  Success: {result.success}")
    print(f"  Execution Time: {result.metrics.duration_ms:.2f}ms")
    print(f"  Retries: {result.metrics.retry_count}")
    print(f"  Workflow ID: {result.workflow_state.workflow_id if result.workflow_state else 'None'}")

    if result.output:
        print("\n Agent Response:")
        print(f"  Message: {result.output.message}")
        print(f"  Reasoning: {result.output.reasoning}")
        print(f"  Data: {result.output.data}")

    print("\n Execution Statistics:")
    stats = agent_runner.get_execution_stats()
    print(f"  Total Executions: {stats['total_executions']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Average Execution Time: {stats['average_execution_time']:.3f}s")

    print("\n Agent execution demonstration completed!")


async def demonstrate_error_handling():
    print("\n Error Handling Demonstration")
    print("=" * 50)

    print(" Testing nonexistent agent...")
    result = await agent_runner.execute_agent("nonexistent_agent", {"user_id": 1})
    print(f"  Result: {result.success} - {result.error}")

    print("\n Testing invalid input...")
    result = await agent_runner.execute_agent("hello_world", {})
    print(f"  Result: {result.success} - {result.error[:100]}...")

    print("\n Error handling demonstration completed!")


async def main():
    try:
        await demonstrate_agent_execution()
        await demonstrate_error_handling()
        print("\nAll demonstrations completed successfully!")
        return 0
    except Exception as e:
        print(f"\n Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)