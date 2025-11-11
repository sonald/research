import asyncio
import random
import time
from typing import Any, Dict, List

# --- Mock Objects to simulate AReaL's components ---

class MockModelRequest:
    """A mock for areal.api.io_struct.ModelRequest."""
    def __init__(self, input_ids: List[int]):
        self.input_ids = input_ids

    def __repr__(self):
        return f"MockModelRequest(input_ids={self.input_ids})"

class MockModelResponse:
    """A mock for areal.api.io_struct.ModelResponse."""
    def __init__(self, output_tokens: List[int]):
        self.output_tokens = output_tokens

    def __repr__(self):
        return f"MockModelResponse(output_tokens={self.output_tokens})"

class MockInferenceEngine:
    """
    A mock for areal.api.engine_api.InferenceEngine that simulates
    asynchronous generation.
    """
    async def agenerate(self, req: MockModelRequest) -> MockModelResponse:
        """Simulates an async call to a remote model."""
        print(f"[{time.strftime('%H:%M:%S')}] Engine received request: {req}")
        # Simulate network latency and model processing time
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)

        # Simulate a model generating some output tokens
        output_tokens = [random.randint(1, 100) for _ in range(10)]
        response = MockModelResponse(output_tokens)

        print(f"[{time.strftime('%H:%M:%S')}] Engine sending response: {response} (after {delay:.2f}s)")
        return response

# --- Simplified RolloutWorkflow Implementation ---

class SimpleRolloutWorkflow:
    """
    A simplified version of areal.api.workflow_api.RolloutWorkflow.
    """
    def __init__(self, workflow_id: int):
        self.workflow_id = workflow_id

    async def arun_episode(
        self, engine: MockInferenceEngine, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs a single rollout episode, simulating a conversation with the model.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Workflow-{self.workflow_id}: Starting episode with data: {data}")

        # 1. Create a request from the input data
        prompt_ids = data.get("prompt_ids", [0])
        request = MockModelRequest(input_ids=prompt_ids)

        # 2. Asynchronously call the inference engine
        response = await engine.agenerate(request)

        # 3. Process the response to create a "trajectory"
        trajectory = {
            "prompt_ids": prompt_ids,
            "output_ids": response.output_tokens,
            "reward": random.random() # Assign a random reward
        }

        print(f"[{time.strftime('%H:%M:%S')}] Workflow-{self.workflow_id}: Finished episode. Trajectory: {trajectory}")
        return trajectory

# --- Main execution logic ---

async def main():
    """
    Simulates the behavior of WorkflowExecutor by running multiple
    rollout workflows concurrently.
    """
    print("--- AReaL Asynchronous Rollout Example ---")
    start_time = time.time()

    # 1. Initialize the mock inference engine
    mock_engine = MockInferenceEngine()

    # 2. Define the number of concurrent rollouts to run
    num_concurrent_rollouts = 5
    print(f"\nStarting {num_concurrent_rollouts} concurrent rollouts...")

    # 3. Create workflow instances and the asyncio tasks
    tasks = []
    for i in range(num_concurrent_rollouts):
        workflow = SimpleRolloutWorkflow(workflow_id=i + 1)
        # Each task will run an episode
        # The input data for each workflow could be different in a real scenario
        input_data = {"prompt_ids": [1, 2, 3 + i]}
        task = asyncio.create_task(workflow.arun_episode(mock_engine, input_data))
        tasks.append(task)

    # 4. Wait for all tasks to complete
    # asyncio.gather runs the tasks concurrently
    results = await asyncio.gather(*tasks)

    end_time = time.time()

    print("\n--- Results ---")
    print(f"All {num_concurrent_rollouts} rollouts completed in {end_time - start_time:.2f} seconds.")
    print("Collected trajectories:")
    for i, trajectory in enumerate(results):
        print(f"  {i+1}: {trajectory}")

if __name__ == "__main__":
    asyncio.run(main())
