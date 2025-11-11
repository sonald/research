# AReaL's Asynchronous Rollout Implementation

## 1. High-Level Overview

AReaL achieves fully asynchronous rollouts through a sophisticated, multi-layered architecture designed for high-throughput, distributed deep learning. The system is built on Python's `asyncio` library, enabling non-blocking I/O and concurrent execution of rollout tasks. This allows the system to efficiently utilize resources by overlapping computation and communication, a critical feature for large-scale reinforcement learning.

The core components of the asynchronous rollout mechanism are:

*   **`RolloutWorkflow`**: An interface that defines the logic for a single rollout episode. Concrete implementations of this interface, such as `MultiTurnWorkflow`, orchestrate the interaction with the inference engine to generate trajectories.
*   **`InferenceEngine`**: An abstraction for the model inference server. The `RemoteSGLangEngine` and `RemotevLLMEngine` are concrete implementations that communicate with the inference server over HTTP. The key method is `agenerate`, which asynchronously sends a request to the server and awaits the response.
*   **`RemoteInfEngine`**: The core of the remote inference functionality. It uses `aiohttp` to make non-blocking HTTP requests to the inference server, allowing it to handle multiple requests concurrently.
*   **`WorkflowExecutor`**: The central orchestrator of the asynchronous rollout process. It manages a pool of worker tasks, each running a `RolloutWorkflow`, and uses a `StalenessManager` to control the number of concurrent rollouts based on the "staleness" of the model weights.

The overall workflow is as follows:

1.  The training script submits rollout requests to the `WorkflowExecutor`.
2.  The `WorkflowExecutor` creates a `RolloutWorkflow` task for each request and adds it to a queue.
3.  The `WorkflowExecutor`'s worker tasks pick up workflows from the queue and execute them.
4.  The `RolloutWorkflow` calls the `InferenceEngine`'s `agenerate` method to get a response from the model.
5.  The `InferenceEngine` sends an asynchronous HTTP request to the inference server and awaits the response.
6.  Once the response is received, the `RolloutWorkflow` processes it and returns a trajectory.
7.  The `WorkflowExecutor` collects the trajectories and returns them to the training script.

## 2. In-Depth Analysis

### 2.1. `RolloutWorkflow`: Defining the Rollout Logic

The `RolloutWorkflow` class in `areal/api/workflow_api.py` is an abstract base class that defines the contract for all rollout workflows. It has a single abstract method, `arun_episode`, which is responsible for running a single episode of the workflow.

```python
# areal/api/workflow_api.py

class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        # ...
```

The `MultiTurnWorkflow` in `areal/workflow/multi_turn.py` is a concrete implementation of this interface. Its `arun_episode` method uses `asyncio.gather` to run multiple rollout tasks concurrently, each of which calls the `_run_one_episode` method.

```python
# areal/workflow/multi_turn.py

class MultiTurnWorkflow(RolloutWorkflow):
    # ...
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        tasks = [
            self._run_one_episode(engine, data) for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)
        # ...
```

The `_run_one_episode` method is where the interaction with the inference engine happens. It calls `engine.agenerate` to get a response from the model and then processes the response to generate a trajectory.

```python
# areal/workflow/multi_turn.py

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], str, str, float, int]:
        # ...
        resp = await engine.agenerate(req)
        # ...
```

### 2.2. `InferenceEngine` and `RemoteInfEngine`: Asynchronous Communication

The `InferenceEngine` is an interface for the model inference server. The `RemoteSGLangEngine` in `areal/engine/sglang_remote.py` is a concrete implementation that uses a `RemoteInfEngine` to communicate with the SGLang inference server.

The `agenerate` method of `RemoteSGLangEngine` simply delegates the call to the `RemoteInfEngine`.

```python
# areal/engine/sglang_remote.py

class RemoteSGLangEngine(InferenceEngine):
    # ...
    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)
```

The `RemoteInfEngine` in `areal/core/remote_inf_engine.py` is where the actual asynchronous communication happens. Its `agenerate` method uses `aiohttp` to send a non-blocking HTTP request to the inference server.

```python
# areal/core/remote_inf_engine.py

class RemoteInfEngine:
    # ...
    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        # ...
        async with aiohttp.ClientSession(...) as session:
            # ...
            while stop_reason not in ["stop", "tool_calls", "length"] and ...:
                # ...
                result = await arequest_with_retry(
                    session=session,
                    addr=server_addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    # ...
                )
                # ...
```

The `arequest_with_retry` function is a helper function that retries the request if it fails.

### 2.3. `WorkflowExecutor`: Orchestrating the Asynchronous Rollout

The `WorkflowExecutor` in `areal/core/workflow_executor.py` is the central orchestrator of the asynchronous rollout process. It manages a pool of worker tasks, each running a `RolloutWorkflow`.

The `submit` method of the `WorkflowExecutor` adds a new rollout task to a queue.

```python
# areal/core/workflow_executor.py

class WorkflowExecutor:
    # ...
    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        # ...
    ) -> None:
        # ...
        self._pending_inputs.append(
            _RolloutTaskInput(
                data=data,
                workflow=resolved_workflow,
                should_accept_fn=resolved_should_accept_fn,
                session_id=session_id,
            )
        )
        # ...
```

The `wait` method of the `WorkflowExecutor` waits for a specified number of rollout tasks to complete and then returns the results.

```python
# areal/core/workflow_executor.py

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | NoResult:
        # ...
        while True:
            # ...
            capacity = self.get_capacity()
            # Submit pending tasks
            for _ in range(capacity):
                if len(self._pending_inputs) == 0:
                    break
                self._commit_one_to_runner()

            if len(self._pending_results) >= count:
                break
            # ...
            try:
                # ...
                batch = self.runner.wait(
                    count=needed, timeout=min(0.1, remaining_timeout)
                )
                # ...
            except TimeoutError:
                pass
        # ...
```

The `_commit_one_to_runner` method creates an async task for the workflow and submits it to the `AsyncTaskRunner`.

```python
# areal/core/workflow_executor.py

    def _commit_one_to_runner(self):
        # ...
        workflow_fn = self._create_workflow_task(pending_task)
        # ...
        try:
            self.runner.submit(workflow_fn)
            # ...
        except TaskQueueFullError:
            # ...
```

The `_create_workflow_task` method creates an async function that executes the workflow.

```python
# areal/core/workflow_executor.py

    def _create_workflow_task(
        self, pending_task: _RolloutTaskInput
    ) -> Callable[[], Awaitable[_RolloutResult | None]]:
        # ...
        async def _execute_workflow() -> _RolloutResult | None:
            # ...
            try:
                # ...
                traj = await pending_task.workflow.arun_episode(
                    self.inference_engine, pending_task.data
                )
                # ...
```
