# Analysis of Asynchronous Off-Policy Training Frameworks for RL-based LLM Training

This document provides a high-level comparison of four reinforcement learning frameworks that are used for training Large Language Models (LLMs) using asynchronous off-policy algorithms. The goal is to understand their underlying architecture, implementation details, and trade-offs.

A detailed analysis of each framework can be found in the corresponding markdown files:
- [Prime-RL Analysis](./prime-rl.md)
- [AReaL Analysis](./AReaL.md)
- [SLIME Analysis](./slime.md)
- [VERL Analysis](./verl.md)

## High-Level Comparison

| Feature                 | Prime-RL | AReaL | SLIME | VERL |
| ----------------------- | -------- | ----- | ----- | ---- |
| **Core Technology**     | Shared Filesystem + HTTP | Direct TCP/IP (via env vars) | Ray Framework | Ray Framework |
| **Architecture**        | 3-Component (Trainer, Orchestrator, Inference) | 2-Component (Trainer, Remote Rollout) | Ray Actors (Trainer, Rollout Manager) | Ray Actors (Controller, Hybrid Worker) |
| **Communication**       | Weights via Filesystem/NCCL. Data via Filesystem. Control via HTTP. | Direct RPC-like calls from Trainer to Rollout workers for control and weight updates. | Ray Object Store for data. Ray remote method calls for control and weights. | Ray remote method calls for control. Ray Object Store for data. |
| **Parameter Sync**      | Orchestrator polls for new weights and pushes update command to inference servers. | Trainer explicitly pauses rollout, pushes weights, and resumes rollout. | Main script calls `update_weights` on trainer actor, which then pushes to rollout actor. | Controller dispatches weight updates to hybrid workers, which then switch context. |
| **LLM Optimizations**   | vLLM for inference, FSDP for training. | SGLang/vLLM for inference, FSDP/Megatron for training. | SGLang for inference, Megatron/FSDP for training. | SGLang/vLLM for inference, FSDP/Megatron for training. |
| **Primary Use Case**    | Large-scale, decentralized training with decoupled components. | Tightly-coupled, controlled asynchronous training with simplified architecture. | Highly parallel, pipelined training leveraging the Ray ecosystem. | Resource-efficient training using hybrid workers that switch between roles. |
