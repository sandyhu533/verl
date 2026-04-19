# veRL Reading Notes

Milestone-based reading log for the veRL codebase. Each numbered file in this
directory corresponds to one milestone below. Priority tags gate attention
budget — fully internalize **[critical]** before moving forward, treat
**[optional]** as "only if you need it."

This branch (`notes`) is not merged into `main` and is not the base for any
PR branches. All future code contributions branch off `upstream/main`.

---

## Priority legend

- **[critical]** — core call chain; fully internalize before moving on.
- **[core]** — important for coverage; read thoroughly and take notes.
- **[optional]** — skim, revisit only when a follow-up question demands it.

## Time budget

| Path | Milestones | Total effort |
|---|---|---|
| **Critical path** | M1 + M2 + M3 + M4 + M5 | ~20 hours |
| **Full core** | + M6 + M7 | ~30 hours |
| **Complete** | + M8 + M9 | ~35 hours |

At 2 h/day, the critical path is ~2 weeks; at 4 h/day, ~1 week. Optional
milestones can be deferred indefinitely.

---

## M1 — Data Protocol & Entry Point **[critical]** (~2 h)

**Why first**: `DataProto` is the universal data container. Every worker call,
every dispatch, every rollout → reward → train hand-off travels as a DataProto.
Reading this first turns later files into concrete object manipulation rather
than abstract tensor dict juggling.

**Focus files**:
- `verl/protocol.py:318–949` — `DataProto` class, `chunk()` (L864), `concat()` (L917), `union()`, `pop()`, serialization
- `verl/trainer/main_ppo.py:36–462` — Hydra entry, `run_ppo()`, `TaskRunner`, `create_rl_dataset()`
- `verl/trainer/config/ppo_trainer.yaml` + `verl/trainer/config/algorithm.py` — config surface (skim only)

**What to trace**: `DataProto.chunk(n)` → how auto-padding works, how non-tensor
batches are split, how `meta_info` flows through; then `DataProto.concat(list)`
→ metric aggregation rules.

**Notes deliverable**: `01-data-protocol.md` — DataProto field taxonomy
(prompts / responses / rewards / advantages), chunk/concat invariants,
serialization path.

**Skip**: Hydra/OmegaConf plumbing, `logger` setup, validation dataloader nuances.

---

## M2 — Single Controller & Ray Orchestration **[critical]** (~6 h)

**Why this milestone is the longest**: The `single_controller` layer is what
makes veRL pluggable across parallelism strategies. It owns the
dispatch/collect mechanism, resource placement, and worker colocation logic —
understanding it is the gate that unlocks every downstream milestone.

**Focus files (read in this order)**:
- `verl/single_controller/base/worker.py:76–349` — `Worker` base class, rank/world_size/mesh registration (`_register_dispatch_collect_info` L86)
- `verl/single_controller/base/worker_group.py:123–256` — `WorkerGroup`, method binding (`_bind_worker_method` L185)
- `verl/single_controller/base/decorator.py:26–445` — `Dispatch` enum, dispatch/collect functions (`dispatch_dp_compute_data_proto` L167, `collect_dp_compute_data_proto` L191, `dispatch_lazy_compute_data_proto` L266)
- `verl/single_controller/ray/base.py:112–160` — `RayResourcePool` (placement groups)
- `verl/single_controller/ray/base.py:412–905` — `RayWorkerGroup` (actor spawn, `execute_all_async` L860)
- `verl/single_controller/ray/base.py:982–1122` — `FusedWorker` (colocated actors)

**What to trace**: end-to-end call chain for `actor_rollout_wg.generate_sequences(batch)`:
1. Driver invokes bound method
2. `dispatch_fn` chunks DataProto into `world_size` slices
3. `execute_all_async` → N Ray tasks
4. Each worker runs `generate_sequences` on its chunk
5. `collect_fn` concatenates results
6. Driver receives merged DataProto

**Notes deliverable**: `02-single-controller.md` —
- Class diagram (Worker → WorkerGroup → RayWorkerGroup, Dispatch modes)
- Full trace of the call chain above
- FusedWorker internals: why `_fuw_execute` uses string-based dispatch, what's fragile about it
- Predefined `Dispatch` modes and their dispatch/collect pairs

**Skip**: legacy non-Ray dispatch paths, `SubRayResourcePool` edge cases,
detached-worker reattachment.

---

## M3 — PPO Training Loop **[critical]** (~3 h)

**Why now**: With DataProto + single_controller internalized, the training loop
reads as pure orchestration.

**Focus files**:
- `verl/trainer/ppo/ray_trainer.py:234–400` — `RayPPOTrainer.__init__`, dataloader wiring
- `verl/trainer/ppo/ray_trainer.py:688–885` — `init_workers()` (resource pool + role-to-worker mapping)
- `verl/trainer/ppo/ray_trainer.py:1281–1700` — `fit()` loop
- `verl/trainer/ppo/core_algos.py:88–830` — advantage estimators

**fit() canonical order to write out**:
```
for epoch, for batch:
  gen_batch = async_rollout_manager.generate_sequences(batch)   # M4 territory
  reward    = _compute_reward_colocate(batch)                   # reward_manager
  old_logp  = _compute_old_log_prob(batch)                      # actor forward, no grad
  compute_advantage(batch, adv_fn)                              # core_algos dispatch
  apply_kl_penalty(batch, kl_ctrl)                              # optional
  actor_metrics  = actor_wg.step(batch)                         # PPO update
  critic_metrics = critic_wg.step(batch)                        # if use_critic
```

**Advantage estimators to read (skip the rest)**:
- `core_algos.py:215–265` — GAE (standard, value-based)
- `core_algos.py:267–330` — GRPO (outcome-only, most deployed)
- `core_algos.py:587–690` — RLOO (leave-one-out, variance-reduced)

**Notes deliverable**: `03-ppo-loop.md` — loop pseudocode, advantage estimator
comparison table, `AdaptiveKLController` behavior.

**Skip**: REMAX, GRPO_VECTORIZED, OPTIMAL_TOKEN_BASELINE and other variants.

---

## M4 — Rollout Engine Integration **[critical]** (~5 h)

**Focus files**:
- `verl/workers/rollout/base.py` — `BaseRollout` ABC, `_ROLLOUT_REGISTRY` (L83–88)
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — `ServerAdapter` (client)
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py:80–1031` — `vLLMHttpServer` Ray actor
- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — chunked weight sync
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py` — SGLang `ServerAdapter`
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py:59–611` — SGLang HTTP server
- `verl/workers/rollout/replica.py` — token buffering / batching

**Call chain to trace**:
```
RayPPOTrainer.fit()
  → async_rollout_manager.generate_sequences(batch)           [agent_loop]
    → ActorRolloutRefWorker.generate_sequences(chunk)         [engine_workers.py]
      → ServerAdapter.generate_sequences(prompts)             [vllm_rollout.py]
        → HTTP/ZMQ → vLLMHttpServer.generate()                [vllm_async_server.py]
          → vLLM AsyncLLM engine
      ← DataProto{responses, log_probs, routed_experts}
```

**Notes deliverable**: `04-rollout-integration.md` — vLLM vs SGLang side-by-side
(adapter pattern, weight sync, MoE `routed_experts` handling, sleep/wake hooks).

**Skip**: `trtllm_rollout` (experimental), `hf_rollout` / `naive` rollout (debug-only).

---

## M5 — Memory Orchestration: Sleep / Wake / Offload **[critical]** (~4 h)

**Focus files**:
- `verl/workers/engine_workers.py:663–727` — `update_weights()` full sleep/wake cycle
- `verl/workers/rollout/base.py:44–69` — `resume(tags)`, `release()`, `update_weights()` contracts
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:51–96` — sleep_level=1 (KV only) vs sleep_level=2 (weights+KV)
- `verl/utils/memory_utils.py` (293 LOC) — `aggressive_empty_cache`, `MemorySnapshotSampler`
- `verl/utils/fsdp_utils.py` — `load_fsdp_model_to_gpu`, `offload_fsdp_model_to_cpu`, `offload_fsdp_optimizer`
- `verl/utils/megatron_utils.py` — `per_tensor_generator()` (named tensor stream for weight sync)
- `verl/workers/sharding_manager/fsdp_ulysses.py` (73 LOC) — FSDP ↔ Ulysses SP resharding

**Timing sequence to internalize and diagram**:
```
PHASE A (training):       actor weights on GPU | KV released    | optimizer on GPU
PHASE B (before rollout): offload optimizer    | offload actor  | resume rollout weights
PHASE C (rollout):        rollout weights+KV on GPU             | actor weights on CPU
PHASE D (after rollout):  release rollout KV   | resume actor   | resume optimizer
```

**Notes deliverable**: `05-memory-orchestration.md` — timing diagram,
sleep_level matrix (level 1 vs 2, when to pick each), MoE `routed_experts`
replay path (the recent `[-len(output.token_ids):]` slicing fix around commit
bcb63864), sharding_manager data resharding across FSDP+Ulysses.

**Skip**: NPU-specific memory paths unless directly relevant.

---

## M6 — FSDP / Engine Worker Internals **[core]** (~4 h)

**Focus files**:
- `verl/workers/engine_workers.py:75–380` — `TrainingWorker` (modern path)
- `verl/workers/engine_workers.py:436–739` — new `ActorRolloutRefWorker` (colocate mode)
- `verl/workers/fsdp_workers.py:146–1000` — legacy `ActorRolloutRefWorker(FSDP)` (reference, more complete)
- `verl/workers/actor/dp_actor.py` — `DataParallelPPOActor` (PPO loss computation)
- `verl/workers/utils/` — common losses, padding

**Notes deliverable**: `06-workers.md` — ActorRolloutRefWorker method map,
backend registry pattern (FSDP/Megatron/TorchTitan/VeOmni), loss computation
trace, rationale for migration from legacy to engine-based workers.

**Skip**: `verl/workers/megatron_workers.py` (1316 LOC, deprecated path).

---

## M7 — Async & Streaming Mode **[core]** (~4 h)

**Focus files**:
- `verl/trainer/main_ppo_sync.py:24–430` — TransferQueue-based zero-copy streaming
- `verl/experimental/agent_loop/` — `AgentLoopManager` async rollout streaming
- `verl/experimental/fully_async_policy/fully_async_trainer.py` — fully-async PPO
- `verl/experimental/reward_loop/` — streaming reward computation
- `verl/experimental/separation/ray_trainer.py` — fully-separate actor/critic/reward

**Notes deliverable**: `07-async-modes.md` — three modes (sync / async /
fully-async) side-by-side, GPU utilization profile sketch for each,
TransferQueue design, where epoch/batch synchronization is removed and what
that costs.

**Skip**: `teacher_loop` unless distillation becomes a direct focus.

---

## M8 — Recipes & Training Variants **[optional]** (~2 h)

Read one recipe end-to-end to learn the extension pattern, then skim titles of
the rest.

**Focus files**:
- `recipe/sppo/main_sppo.py` — simplest variant
- `recipe/dapo/main_dapo.py` — decoupled actor-policy
- `recipe/flowrl/main_flowrl.py` — flow-based RL (2025–2026 addition)

**Notes deliverable**: `08-recipes.md` — extension pattern (trainer + config
override structure). Not exhaustive per-recipe coverage.

**Skip**: GVPO, spin, rep_exp, prime, gkd unless a specific question drags
them in.

---

## M9 — Megatron Backend & Distillation **[optional]** (~3 h, skip unless needed)

- `verl/workers/megatron_workers.py` — deprecated for v0.8.0
- `verl/experimental/teacher_loop/` — streaming teacher for distillation

---

## Progress tracker

- [ ] M1 — Data Protocol & Entry Point
- [ ] M2 — Single Controller & Ray Orchestration
- [ ] M3 — PPO Training Loop
- [ ] M4 — Rollout Engine Integration
- [ ] M5 — Memory Orchestration
- [ ] M6 — FSDP / Engine Worker Internals
- [ ] M7 — Async & Streaming Mode
- [ ] M8 — Recipes & Training Variants
- [ ] M9 — Megatron Backend & Distillation

---

## Critical files (quick-reference index)

| File | Milestone |
|---|---|
| `verl/protocol.py` | M1 |
| `verl/trainer/main_ppo.py` | M1 |
| `verl/single_controller/base/worker.py` | M2 |
| `verl/single_controller/base/worker_group.py` | M2 |
| `verl/single_controller/base/decorator.py` | M2 |
| `verl/single_controller/ray/base.py` | M2 |
| `verl/trainer/ppo/ray_trainer.py` | M3 |
| `verl/trainer/ppo/core_algos.py` | M3 |
| `verl/workers/rollout/base.py` | M4 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | M4 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | M4 |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | M4 |
| `verl/workers/engine_workers.py` | M5, M6 |
| `verl/utils/memory_utils.py` | M5 |
| `verl/utils/fsdp_utils.py` | M5 |
| `verl/workers/sharding_manager/fsdp_ulysses.py` | M5 |
| `verl/workers/fsdp_workers.py` | M6 |
| `verl/trainer/main_ppo_sync.py` | M7 |
| `verl/experimental/fully_async_policy/` | M7 |
