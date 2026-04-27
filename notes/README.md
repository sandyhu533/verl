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

## M1 — Data Protocol & Entry Point **[critical]** (~2.5 h)

`DataProto` is veRL's universal data container; the `main_ppo.py` entry chain
is how a shell command becomes a running Ray cluster. Read M1 first — later
milestones assume you already think in DataProto terms.

Detailed reading plan, function map, and line-by-line annotations: **[M1.md](M1.md)**.

**Notes deliverable**: `01-data-protocol.md` — field taxonomy, chunk/concat/
union/repeat invariants, written after M1 is done.

**Acceptance — you've understood M1 when you can:**
- Explain the three-field model (`batch` / `non_tensor_batch` / `meta_info`) and why veRL didn't just use a single TensorDict or a plain dict.
- Predict what happens if you call `.chunk(N)` on a DataProto whose batch size isn't divisible by N — and why `auto_padding` exists.
- Explain why `union()` enforces strict equality on `meta_info` instead of last-write-wins.
- Trace the path from `python3 -m verl.trainer.main_ppo` to a running Ray cluster (Hydra → `run_ppo` → `TaskRunner` Ray actor → `init_workers` → `fit`).
- Justify why `TaskRunner` itself is a Ray actor instead of running on the driver.
- Explain `repeat(interleave=True)` semantics for GRPO and what would break if you used `interleave=False`.
- (M1.md has the full 8-question self-test.)

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

**Acceptance — you've understood M2 when you can:**
- Draw the dispatch/collect flow for `actor_rollout_wg.generate_sequences(batch)`: who chunks, who fans out, who merges, on which process.
- List at least 4 `Dispatch` modes and explain when each one fires (`DP_COMPUTE_PROTO` vs `ONE_TO_ALL` vs `MEGATRON_COMPUTE_PROTO` vs `DIRECT`).
- Explain `_bind_worker_method`: how a method decorated with `@register` on `Worker` becomes a callable on `WorkerGroup` with auto-dispatch.
- Justify placement-group `STRICT_PACK` for actor+rollout colocation — what breaks if you use `SPREAD`.
- Explain `FusedWorker` string-based dispatch: what problem it solves (multiple roles in one Ray actor for memory sharing) and what's fragile about it.
- Predict the failure mode when worker `world_size` and DataProto batch size mismatch under `DP_COMPUTE_PROTO`.
- Articulate why veRL is "single controller, multi worker" (HybridFlow §3) rather than pure SPMD like Megatron — what flexibility this buys at what cost.

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

**Acceptance — you've understood M3 when you can:**
- Recite the `fit()` per-step skeleton (rollout → reward → old_logp → advantage → KL → actor.step → critic.step) without looking, and name which DataProto fields each stage adds.
- Compare GAE vs GRPO vs RLOO in one sentence each: what baseline they use, why GRPO drops the value head, why RLOO is variance-reduced.
- Explain `use_kl_loss=True` vs `algorithm.use_kl_in_reward=True` — why they're mutually exclusive and which one GRPO uses.
- Trace where `old_log_prob` comes from and why it must be computed *before* the actor weight update (importance ratio correctness).
- Explain `AdaptiveKLController`: what signal it adapts on, what would happen if you fixed `kl_coef` instead.
- Identify the synchronization barriers in `fit()` — where does the loop block on `ray.get`, and which of those would `main_ppo_sync.py` (M7) eliminate.

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

**Acceptance — you've understood M4 when you can:**
- Explain why veRL uses an HTTP/ZMQ `ServerAdapter` → Ray-actor `vLLMHttpServer` boundary instead of calling vLLM as an in-process library.
- Compare vLLM vs SGLang adapters: what's identical (BaseRollout contract), what diverges (radix cache vs PagedAttention block manager), and why veRL supports both.
- Describe `update_weights()` end-to-end: how trained actor weights reach the rollout engine without a full model checkpoint roundtrip (bucketed/chunked transfer, why).
- Explain MoE `routed_experts` handling: why rollout must return expert routing info and what the training side does with it.
- Justify the `sleep(level=1)` vs `sleep(level=2)` choice from the rollout's perspective — what state survives each level.
- Predict what breaks if rollout TP size ≠ actor TP size (and explain how `sharding_manager` bridges it).

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

**Acceptance — you've understood M5 when you can:**
- Draw the 4-phase timeline (training / pre-rollout / rollout / post-rollout) and label which tensors live on GPU vs CPU at each phase.
- Explain `sleep_level=1` vs `sleep_level=2` matrix: what each level releases, when you'd pick which (KV cache only vs weights+KV), and the wake-up cost difference.
- Trace `update_weights()` line by line: who initiates, who does FSDP all-gather, how the named tensor stream gets to the rollout actor, when the actor releases its training-side copy.
- Explain why `aggressive_empty_cache` is needed beyond `torch.cuda.empty_cache()` — what fragmentation pattern triggers OOM that the standard call misses.
- Justify offloading the *optimizer* (not just weights) before rollout — what fraction of memory it actually frees for a 7B Adam model.
- Identify the colocation invariant that makes the whole sleep/wake dance worth it (vs separate actor + rollout GPU pools): GPU $ savings, what you give up.

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

**Acceptance — you've understood M6 when you can:**
- Map the backend registry: how a config string (`fsdp` / `megatron` / `torchtitan` / `veomni`) routes to a concrete `TrainingWorker` implementation.
- Trace `actor_wg.step(batch)` from worker entry to the PPO loss tensor: forward → log-prob → ratio → clip → loss → backward → optimizer step.
- Explain the migration motivation from `fsdp_workers.py` (legacy) to `engine_workers.py` (modern) — what the new abstraction lets veRL do that the old one couldn't.
- Justify gradient checkpointing on/off trade for a 7B Qwen at the configured micro-batch size.
- Explain `use_remove_padding=True`: how variable-length sequences are packed and what it saves vs naive padding.
- Identify where Ulysses sequence parallelism plugs in and what kind of sequence length it's needed for.

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

**Acceptance — you've understood M7 when you can:**
- Sketch the GPU-utilization timeline for sync vs async vs fully-async mode and show where the bubbles disappear.
- Explain TransferQueue: what zero-copy property it guarantees, who is producer/consumer, where it sits between rollout and training.
- Identify the synchronization barriers `main_ppo_sync.py` removes vs the ones it must keep (gradient sync, KL ref alignment).
- Explain `AgentLoopManager`: how it streams partial rollouts back to training before all sequences finish.
- Articulate the staleness trade-off in fully-async PPO — how many policy versions of drift are tolerated, and what corrects for it (importance ratio, KL).
- Predict which workload shape (long-tail responses vs uniform) benefits most from async mode and why.

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

**Acceptance — you've understood M8 when you can:**
- Describe the recipe extension pattern in one paragraph: which base classes you subclass, which config blocks you override, what stays untouched.
- Explain why DAPO needs a decoupled actor-policy and what concrete code change makes that possible (vs vanilla PPO).
- Predict where you'd add a new `MyRLAlgo` recipe — list the 3-5 files you'd touch.

---

## M9 — Megatron Backend & Distillation **[optional]** (~3 h, skip unless needed)

- `verl/workers/megatron_workers.py` — deprecated for v0.8.0
- `verl/experimental/teacher_loop/` — streaming teacher for distillation

**Acceptance — you've understood M9 when you can:**
- Explain why the Megatron path was deprecated in favor of the engine-based abstraction (maintenance burden, TP/PP coupling).
- Describe `teacher_loop` at the contract level: what the teacher emits, what the student consumes, where logits are matched.
- Justify whether distillation belongs in this repo at all, or as a downstream consumer.

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
