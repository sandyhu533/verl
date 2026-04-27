# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, is_npu_available, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import diffusion_loss, ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """Engine-agnostic trainable Ray worker shared by actor, critic, and ref-policy roles.

    What

      - Tinker-like RayWorkerGroup facade (https://thinkingmachines.ai/tinker/) exposing
        coarse-grained RPCs (train_batch, infer_batch, save/load_checkpoint, to) to the
        single-controller trainer.
      - Holds a single `self.engine: BaseEngine` constructed via `EngineRegistry.new(...)`;
        all forward/backward/step/offload work is delegated there, so this class is pure
        orchestration and metric plumbing (DP all-reduce of loss, all-gather of metrics,
        MFU via FlopsCounter).

    Lifecycle

      - `__init__`: initialize Ray PG, NUMA pin, resolve engine+optim configs (optionally
        via `auto_select_engine_optim_fn`), instantiate BaseEngine, register train-mesh
        dispatch info, build FlopsCounter.
      - `reset()`: lazy `engine.initialize()` (allocate params/optimizer/grad buffers).
      - `set_loss_fn()`: inject ppo/diffusion/distillation loss closure from parent.
      - Steady state: `train_mini_batch` -> chunks into `train_batch` micro-steps under
        `engine.train_mode()`; `infer_batch` under `engine.eval_mode()` with optional
        `disable_adapter()` for LoRA ref-like paths.

    Called by

      - `ActorRolloutRefWorker.init_model` constructs one TrainingWorker per sub-role
        (actor, ref) locally — not via Ray — so methods run in-process on the outer
        hybrid worker's GPUs.
      - Future CriticWorker / standalone ActorWorker paths reuse this same class.

    Call graph (THIS ENTITY)

      - train_mini_batch -> make_iterator -> train_batch -> engine.train_batch(loss_fn)
        -> _postprocess_output (DP all-reduce loss, allgather_dict_into_dict metrics,
        FlopsCounter.estimate_flops -> mfu).
      - infer_batch -> engine.infer_batch -> _postprocess_output (forward_only mfu /= 3).
      - save_checkpoint / load_checkpoint / to -> engine.*.

    Branches

      - `engine_config.strategy` in {"fsdp", "fsdp2", "megatron", "torchtitan"} selects
        the BaseEngine subclass via EngineRegistry; this class is unaware of which.
      - NPU path sets `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (torch_npu lacks
        `set_expandable_segments`).
      - `is_mp_src_rank_with_outputs()` gates whether non-TP-src ranks materialize
        `final_output` (TP/PP ranks return None to save serialization).

    Why

      - Replaces per-backend duplication in legacy `fsdp_workers.py`: actor/critic/ref
        RPC surface is written once here; swapping FSDP<->Megatron<->TorchTitan is a
        config flip, matching HybridFlow's engine-abstraction goal.
      - Keeps scheduling (single controller) decoupled from parallelism strategy so the
        trainer's `.update_actor(...)` stays identical across backends.
    """

    def __init__(self, config: TrainingWorkerConfig):
        """Wire up config, registries, and profiler — but do NOT build the engine yet.

        What

          - Stash the `TrainingWorkerConfig` sub-fields (model / engine / optimizer /
            checkpoint) onto `self`, initialize Ray's global process group, set NUMA
            affinity, and construct `self.engine = EngineRegistry.new(...)` — the
            backend-specific BaseEngine (FSDP / Megatron / TorchTitan / VeOmni / ...).
          - Register this worker's dispatch/collect info under mesh_name="train" so
            the single controller can route `DP_COMPUTE_PROTO` RPCs onto its DP ranks.

        Lifecycle

          - Called once when this object is instantiated (either as a plain Python
            object inside `ActorRolloutRefWorker.init_model`, or as a Ray actor for
            the SFT trainer / standalone critic path).
          - `self.engine` is CONSTRUCTED here but NOT initialized; param / optimizer
            / grad allocation happens later in `reset()`.
          - `self.loss_fn` starts as None; the parent injects it via `set_loss_fn()`.

        Called by

          - `ActorRolloutRefWorker.init_model` builds `self.actor = TrainingWorker(...)`
            and `self.ref = TrainingWorker(...)` in-process.
          - `main_ppo.py` aliases `CriticWorker = TrainingWorker` under the
            `use_legacy_worker_impl="disable"` path.
          - `sft_trainer[_ray].py` uses it standalone as `self.training_client`.

        Call graph (THIS ENTITY)

          - __init__ -> initialize_global_process_group_ray -> set_numa_affinity
            -> EngineRegistry.new -> _register_dispatch_collect_info(mesh="train")
            -> FlopsCounter(hf_config).

        Branches

          - `engine_config is None` + `auto_select_engine_optim_fn` set: resolve
            (engine_config, optimizer_config) from the model config at runtime.
          - NPU: set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (torch_npu
            lacks `set_expandable_segments`).
          - Diffusion models: `flops_counter = None` (MFU not yet supported).

        Why

          - Keep constructor cheap and deterministic so Ray actor placement + config
            validation are separate from heavy engine materialization. `reset()` is
            where shards actually allocate, which the trainer can sequence across
            actor/ref/critic to control peak memory.
        """
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        # TODO(jhz): Switch to `set_expandable_segments` when the torch_npu library
        # supports `torch.npu.memory._set_allocator_settings`
        if is_npu_available:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            # for Diffusion models, FlopsCounter is not supported yet.
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual offload/reload of model / optimizer / grad between device and CPU.

        What

          - Thin control-plane RPC that forwards to `self.engine.to(...)`; the engine
            decides how to stream FSDP flat-params, Megatron bucketed grads, or
            TorchTitan sharded optimizer state in/out of device memory.

        Lifecycle

          - Invoked by the trainer (or by `ActorRolloutRefWorker.to`) around phases
            where VRAM must be freed — e.g. before rollout step in colocated mode,
            or after `update_weights` to offload the actor while rollout runs.

        Called by

          - `ActorRolloutRefWorker.to` (same file) and higher-level offload hooks
            in the ray trainer; `dispatch_mode=ONE_TO_ALL` so every rank executes.

        Call graph (THIS ENTITY)

          - to -> engine.to(device, model, optimizer, grad).

        Branches

          - `device == "device"` is resolved to the concrete device name (cuda/npu)
            via `get_device_name()` before forwarding.

        Why

          - HybridFlow's 3D-HybridEngine co-locates train+rollout, so explicit
            offload toggles (rather than implicit auto-offload) are required to
            shape peak-memory around rollout's KV cache.
        """
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        """Install the loss callable consumed by train_batch / train_mini_batch.

        What

          - Stash a loss closure (e.g. `partial(ppo_loss, config=actor_config)`,
            `diffusion_loss`, or `distillation_ppo_loss`) onto `self.loss_fn`.
          - `train_batch` passes it to `engine.train_batch(data, loss_function=...)`;
            the engine calls it per micro-batch with model_outputs + labels.

        Lifecycle

          - Called once, immediately after `reset()`, by the parent
            `ActorRolloutRefWorker.init_model` (`self.actor.set_loss_fn(self.loss_fn)`).
          - Ref workers never get a loss_fn installed (they only run `infer_batch`
            with `compute_loss=False`), which matches the `loss_fn is None` assert
            inside `train_batch`.

        Called by

          - `ActorRolloutRefWorker.set_loss_fn` (same file) on the actor only.

        Call graph (THIS ENTITY)

          - set_loss_fn -> attribute assignment; consumed later by train_batch.

        Branches

          - None; pure setter.

        Why

          - Keeps `TrainingWorker` loss-agnostic so the same class serves PPO,
            SFT, diffusion, and distillation without subclassing.
        """
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """Materialize the BaseEngine: allocate params / optimizer / grad buffers.

        What

          - Calls `self.engine.initialize()`, which is where the backend actually
            builds the model: FSDP flat-param wrapping, Megatron parallel state +
            distributed optimizer, TorchTitan parallelize_module passes, etc.
          - Separates cheap construction (in `__init__`) from expensive allocation
            so the trainer can sequence actor/ref/critic builds to cap peak memory.

        Lifecycle

          - Called exactly once per TrainingWorker by `ActorRolloutRefWorker.init_model`
            right after `TrainingWorker(...)` construction (`self.ref.reset()` then
            `self.actor.reset()`), before any data-plane RPC.
          - After this call, the engine is ready for `train_batch` / `infer_batch`
            and `get_dispatch_collect()` can be queried by the parent to re-register
            the "actor" / "ref" dispatch mesh aliases.

        Called by

          - `ActorRolloutRefWorker.init_model` at lines `self.ref.reset()` /
            `self.actor.reset()`; `dispatch_mode=ONE_TO_ALL`.

        Call graph (THIS ENTITY)

          - reset -> engine.initialize (backend-specific param / optim allocation).

        Branches

          - Backend choice (`fsdp` / `fsdp2` / `megatron` / `torchtitan` / ...) was
            fixed in `__init__` via EngineRegistry; this method is oblivious.

        Why

          - Lazy initialize hook matches HybridFlow's "build engine when trainer
            schedules it" model and leaves room for later `reload ckpt + reset
            states` behavior without changing the RPC surface.
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """Private helper: DP-reduce loss, DP-gather metrics, compute MFU.

        What

          - Pop `loss` and `metrics` from the engine's raw output dict; average
            `loss` across the DP group via `all_reduce(AVG)`; all-gather
            non-reduced metrics with `allgather_dict_into_dict`.
          - Compute MFU via `FlopsCounter.estimate_flops(global_token_num, dt)`
            divided by world_size (and /3 for forward-only since backward ~ 2x fwd).
          - Package the surviving `model_output` TensorDict together with
            `{"metrics": final_metrics}` for return.

        Lifecycle

          - Called at the tail of `train_batch` and `infer_batch`, only on ranks
            where `engine.is_mp_src_rank_with_outputs()` is True (TP/PP non-src
            ranks return None earlier to save serialization).

        Called by

          - `TrainingWorker.train_batch` (forward_only=False).
          - `TrainingWorker.infer_batch` (forward_only=True, MFU /= 3).

        Call graph (THIS ENTITY)

          - _postprocess_output -> all_reduce(loss) -> allgather_dict_into_dict
            -> FlopsCounter.estimate_flops -> tu.get_tensordict.

        Branches

          - `dp_group is None`: skip collectives (single-DP or TP-only setup).
          - `flops_counter is None` (diffusion path) or `global_token_num is None`:
            skip MFU.
          - `mtp_losses*` keys: flatten list-of-single-element sublists and average.

        Why

          - Centralizes the metric plumbing so both train and infer paths emit
            identical-shape dicts into the trainer, which simplifies downstream
            aggregation in `ray_trainer`.
        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """PPO-style multi-epoch update: split a batch into mini-batches, step per mini.

        What

          - Entry point for a full PPO update over a global batch. Implements the
            classic Schulman-2017 PPO schedule at two outer levels, leaving the
            innermost micro-batch loop to `engine.train_batch`:

              for epoch in range(ppo_epochs):            # handled here via `epochs`
                for mini_batch in split(batch, mini_bs): # this method
                  for micro_batch in split(mini, micro_bs):  # inside engine.train_batch
                    loss.backward()
                  optimizer.step()  # at mini-batch boundary -> update_lr_scheduler

          - optimizer.step happens ONCE per mini-batch (not per micro) — the flag
            `update_lr_scheduler=(batch_idx == total_num_iterations - 1)` triggers
            the lr scheduler tick at the end of the whole train_mini_batch call,
            while the engine's internal micro-batch loop handles grad accumulation.

        Lifecycle

          - Called once per PPO optimization step from the trainer, after advantage
            computation and log-prob rollout. Runs under `engine.train_mode()`
            context so activation ckpt / grad-enabled / FSDP train mode are set.
          - Emits aggregated metrics (loss, grad_norm, lr, kl, etc.) flattened
            across DP + micro-batches via `_postprocess_output` inside `train_batch`,
            then one more outer aggregation here via `Metric.aggregate_dp`.

        Called by

          - `ActorRolloutRefWorker.update_actor` -> `self.actor.train_mini_batch(data)`.
          - `ray_trainer.RayPPOTrainer` / `main_ppo_sync` -> `critic_wg.train_mini_batch(batch)`
            (when `CriticWorker = TrainingWorker` alias is active).
          - `tests/models/test_diffusers_fsdp_engine.py` for the diffusion PPO path.
          - `dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train")`:
            TrainingWorker owns a "train" mesh, and `ActorRolloutRefWorker.init_model`
            re-registers it as "actor" / "ref" mesh via `set_dispatch_collect(...)`
            so the single controller can route DP chunks correctly through the
            outer hybrid worker.

        Call graph (THIS ENTITY)

          - train_mini_batch
            -> tu.make_iterator (DP-rank-seeded shuffle)
            -> engine.train_mode() context
            -> loop: train_batch(mini_batch_td)
                  -> engine.train_batch(loss_fn)  # inner micro-batch loop + step
                  -> _postprocess_output
            -> Metric.aggregate_dp across micro-batch outputs
            -> return TensorDict({"metrics": ...}) on src rank, None elsewhere.

        Branches

          - `mini_batch_size` XOR `num_mini_batch`: one must be provided; the other
            is derived from the per-DP batch size.
          - `"input_ids" in mini_batch_td`: language-model path all-gathers
            `global_token_num` across DP for MFU; diffusion path skips this.
          - `is_mp_src_rank_with_outputs()`: non-src ranks return None (save serde).
          - `disable_auto_offload` is forced True for inner `train_batch` calls so
            the engine doesn't offload between micro-batches within one mini.

        Why

          - Keeping the mini-batch loop here (rather than inside the engine) lets
            the single controller observe per-mini-batch metrics for logging /
            early-stop decisions, while still amortizing optimizer step + lr
            scheduler across micro-batches, which is the performance-correct
            PPO recipe (InstructGPT §C.2).
        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        """Single forward+backward train step — no outer mini-batch loop.

        What

          - Run one training pass on `data`: forward under `engine.train_mode()`,
            compute loss via the installed `self.loss_fn`, backward + optimizer
            step handled by the engine's internal micro-batch loop, then
            post-process metrics (DP reduce loss, MFU, optional lr update).
          - Used both as the inner step of `train_mini_batch` and as a direct
            public RPC for the SFT trainer (which does its own dataloader loop).

        Lifecycle

          - When called from `train_mini_batch`, `disable_auto_offload=True` is
            already set so the engine stays resident across mini-batches.
          - `update_lr_scheduler` is True only at the last micro-batch iteration
            of the enclosing mini-batch; the engine ticks the scheduler then.

        Called by

          - `TrainingWorker.train_mini_batch` (internal).
          - `sft_trainer[_ray].py` -> `self.training_client.train_batch(data)`.
          - `tests/models/test_engine.py` for PPO engine smoke tests.
          - `dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train")`.

        Call graph (THIS ENTITY)

          - train_batch
            -> engine.train_mode() context
            -> engine.train_batch(data, loss_function=self.loss_fn)  # micro loop
            -> engine.lr_scheduler_step (if update_lr_scheduler)
            -> _postprocess_output (src rank only).

        Branches

          - `engine_config.forward_only`: asserted False (ref workers must not
            call train_batch).
          - `is_mp_src_rank_with_outputs()`: non-src returns None.
          - Default engineering keys (`use_remove_padding`, `use_dynamic_bsz`,
            `max_token_len_per_gpu`, `micro_batch_size_per_gpu`, `use_fused_kernels`)
            injected if not already in `data`.

        Why

          - Keeps the raw "one training step" primitive exposed so non-PPO trainers
            (SFT, unit tests) can reuse it without paying for the mini-batch loop,
            while PPO composes it through `train_mini_batch`.
        """
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        """No-grad forward: produce log-probs / values / (optional) eval loss.

        What

          - Forward-only pass under `engine.eval_mode()` (no grads, no activation
            retention) used to compute per-token log-probs for the policy (actor)
            and reference (ref) models, and values for the critic.
          - Because activations aren't kept, the engine can safely run with larger
            `infer_max_token_len_per_gpu` / `infer_micro_batch_size_per_gpu` than
            the training path — the `default_keys` dict uses the `infer_*` fields.

        Lifecycle

          - Called during the rollout evaluation phase of each PPO iteration,
            after rollout generation and before advantage / KL computation.
          - Emits metrics with `forward_only=True` so `_postprocess_output`
            divides MFU by 3 to approximate fwd-only vs fwd+bwd ratio.

        Called by

          - `ActorRolloutRefWorker.compute_log_prob` -> `self.actor.infer_batch(data)`.
          - `ActorRolloutRefWorker.compute_ref_log_prob` -> `self.ref.infer_batch(data)`.
          - `ray_trainer` / `main_ppo_sync` -> `critic_wg.infer_batch(batch)` for
            `compute_values` when `CriticWorker = TrainingWorker`.
          - `sft_trainer[_ray].py` -> validation pass.
          - `dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train")`.

        Call graph (THIS ENTITY)

          - infer_batch
            -> engine.eval_mode() context
            -> [optional engine.disable_adapter() if no_lora_adapter]
            -> engine.infer_batch(data, loss_function=...)
            -> _postprocess_output(forward_only=True).

        Branches

          - `compute_loss` (True by default): pass `self.loss_fn` so SFT validation
            can evaluate loss in eval mode; ref-log-prob paths set it False.
          - `no_lora_adapter`: open `engine.disable_adapter()` context so that a
            LoRA-adapted actor can produce "base-model" log-probs for the ref
            computation (avoids maintaining a separate ref model with LoRA).
          - `is_mp_src_rank_with_outputs()`: non-src returns None.

        Why

          - Mirrors InstructGPT §3's rollout-eval log-prob pattern; keeping it
            under engine.eval_mode() with no_grad is what lets rollout + log-prob
            + ref-log-prob all fit on the same GPUs in 3D-HybridEngine colocation.
        """
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Delegate to `engine.save_checkpoint` — each rank writes its own shard.

        What

          - Per-rank sharded checkpoint write: FSDP writes flat-param shards,
            Megatron writes TP/PP-partitioned shards, TorchTitan writes DCP
            sharded tensors. `max_ckpt_to_keep` rotates older checkpoint dirs.

        Lifecycle

          - Called from `ActorRolloutRefWorker.save_checkpoint` (actor role only)
            at the trainer's checkpoint cadence; `dispatch_mode=ONE_TO_ALL`.

        Called by

          - `ActorRolloutRefWorker.save_checkpoint` (same file), which asserts
            `"actor" in self.role` before delegating.

        Call graph (THIS ENTITY)

          - save_checkpoint -> engine.save_checkpoint(local, hdfs, step, keep).

        Branches

          - None at this layer; backend-specific logic lives in the engine.

        Why

          - Sharded write avoids rank-0 gather, which is prohibitive at
            tens-of-B-param scale. Matches Megatron / TorchTitan / FSDP DCP
            conventions so resharding-on-load is feasible.
        """
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """Delegate to `engine.load_checkpoint` — each rank reads its own shard.

        What

          - Per-rank sharded checkpoint read matching the layout written by
            `save_checkpoint`. `del_local_after_load=True` frees the local
            checkpoint dir after the shard is materialized into the engine.

        Lifecycle

          - Called from `ActorRolloutRefWorker.load_checkpoint` (actor role only)
            at trainer startup / resume; `dispatch_mode=ONE_TO_ALL`.

        Called by

          - `ActorRolloutRefWorker.load_checkpoint` (same file), which asserts
            `"actor" in self.role` before delegating.

        Call graph (THIS ENTITY)

          - load_checkpoint -> engine.load_checkpoint(local, hdfs, del_local).

        Branches

          - None at this layer; resharding / mapping lives in the engine.

        Why

          - Mirrors `save_checkpoint`'s sharded layout so resume cost is
            O(shard_size / rank) rather than rank-0-bottlenecked.
        """
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Unified engine-agnostic hybrid worker: actor + rollout + ref_policy in one Ray actor.

    What

      - The new `use_legacy_worker_impl=False` replacement for
        `verl.workers.fsdp_workers.ActorRolloutRefWorker`. Same 3-in-1 co-location as
        HybridFlow 3D-HybridEngine, but built on `TrainingWorker` + `BaseEngine` so
        the training backend (FSDP / Megatron / TorchTitan) is chosen at config time
        via `config.actor.strategy` instead of being hard-wired into this class.
      - Owns up to three sub-components:
          - `self.actor: TrainingWorker`   (trainable policy, `ppo_loss` / `diffusion_loss`
            / `distillation_ppo_loss`),
          - `self.ref:   TrainingWorker`   (frozen forward-only reference, MTP disabled),
          - `self.rollout: BaseRollout`    (vLLM / SGLang / TRT-LLM inference engine).

    Lifecycle

      - `__init__`: parse `role` bitfield (_is_actor/_is_rollout/_is_ref), build profiler,
        detect Megatron router_replay to toggle `enable_routing_replay` decorator.
      - `init_model`: build ref -> actor -> rollout -> checkpoint engine in that order,
        registering per-sub-role dispatch meshes ("ref", "actor") with the single
        controller; ends with `aggressive_empty_cache` so colocated vLLM sees free VRAM.
      - Steady state: `compute_ref_log_prob`, `compute_log_prob`, `update_actor`,
        `update_weights` (trainer->rollout param sync), `save/load_checkpoint`.

    Called by

      - `verl.trainer.ppo.ray_trainer.RayPPOTrainer` via its RayWorkerGroup when
        `config.actor.use_legacy_worker_impl` is False (selected in main_ppo's worker
        resolution path; legacy users still land in `fsdp_workers.py`).

    Call graph (THIS ENTITY)

      - update_actor -> actor.train_mini_batch -> (TrainingWorker call graph).
      - compute_log_prob / compute_ref_log_prob -> {actor,ref}.infer_batch.
      - update_weights -> actor.engine.get_per_tensor_param -> rollout.update_weights
        (sync path) OR checkpoint_engine.send_weights (async/disaggregated path).

    Branches

      - `config.actor.strategy`: "fsdp"/"fsdp2"/"megatron"/"torchtitan" -> propagated
        through `TrainingWorkerConfig` into each `TrainingWorker.engine`.
      - Megatron-only: `enable_routing_replay` gates MoE router replay via
        `_with_routing_replay_flag` on compute/update RPCs.
      - `checkpoint_engine.backend != "naive"`: async/disaggregated rollout -> weights
        shipped via CheckpointEngine; else colocated path resumes rollout, NCCL-syncs
        params, re-offloads actor, resumes KV cache.
      - LoRA: `peft_merge=True` merges adapters (single update); else two-phase base+
        adapter sync with `base_sync_done` guard.

    Why

      - HybridFlow §4 "3D-HybridEngine" argues co-locating train + rollout on the same
        GPUs wins on memory/bandwidth but needs a backend-abstract trainer. The legacy
        class hard-coded FSDP. This unified class delegates to `TrainingWorker.engine`,
        so the same RPC surface works for Megatron 3D-parallel training or TorchTitan
        without changing the trainer's control flow.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        """Parse the role bitfield and set up the profiler; defer all model/engine build to init_model.

        What

          - Decode `role` string ("actor" | "rollout" | "ref" | "actor_rollout" |
            "actor_rollout_ref") into three boolean flags `_is_actor`, `_is_rollout`,
            `_is_ref` that gate which sub-components get constructed later.
          - Leave `self.actor`, `self.ref`, `self.rollout` as None placeholders; they
            are materialized lazily in `init_model` so Ray actor placement is cheap
            and deterministic.
          - Build the DistProfiler by picking the profiler sub-config that matches
            the dominant sub-role (actor > rollout > ref).
          - Detect whether Megatron MoE router-replay is enabled, arming the
            `_with_routing_replay_flag` decorator on compute/update paths.

        Lifecycle

          - Called once when Ray instantiates this actor from RayWorkerGroup.
          - Pairs with `init_model`, which is the real heavy lifter.

        Called by

          - `RayPPOTrainer.init_workers` -> `RayWorkerGroup(..., cls=ActorRolloutRefWorker)`
            -> Ray constructs this actor on each GPU and invokes `__init__`.

        Call graph (THIS ENTITY)

          - __init__ -> Worker.__init__ -> omega_conf_to_dataclass(ProfilerConfig)
            -> DistProfilerExtension.__init__(DistProfiler(...)).

        Branches

          - Profiler source: _is_actor -> config.actor.profiler; elif _is_rollout ->
            config.rollout.profiler; else -> config.ref.profiler.
          - `enable_routing_replay` True only when `config.actor.strategy == "megatron"`
            AND `megatron.router_replay.mode != "disabled"`.

        Why

          - Keeps Ray actor startup cheap; heavy model/engine allocation moves to
            `init_model` so the trainer can sequence ref/actor/rollout construction
            across workers to control peak GPU memory.
        """
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.actor.strategy == "megatron" and self.config.actor.megatron.router_replay.mode != "disabled"
        )

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        """Control-plane delegate: inject the PPO/diffusion/distillation loss closure into the actor TrainingWorker.

        What

          - Thin pass-through to `self.actor.set_loss_fn` so the trainer can swap
            the loss closure at runtime without rebuilding the engine.
          - Only touches the actor sub-worker; ref is forward-only (no loss) and
            rollout is pure inference.

        Lifecycle

          - Typically unused after `init_model` (which wires loss_fn internally),
            but exposed so experiments can override loss mid-run.

        Called by

          - Rare: custom trainer overrides that want a non-default loss closure.
          - `RayPPOTrainer` does not call this directly in the standard path;
            init_model already picks {ppo, diffusion, distillation} by config.

        Call graph (THIS ENTITY)

          - trainer -> actor_rollout_wg.set_loss_fn (RayWorkerGroup dispatch)
            -> ONE_TO_ALL broadcast -> THIS METHOD -> self.actor.set_loss_fn
            -> TrainingWorker stores closure on self.loss_fn.

        Branches

          - None.

        Why

          - Keeps the control plane (loss choice, offload, save/load) on
            ONE_TO_ALL while the data plane uses DP_COMPUTE_PROTO meshes.
        """
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual offload/reload of the actor sub-component (not ref, not rollout).

        What

          - Move the actor's model params / optimizer state / grad buffers between
            GPU ("device") and CPU by delegating to `self.actor.to`, which forwards
            to `TrainingWorker.engine.to`.
          - Intentionally does NOT touch self.ref (frozen; its own `to` is managed
            separately) and does NOT touch self.rollout (the rollout engine has
            its own sleep/resume API exercised in `update_weights`).

        Lifecycle

          - Called around `update_weights` in the colocated HybridFlow path: after
            weight sync the trainer offloads actor params to CPU so rollout's KV
            cache can reclaim that VRAM before generation.
          - Also called between PPO phases to keep peak memory under the
            train+rollout colocation budget.

        Called by

          - `RayPPOTrainer.fit` around rollout generation / log-prob recompute.

        Call graph (THIS ENTITY)

          - trainer -> actor_rollout_wg.to (RayWorkerGroup dispatch) -> ONE_TO_ALL
            broadcast -> THIS METHOD -> self.actor.to -> engine.to(device, ...).

        Branches

          - `device in {"cpu", "device"}`; "device" resolves to get_device_name()
            inside TrainingWorker.to.

        Why

          - HybridFlow §4 colocation: the same GPUs host both training shards and
            rollout KV cache. Explicit `to("cpu")` is how the controller frees
            VRAM for the inference engine without tearing down the actor.
        """
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Four-phase build of the 3-in-1 hybrid worker: ref -> actor -> rollout -> checkpoint_engine.

        What

          - Phase 1 (ref): if _is_ref, synthesize a forward-only TrainingWorker
            with MTP forced off and infer-side batching keys (log_prob_*) remapped
            onto engine_config. Register dispatch info under mesh_name="ref".
          - Phase 2 (actor): if _is_actor, build a trainable TrainingWorker, pick
            loss_fn (distillation_ppo_loss / diffusion_loss / ppo_loss), inject it
            via `self.actor.set_loss_fn`, register mesh_name="actor".
          - Phase 3 (rollout): if _is_rollout, build the BaseRollout (vLLM /
            SGLang / TRT-LLM) on a (dp, infer_tp, infer_pp) device mesh. NOTE:
            under use_legacy_worker_impl="disable", `generate_sequences` is NOT
            a method on this class — the trainer calls
            `async_rollout_manager.generate_sequences(...)` via RolloutReplica.
            So `self.rollout` here is a CONTROL-plane handle (weight sync /
            sleep / wake in `update_weights`), not the data-generation entry.
          - Phase 4 (checkpoint_engine): if _is_actor, instantiate from
            CheckpointEngineRegistry. Backend "naive" means colocated NCCL sync;
            any other backend means async/disaggregated weight push via
            `checkpoint_engine.send_weights`.
          - Finish with `aggressive_empty_cache(force_sync=True)` so colocated
            vLLM sees freed VRAM via cudaMemGetInfo.

        Lifecycle

          - Called exactly once after __init__, before any PPO step. Allocates
            all shards / KV cache / optimizer state on the GPU.

        Called by

          - `RayPPOTrainer.init_workers` -> actor_rollout_wg.init_model()
            (ONE_TO_ALL broadcast). Legacy users land in `fsdp_workers.py`.

        Call graph (THIS ENTITY)

          - -> TrainingWorker(ref) -> self.ref.reset -> set_dispatch_collect("ref")
          - -> TrainingWorker(actor) -> self.actor.reset -> actor.set_loss_fn
            -> set_dispatch_collect("actor")
          - -> get_rollout_class(name, mode)(...) -> self.rollout
          - -> CheckpointEngineRegistry.new(backend, ...) -> self.checkpoint_engine
          - -> aggressive_empty_cache(force_sync=True).

        Branches

          - Loss fn: distillation_enabled -> distillation_ppo_loss;
            model_type == "diffusion_model" -> diffusion_loss; else -> ppo_loss.
          - Checkpoint backend == "naive" vs async/disaggregated drives the
            update_weights branch later.
          - LoRA: `peft_merge` flag latched from `model_config.lora.merge` here
            gates the two-phase base+adapter sync in update_weights.

        Why

          - Separate ref/actor meshes because they can have different TP/PP/DP
            configs; registering per-role dispatch lets the single controller
            route data-plane RPCs (compute_ref_log_prob vs compute_log_prob) to
            the right shard topology.
          - Build order matters: ref first (smallest, no optimizer), then actor
            (peak training memory), then rollout (takes whatever VRAM is left),
            then checkpoint_engine (pure control plane). This staging keeps peak
            GPU memory predictable during init.
        """
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # phase 1: build ref model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # phase 2: build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            elif model_config.get("model_type", "language_model") == "diffusion_model":
                self.loss_fn = partial(diffusion_loss, config=actor_config)
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # phase 3: build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # phase 4: build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        """Data-plane entry for frozen-ref forward pass; dispatched on mesh_name="ref".

        What

          - No-grad ref-policy forward to produce reference log-probs used in the
            KL term of PPO. Delegates to `self.ref.infer_batch` which runs under
            `engine.eval_mode()` in the ref TrainingWorker.
          - Routed onto mesh_name="ref" (separate from actor's mesh) so the ref
            model can have its own TP/PP/DP sharding independent of the actor.

        Lifecycle

          - Hot PPO path. Called once per rollout batch, before update_actor.

        Called by

          - `RayPPOTrainer.fit` -> actor_rollout_wg.compute_ref_log_prob(batch)
            (make_nd_compute_dataproto_dispatch_fn mesh="ref").

        Call graph (THIS ENTITY)

          - trainer -> DP_COMPUTE_PROTO(ref) -> THIS METHOD
            -> self.ref.infer_batch -> engine.infer_batch (forward-only).

        Branches

          - `_with_routing_replay_flag(enabled=False)`: ref does NOT record MoE
            router decisions (only actor/update paths set enabled=True to replay
            the same expert choices across log-prob recompute and gradient step).
          - Returns None on non-MP-src ranks; CPU-materialize on the src rank.

        Why

          - Thin wrapper, but sits on the hottest PPO path — the real compute
            lives in TrainingWorker.infer_batch. Keeping this method tiny makes
            the trainer-side RPC surface legible.
        """
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Data-plane entry for actor no-grad forward (old-policy log-probs); dispatched on mesh_name="actor".

        What

          - No-grad actor forward to obtain `log_pi_theta_old` over the rollout
            tokens. Needed for PPO's importance-sampling ratio. Delegates to
            `self.actor.infer_batch` under `engine.eval_mode()`.
          - Routed onto mesh_name="actor" (not "ref") so it lands on the actor's
            TP/PP/DP shards.

        Lifecycle

          - Hot PPO path. Called once per rollout batch, paired with
            compute_ref_log_prob, before update_actor.

        Called by

          - `RayPPOTrainer.fit` -> actor_rollout_wg.compute_log_prob(batch).

        Call graph (THIS ENTITY)

          - trainer -> DP_COMPUTE_PROTO(actor) -> THIS METHOD
            -> self.actor.infer_batch -> engine.infer_batch.

        Branches

          - `_with_routing_replay_flag(enabled=True)`: on MoE Megatron, the actor
            RECORDS expert-routing decisions here so update_actor can REPLAY them
            for a consistent gradient. Contrasts with compute_ref (enabled=False).
          - Returns None on non-MP-src ranks; CPU-materialize on the src rank.

        Why

          - Thin wrapper, but sits on the hottest PPO path — real compute lives
            in TrainingWorker.infer_batch. Separate actor-mesh dispatch from ref
            keeps sharding topologies decoupled.
        """
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        """Data-plane entry for the PPO gradient update; dispatched on mesh_name="actor".

        What

          - Run the PPO (or diffusion / distillation) gradient step on the actor
            model by delegating to `self.actor.train_mini_batch`, which chunks
            the batch into mini-batches, calls engine.train_batch under
            `engine.train_mode()` with `self.loss_fn`, and all-reduces metrics.
          - Routed onto mesh_name="actor".

        Lifecycle

          - Hot PPO path. Called after compute_ref_log_prob + compute_log_prob
            produce old-policy log-probs; this is the gradient step.

        Called by

          - `RayPPOTrainer.fit` -> actor_rollout_wg.update_actor(batch).

        Call graph (THIS ENTITY)

          - trainer -> DP_COMPUTE_PROTO(actor) -> THIS METHOD
            -> self.actor.train_mini_batch -> (N micro-batches of) train_batch
            -> engine.train_batch(loss_fn=ppo_loss/...) -> optimizer.step.

        Branches

          - `_with_routing_replay_flag(enabled=True)`: replays the MoE expert
            routing decisions recorded during compute_log_prob so the gradient
            is taken with respect to the SAME experts the old policy used.
          - Returns None on non-MP-src ranks; CPU-materialize metrics on src.

        Why

          - Trainer-facing thin RPC; real training math lives in TrainingWorker.
            The important design decision here is the actor-mesh dispatch +
            routing-replay flag, both of which are backend-invariant.
        """
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """Control-plane delegate: restore actor weights/optimizer from local or HDFS checkpoint.

        What

          - Thin pass-through to `self.actor.load_checkpoint` which delegates to
            `TrainingWorker.engine.load_checkpoint` (backend-specific: FSDP uses
            dcp; Megatron uses distributed-save; TorchTitan uses dcp).
          - Asserts actor role; ref is stateless (frozen from HF weights) and
            rollout has its own weight-sync path via update_weights.

        Lifecycle

          - Called on resume-from-checkpoint at trainer startup, not in the hot
            PPO loop.

        Called by

          - `RayPPOTrainer.fit / init_workers` ->
            actor_rollout_wg.load_checkpoint(path) (ONE_TO_ALL).

        Call graph (THIS ENTITY)

          - trainer -> actor_rollout_wg.load_checkpoint (RayWorkerGroup dispatch)
            -> ONE_TO_ALL broadcast -> THIS METHOD -> self.actor.load_checkpoint
            -> engine.load_checkpoint.

        Branches

          - `del_local_after_load=True` deletes local shards after load to save
            disk.

        Why

          - Single choke point for the trainer's resume path; the backend
            details (sharded vs replicated, format) stay hidden in the engine.
        """
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Control-plane delegate: persist actor weights/optimizer/rng to local or HDFS.

        What

          - Thin pass-through to `self.actor.save_checkpoint` which delegates to
            `TrainingWorker.engine.save_checkpoint`.
          - Actor-only: ref has no trainable state; rollout's weights live in
            the inference engine and are re-derived from actor on next
            update_weights, so they don't need independent checkpoints.

        Lifecycle

          - Called at configured `save_freq` intervals from the PPO loop, plus
            on final iteration.

        Called by

          - `RayPPOTrainer.fit` ->
            actor_rollout_wg.save_checkpoint(path, step) (ONE_TO_ALL).

        Call graph (THIS ENTITY)

          - trainer -> actor_rollout_wg.save_checkpoint (RayWorkerGroup dispatch)
            -> ONE_TO_ALL broadcast -> THIS METHOD -> self.actor.save_checkpoint
            -> engine.save_checkpoint (may upload to HDFS afterward).

        Branches

          - `max_ckpt_to_keep` prunes older checkpoints after successful save.

        Why

          - Keeps the RayPPOTrainer's save path backend-agnostic; FSDP dcp,
            Megatron distributed-save, and TorchTitan dcp all sit behind the
            same ONE_TO_ALL RPC.
        """
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """HybridFlow 3D-HybridEngine weight reshard: push trained actor params onto the rollout engine.

        What

          - This is HybridFlow §4's "3D-HybridEngine" weight resharding step:
            move params from the TRAIN shard layout (e.g. FSDP/Megatron TP/PP)
            onto the INFER shard layout (e.g. vLLM/SGLang TP) on the same GPUs.
          - Two dispatch modes driven by `config.rollout.checkpoint_engine.backend`:
            - "naive" (colocated): trainer and rollout share GPUs. Weights flow
              via `engine.get_per_tensor_param` -> `rollout.update_weights`
              (NCCL broadcast / zero-copy tensor handoff inside the process).
            - not "naive" (async/disaggregated): rollout runs on separate GPUs.
              Weights are shipped via `checkpoint_engine.send_weights` to a
              remote CheckpointEngine (e.g. network blob store / RDMA), then
              the remote rollout pulls and applies them out of band.
          - Async dispatch_mode (`blocking=False`): returns a coroutine so the
            trainer can overlap weight sync with other work.

        Phases (colocated path)

          - 0. Early return if backend != "naive" (disaggregated path).
          - 1. `rollout.resume(tags=["weights"])` — wake the rollout engine's
            weight slots (KV cache stays paged-out until phase 4).
          - 2. Call `engine.get_per_tensor_param(base_sync_done=True)` to gather
            shards into HF-keyed tensors.
          - 3. Two-phase LoRA sync (if adapter mode, not merge):
               3a. If `not self.base_sync_done`: first push BASE weights
                   (`get_per_tensor_param(base_sync_done=False)` then
                   `rollout.update_weights(..., base_sync_done=False)`).
               3b. Then push adapter weights via second
                   `rollout.update_weights(..., base_sync_done=True)`.
             For `peft_merge=True`, LoRA is merged into base in the engine so
             phase 3a is skipped and rollout sees a plain HF weight update with
             peft_config=None.
          - 4. Offload actor params to CPU (if `is_param_offload_enabled`) to
            free VRAM for rollout KV cache; `aggressive_empty_cache(force_sync)`.
          - 5. `rollout.resume(tags=["kv_cache"])` — wake KV cache so generation
            can run.

        Lifecycle

          - Called once per PPO outer step, after update_actor and before
            generate_sequences.

        Called by

          - `RayPPOTrainer.fit` -> actor_rollout_wg.update_weights(step).

        Call graph (THIS ENTITY)

          - disaggregated: -> engine.get_per_tensor_param
            -> checkpoint_engine.send_weights (awaited).
          - colocated: -> rollout.resume(weights)
            -> engine.get_per_tensor_param [+maybe base path]
            -> rollout.update_weights -> engine.to("cpu")
            -> aggressive_empty_cache -> rollout.resume(kv_cache).

        Branches

          - `backend != "naive"` -> disaggregated path (phase 0 only).
          - `self.peft_merge` + `peft_config is None` -> plain single-sync.
          - `not self.peft_merge` + `peft_config is not None` -> two-phase
            base+adapter; guarded by `self.base_sync_done`.
          - `self.config.rollout.free_cache_engine` gates
            resume(weights) / resume(kv_cache).

        Why

          - The NCCL zero-copy trade-off: colocated path avoids a PCIe / network
            hop by doing the train-shard -> infer-shard reshape in-GPU, at the
            cost of blocking the rollout during the window. The async
            disaggregated path frees the trainer to continue, at the cost of
            shipping the weights over the fabric. `update_weights` is the one
            RPC where this trade-off is concretely materialized.
        """

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if self.config.rollout.checkpoint_engine.backend != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Generic non-blocking DP_COMPUTE RPC that forwards a method call onto self.checkpoint_engine.

        What

          - Dynamic dispatcher: look up `method` on `self.checkpoint_engine`
            (a CheckpointEngine plugin instance) and invoke it with the given
            args/kwargs. Lets the trainer call backend-specific APIs (e.g.
            `wait_for_send`, `broadcast_bucket`, `pull_weights`) without this
            class having to enumerate them.
          - `dispatch_mode=Dispatch.DP_COMPUTE, blocking=False` — non-blocking
            so the trainer can overlap an async weight push with other PPO
            work; DP_COMPUTE (not DP_COMPUTE_PROTO) because the args/kwargs
            are not a TensorDict payload.

        Lifecycle

          - Used only on the async/disaggregated rollout path where
            `checkpoint_engine.backend != "naive"`. On the colocated path,
            update_weights goes directly to `rollout.update_weights` and this
            method is not needed.

        Called by

          - `RayPPOTrainer.fit` on the async rollout branch, typically to
            signal / poll / flush the CheckpointEngine (e.g. wait for the
            pending `send_weights` to complete before the next step).

        Call graph (THIS ENTITY)

          - trainer -> actor_rollout_wg.execute_checkpoint_engine(method, ...)
            -> DP_COMPUTE (non-blocking) -> THIS METHOD
            -> getattr(self.checkpoint_engine, method)(...).

        Branches

          - None in this method; all branching lives inside the resolved
            CheckpointEngine subclass.

        Why

          - Keeps the CheckpointEngine plugin surface open-ended: new backends
            (RDMA, disaggregated Ray actor, blob store) can add methods
            without editing this class or adding new @register entries.
        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
