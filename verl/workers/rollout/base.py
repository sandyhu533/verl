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

import importlib
from abc import ABC, abstractmethod
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DiffusionModelConfig, HFModelConfig, RolloutConfig

__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    """Four-method contract every rollout backend implements.

    What:

      - The veRL-specific abstraction that unifies vLLM / SGLang / TRT-LLM /
        HF / Naive inference engines behind a single interface.
      - Four methods form the contract:
          * ``generate_sequences`` — sync batch inference, used by classical
            PPO trainers that expect a blocking call returning a DataProto.
          * ``update_weights`` — stream new actor weights into the inference
            engine after each training step (HybridEngine reshard hook).
          * ``resume`` — wake GPU memory (weights and/or kv_cache) from the
            sleep state so generation can proceed.
          * ``release`` — put the inference engine to sleep, freeing GPU
            memory back to the training side.

    Lifecycle:

      - Instantiated inside each rollout worker (see ``engine_workers.py``
        Phase 3 build) with a ``RolloutConfig`` + ``HFModelConfig`` +
        ``DeviceMesh`` describing the TP/DP shard this replica owns.
      - Per PPO iteration the Sharding Manager drives the sleep/wake cycle:
        ``resume(weights) -> update_weights(stream) -> resume(kv_cache) ->
        generate_sequences -> release``.
      - ``release`` returns GPU memory so FSDP / Megatron training shards
        can reclaim it on the same devices (HybridEngine co-location).

    Called by:

      - ``verl/workers/engine_workers.py`` — owns ``self.rollout`` and
        wires it into the training actor worker.
      - ``verl/checkpoint_engine/base.py`` — uses the ``BaseRollout`` as
        the ``server_adapter`` target for weight transfer.
      - ``verl/trainer/ppo/ray_trainer.py`` — indirectly, via the Ray
        actor group that wraps these workers.

    Call graph:

      - Concrete subclasses: ``NaiveRollout``, ``HFRollout``, and the
        async ``ServerAdapter`` variants under
        ``vllm_rollout`` / ``sglang_rollout`` / ``trtllm_rollout``.
      - ``get_rollout_class`` below picks the class at runtime from
        ``_ROLLOUT_REGISTRY``.

    Why:

      - Thin ABC on purpose: the heavy lifting (paged kv cache, continuous
        batching, TP communicator) lives inside each backend; veRL only
        needs a uniform seam for the Sharding Manager to call into.
      - ``resume`` / ``release`` correspond to HybridFlow §4 3D-HybridEngine
        sleep-wake: training and rollout share GPUs, so memory must be
        yielded between phases rather than statically partitioned.
      - ``update_weights`` takes a streaming generator (not a full state
        dict) so the reshard can overlap transfer with consumption and
        avoid materializing 2x weights on the rollout side.
      - ``generate_sequences`` stays sync to keep the classical PPO path
        simple; async backends expose an ``ServerAdapter`` that wraps
        the same interface for the agent-loop path.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig | DiffusionModelConfig,
        device_mesh: DeviceMesh,
        *args,
        **kwargs,
    ):
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig | DiffusionModelConfig = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Wake the inference engine from the sleep state.

        What:

          - Re-materializes one or both of {``weights``, ``kv_cache``}
            on GPU so that ``generate_sequences`` can execute.

        Lifecycle:

          - Called at the start of each rollout phase by the Sharding
            Manager, typically twice: once with ``["weights"]`` before
            ``update_weights``, and once with ``["kv_cache"]`` right
            before generation.

        Branches:

          - ``tags`` controls which pools are resumed; callers may pass
            a subset so weights can be updated while the kv cache stays
            freed (the kv cache is the larger allocation).

        Why:

          - Paired with ``release`` to implement HybridFlow §4
            3D-HybridEngine sleep-wake: the rollout engine yields its
            memory between PPO iterations so colocated training shards
            can use the same GPUs.
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Stream fresh actor weights into the inference engine.

        What:

          - Replaces the rollout-side model parameters with the latest
            training-side values after each PPO update step.

        Lifecycle:

          - Invoked by the Sharding Manager between training and
            generation phases, after ``resume(["weights"])``.

        Called by:

          - The weight-transfer path inside
            ``verl/workers/engine_workers.py`` and the checkpoint
            engine's ``server_adapter`` flow.

        Why:

          - Takes a streaming generator (name, tensor) rather than a
            full state dict so transfer and engine-side loading can
            overlap, avoiding a peak of 2x weight memory on the
            rollout shard.
          - Central to HybridFlow §4 3D-HybridEngine: training may use
            FSDP/Megatron sharding while rollout uses a different TP
            layout, so the generator is the natural place to do the
            per-parameter reshard/all-gather.
        """
        pass

    @abstractmethod
    async def release(self):
        """Put the inference engine to sleep, freeing GPU memory.

        What:

          - Releases both weights and kv cache pools back to the
            allocator so colocated training shards can use the HBM.

        Lifecycle:

          - Called at the end of each rollout phase, after all
            ``generate_sequences`` calls for the current PPO iteration
            have returned.

        Why:

          - Counterpart to ``resume``; implements the sleep half of
            the HybridFlow §4 3D-HybridEngine sleep-wake cycle.
          - Without this, rollout memory (often tens of GB for the
            paged kv cache) would stay pinned during the backward /
            optimizer step and starve training.
        """
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Blocking batch inference entry point for classical PPO.

        What:

          - Takes a batch of prompts and returns the generated
            continuations as a ``DataProto``, synchronously.

        Lifecycle:

          - Called from the rollout phase of the PPO trainer loop
            after ``update_weights`` and the kv-cache resume.

        Called by:

          - The sync rollout path used by ``verl/trainer/ppo/ray_trainer``
            for non-agent workloads (e.g. ``NaiveRollout``, ``HFRollout``).

        Branches:

          - ``NotImplementedError`` on the base class: async
            ``ServerAdapter`` backends deliberately do not implement
            this and expose an agent-loop API instead.

        Why:

          - Keeping the sync signature on the ABC lets the classical
            PPO trainer treat rollout as a plain function call, while
            the async server backends route generation through their
            own event-loop schedulers for better throughput under
            continuous batching.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("vllm_omni", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}


def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
