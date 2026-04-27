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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import time
from typing import Any, Generator, Optional

import ray
import torch
from packaging import version as vs
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import get_device_id, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.utils import get_device_uuid

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.

    What:
      - Thin sync-style CLIENT that drives a remote vLLM async-engine SERVER actor
        (spawned by vLLMReplica) over Ray RPC. Implements the BaseRollout four-method
        API (generate_sequences / update_weights / resume / release).
      - Sibling file: vllm_async_server.py hosts the matching `collective_rpc` server
        that terminates the RPCs dispatched from here.

    Lifecycle:
      - __init__ computes this worker's (replica_rank, rollout_rank, node_rank) from
        RANK/RAY_LOCAL_WORLD_SIZE and the (TP * DP * PP) rollout group size; the Ray
        actor handle to the server is resolved lazily on first _execute_method call
        (server is launched AFTER the hybrid engine so it is not addressable at ctor
        time).
      - Per training step: SharingManager.__enter__ -> update_weights (reshard from
        FSDP/Megatron trainer into vLLM) -> resume(kv_cache) -> rollout ->
        release() -> SharingManager.__exit__.

    Call graph:
      - Upward: vLLMReplica / AsyncLLMServerManager and RayPPOTrainer's rollout phase.
      - Downward: ray.actor -> vllm_async_server.AsyncvLLMServer.collective_rpc ->
        vllm.AsyncLLMEngine.collective_rpc -> per-GPU worker methods
        (wake_up/sleep/update_weights_from_ipc). Bulk weight tensors bypass Ray and
        travel via a per-device ZMQ IPC socket driven by BucketedWeightSender.

    Why:
      - vLLM runs in-process as a library (not an external HTTP server) so the
        HybridEngine (HybridFlow S4) can COLOCATE training and rollout on the same
        GPUs and avoid keeping 2x model copies in HBM. sleep/wake (PagedAttention
        SOSP'23 KV-release semantics) frees the KV cache and optionally weights
        between phases.
      - Weight reshard (trainer sharding -> rollout TP sharding) goes over CUDA IPC
        handles bucketed through ZMQ rather than Ray object store: zero-copy GPU<->GPU
        on the same host, avoids pickling multi-GB tensors, and lets the server-side
        worker pull just its TP shard. Falls back to /dev/shm when IPC is unavailable
        (e.g. Ascend NPU without CANN >= 8.3.RC1).
      - Only rollout_rank 0 issues the RPC; the server's collective_rpc fans out to
        all TP/PP workers. This keeps the client SPMD-safe while the engine stays
        collective internally.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        """
        What:
          - Compute rank coordinates inside the rollout replica and pick the ZMQ
            endpoint / sleep level used for the lifetime of this adapter.

        Lifecycle:
          - Constructed once per rollout worker when vLLMReplica binds a BaseRollout;
            server_handle is resolved LAZILY in _execute_method because the vLLM
            async server actor is launched after the hybrid engine is up.

        Branches:
          - replica_rank defaults to (global rank // rollout_world_size); override
            honored when the caller already knows the replica index.
          - sleep_level=1 is forced when layered_summon is set or when expert
            parallelism is on with an old vLLM (< 0.11.0, see vllm issue #25171).
            Otherwise uses VLLM_SLEEP_LEVEL (typically 2 = also drop weights).
          - use_shm=True when the device runtime cannot expose CUDA IPC handles
            (non-CUDA or older Ascend CANN); bulk weight transfer then falls back
            to shared memory via BucketedWeightSender.

        Why:
          - One ZMQ socket per physical device (keyed by device UUID) so multiple
            colocated replicas on the same host do not collide, and so the sender
            in the trainer process and the receiver in the vLLM worker process can
            find each other without going through Ray.
        """
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        # Use replica_rank + node-local rank to form ZMQ handle instead of GPU UUID,
        # because CheckpointEngineWorker and vLLM worker may see different GPU UUIDs
        # when CUDA_VISIBLE_DEVICES differs between processes (common on ROCm/AMD).
        # Must use node-local rank (not rollout_rank) so it matches vLLM worker's
        # local_rank on every node. Include replica_rank to avoid collisions when
        # multiple replicas share a node.
        local_rank = self.rollout_rank % local_world_size
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-replica-{self.replica_rank}-rank-{local_rank}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.

        What:
          - Generic RPC dispatch: forwards a named method + args to the vLLM async
            server actor, which internally calls engine.collective_rpc to fan out
            to every TP/PP worker inside the engine.

        Lifecycle:
          - First call resolves the named Ray actor
            `{prefix}server_{replica_rank}_{node_rank}` and caches the handle. The
            server is created by vLLMReplica after the hybrid engine has
            initialized, so this lazy lookup is load-bearing.

        Call graph:
          - resume / release / update_weights -> _execute_method(method, ...)
            -> ray actor .collective_rpc.remote(...) -> vllm_async_server
            .collective_rpc -> vllm AsyncLLMEngine.collective_rpc -> per-worker
            method (wake_up/sleep/update_weights_from_ipc/clear_kv_cache).

        Branches:
          - rollout_rank != 0 returns None: avoids N identical SPMD callers all
            issuing the same collective; only the group leader talks to the server.
          - non_block=True returns the awaitable so the caller can overlap RPC
            round-trip with work that must happen first on the client side (see
            update_weights: server is told to start receiving BEFORE the sender
            begins pushing tensors on ZMQ).

        Why:
          - Single chokepoint keeps the Ray boundary narrow and makes RPC method
            names (strings) the contract with the server-side dispatcher, letting
            vLLM internals evolve without the trainer importing vLLM symbols.
        """
        if self.rollout_rank != 0:
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.

        What:
          - Wake the vLLM engine: re-allocate the tagged GPU buffers (weights and/or
            paged-KV blocks) that were freed by release()/sleep().

        Lifecycle:
          - Called by the rollout sharding manager when the hybrid engine transitions
            from training to rollout. Typical order within update_weights flow:
            resume(["weights"]) -> update_weights -> resume(["kv_cache"]).

        Branches:
          - Guarded by free_cache_engine: when colocation is disabled vLLM owns HBM
            full-time and there is nothing to wake.

        Why:
          - Implements the rollout half of HybridEngine memory time-sharing (vLLM
            SOSP'23 KV-cache release / HybridFlow S4). Weights and KV blocks are
            reclaimed for training and returned for generation instead of reserving
            both concurrently.
        """
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory.

        What:
          - Put the vLLM engine to sleep: free KV blocks and (at level 2) also the
            weight tensors so training can reclaim HBM.

        Lifecycle:
          - Called by the sharding manager when leaving the rollout phase.

        Branches:
          - sleep_level = VLLM_SLEEP_LEVEL (2) normally -> drop weights + KV.
          - sleep_level = 1 when layered_summon or EP>1 on vLLM <0.11.0 -> keep
            weights, drop KV only (trade HBM for fewer weight reloads).

        Why:
          - Pair to resume(). This is the training-side reclamation step of the
            HybridEngine time-share: without it, vLLM would keep its PagedAttention
            block pool reserved during the optimizer step, leaving no room for FSDP
            full-parameter gather / Megatron activations.
        """
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})

    @torch.no_grad()
    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers.

        What:
          - Reshard trainer-side weights (FSDP or Megatron TP/PP layout) into the
            vLLM rollout's TP layout, handing them over through a bucketed
            zero-copy channel instead of Ray object store.

        Lifecycle:
          - Entry point of the HybridEngine weight-reload path. Called by the
            rollout sharding manager on every training step once the actor has
            produced fresh weights.

        Call graph:
          - _execute_method("update_weights_from_ipc", non_block=True) primes the
            server-side receiver on all TP workers.
          - BucketedWeightSender.async_send_weights(weights) streams (name, tensor)
            pairs over ZMQ using CUDA IPC handles (or /dev/shm fallback), grouped
            into `update_weights_bucket_megabytes`-sized buckets to amortize
            per-message overhead.
          - await future: barrier — wait until every TP worker has ingested its
            shard before releasing the generator.
          - rollout_rank 0 then RPCs clear_kv_cache (prefix cache is invalidated
            because weights changed) and set_global_steps for telemetry.

        Branches:
          - use_shm toggles the transport mode; the server also receives this flag
            so sender and receiver agree.
          - global_steps is optional (skipped for cold eval paths).

        Why:
          - HybridFlow S4 3D-HybridEngine: trainer and rollout have DIFFERENT
            parallelism geometries, so weights must be reassembled (all-gather
            along trainer's TP, scatter along rollout's TP) each step. Doing this
            via CUDA IPC + ZMQ keeps the transfer on-device, avoids pickling
            multi-GB tensors through Ray, and lets the rollout worker pull exactly
            its shard. Bucketing is the throughput knob: too small => ZMQ overhead
            dominates; too large => tail latency + HBM spike on the receiver.
          - Prefix-cache reset is mandatory: radix blocks are indexed by KV
            contents which depend on the old weights; reusing them after an
            update would produce silently wrong logits.
        """
        start_time = time.time()

        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset prefix cache after updating weights
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix matching the rollout type (e.g. 'vllm_' or 'vllm_omni_')."""
        return f"{self.config.get('name', 'vllm')}_"

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ServerAdapter uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.

        What:
          - Deliberately-disabled BaseRollout hook. ServerAdapter only speaks to an
            async vLLM server, so DataProto -> SamplingParams -> engine.generate
            -> DataProto batching must go through AsyncLLMServerManager instead.

        Call graph:
          - Expected async path: vLLMReplica owns this adapter for control-plane
            ops; generation requests are routed per-request to the async server's
            /generate endpoint by AsyncLLMServerManager.

        Why:
          - SPMD in-process generate() was retired in PR #4411; N-sample expansion
            is already flattened to an n*B batch by the trainer before the rollout
            phase, and the async server gives better request-level interleaving
            (ORCA-style iteration-level scheduling) than a blocking ray.get on a
            whole batch. Keeping this as a hard raise prevents silent fallback.
        """
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/verl-project/verl/issues/4682 for more details."
        )
