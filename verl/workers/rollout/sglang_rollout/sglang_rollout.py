# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from peft import LoraConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import (
    SGLANG_LORA_NAME,
    get_named_tensor_buckets,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.

    What

      - Sync-friendly BaseRollout façade over the async SGLang HTTP server. Unlike the in-process
        vLLMRollout, SGLang runs as a separate server process (launched elsewhere as a detached Ray
        actor named ``sglang_server_{replica}_{node}``); this class is a thin HTTP client that drives
        that server via ``AsyncHttpServerAdapter``.
      - Implements the BaseRollout four-method contract (resume / release / update_weights / generate)
        so trainer-side sharding/rollout orchestration is identical to the vLLM path.

    Lifecycle

      - Constructed once per rollout worker by the FSDP/Megatron workers. The actual HTTP engine
        handle is created lazily on first use (``_init_server_adapter``) because the SGLang server
        is launched *after* the hybrid training engine, so we cannot bind a client at __init__ time.
      - Only ``infer_tp`` local-rank-0 holds a live HTTP client; other TP ranks are no-ops that still
        need to occupy a GPU slot so Ray does not reschedule another actor onto the same device.

    Call graph

      - resume / release / update_weights delegate to ``AsyncHttpServerAdapter`` -> SGLang server
        (``/resume_memory_occupation``, ``/release_memory_occupation``, sgl_update_weights).
      - update_weights piggybacks on SGLang's CUDA-IPC weight-sync (THUDM/slime style), bucketed via
        ``get_named_tensor_buckets`` to amortize per-bucket IPC handle rebuild cost.

    Why (SGLang vs vLLM)

      - SGLang's win is the RadixAttention prefix cache -> shared prefixes across multi-turn tool
        loops and structured-program agents are deduplicated; grammar-constrained decoding is native.
      - Same BaseRollout contract means RL trainers (GRPO/PPO) swap engines by config only.
      - The out-of-process server model forces the sync/async bridge here: the Ray actor stays sync
        while the SGLang client is async, hence ``AsyncHttpServerAdapter`` + ``await`` call sites.

    Paper pointer

      - SGLang: Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs"
        (radix prefix cache is the multi-turn differentiator).
      - update_weights path corresponds to HybridFlow §4 3D-HybridEngine weight resharding: training
        layout -> inference layout -> push to inference engine without extra all-gather on the host.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        """
        What

          - Resolve this worker's position within the rollout replica grid and stash config; do NOT
            connect to the SGLang server yet (lazy in ``_init_server_adapter``).

        Lifecycle

          - Called once per rollout worker during FSDP/Megatron worker startup, before the SGLang
            HTTP server actor exists.

        Branches

          - fp8 quantization: inject a fixed block-quant config onto ``hf_config`` so SGLang's weight
            loader materializes an fp8 model rather than bf16. Gated on sglang>=0.5.5.
          - replica_rank == -1: derive replica from global RANK // rollout_world_size (standard
            hybrid-engine co-location); otherwise trust an externally assigned replica id.

        Why

          - replica_rank / node_rank / local_rank are recomputed (not reused from env) because the
            rollout group is a *subset* of the global world (one replica spans TP x DP ranks only),
            so its internal ordering differs from global RANK.
          - sleep_level is mutable state touched by engine_workers.update_weights() to switch between
            LoRA-adapter fast path (sleep_level=1, keep base weights) and full merge (sleep_level=2).
        """
        super().__init__(config, model_config, device_mesh)
        if self.config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            self.model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

        # sleep_level controls what gets released during sleep/release:
        #   2 (default) = release weights + kv_cache (full sleep, merge path)
        #   1 = release kv_cache only (keep base weights, adapter path)
        # Set by engine_workers.update_weights() when lora.merge=False.
        self.sleep_level = 2

    async def _init_server_adapter(self):
        """
        What

          - Lazily bind this rank to its SGLang HTTP server (one server per (replica, node)) by
            looking up the named Ray actor and reading the bound host/port.

        Lifecycle

          - Guarded by idempotent ``self._engine is not None`` check, invoked at the top of every
            public method (resume/release/update_weights). Runs once per worker.

        Branches

          - device_mesh missing: synthesize a (dp, infer_tp, infer_pp) CPU mesh from torch.distributed
            world size. Required because weight-sync IPC rebuild uses the ``infer_tp`` sub-mesh.
          - ``infer_tp`` local-rank != 0: early-return without an HTTP client; only TP rank 0 talks
            to the server (SGLang server internally fans out to its own TP workers).

        Why

          - Lazy init: the SGLang server actor is launched *after* the hybrid engine because it must
            inherit CUDA memory after training state is placed. Constructing the HTTP client in
            __init__ would race the server startup.
        """
        if self._engine is not None:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only init http server adapter in tp rank 0
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.

        What

          - Ask the SGLang server to re-materialize previously released GPU state (weights and/or
            kv_cache) so the engine is ready to serve generate requests.

        Lifecycle

          - Called by the sharding manager right before a rollout batch, mirroring vLLM's wake_up().
          - Paired with ``release`` which is called after the rollout batch to hand GPU memory back
            to the training engine (HybridFlow co-location).

        Branches

          - ``free_cache_engine`` False: skip; memory is never released in this mode so no-op.
          - Non-tp-rank-0: skip; SGLang server handles internal fan-out.

        Why

          - Tag-selective resume (vs unconditional wake) supports the LoRA fast path where weights
            stay resident and only kv_cache needs to come back.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory.

        When sleep_level=1 (LoRA adapter mode), only releases kv_cache
        to keep base weights alive across training iterations.
        When sleep_level=2 (default/merge mode), releases everything.

        What

          - Hand GPU memory (kv_cache and optionally weights) back to the training engine so the
            next optimizer step has headroom. SGLang's equivalent of vLLM ``sleep``.

        Lifecycle

          - Invoked by the sharding manager after a rollout batch completes, before training
            resumes on the co-located GPUs.

        Branches

          - sleep_level == 1: LoRA adapter path, keep base weights resident (saves the reload cost
            next iteration); release kv_cache only.
          - sleep_level == 2: full release including weights; next ``resume`` + ``update_weights``
            will repopulate them.

        Why

          - Matches HybridFlow's co-located memory model: training and rollout time-share the same
            GPUs, so peak memory per phase is bounded by releasing the other phase's state.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            if self.sleep_level == 1:
                tags = ["kv_cache"]
            else:
                tags = ["kv_cache", "weights"]
            await self._engine.release_memory_occupation(tags=tags)

    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39

        What

          - Push freshly trained weights from the training engine into the SGLang inference server.
            Streams named tensors through size-bounded buckets and CUDA IPC so we never materialize
            the full model twice on the host.

        Lifecycle

          - Called by the sharding manager each training step between optimizer.step() and the next
            rollout (the "resharding" edge in HybridFlow §4). Follows ``resume(['weights'])`` if the
            engine was fully asleep.

        Call graph

          - bucket loop -> ``sgl_update_weights`` -> SGLang server (rebuild_cuda_tensor IPC handles
            on the server side, zero-copy into inference weight slots).
          - On TP rank 0: ``flush_cache`` (invalidate radix prefix cache entries keyed on old
            weights) and bump ``global_steps`` on the server actor for telemetry / checkpoint
            coordination.

        Branches

          - peft_config + base_sync_done: LoRA adapter path. Unload prior adapter if present, then
            push adapter tensors via ``load_lora_adapter_from_tensor`` (serialize once per infer_tp
            rank because the server-side multi-proc deserializer expects per-rank payloads).
          - fp8 quantization: convert bf16 -> fp8 on-the-fly via SGLangFP8QuantizerHelper before
            bucketing so the server never sees bf16.
          - Full-weight path: bucket tensors by ``update_weights_bucket_megabytes`` and stream.

        Why

          - Buckets trade one big IPC (OOM risk on server side during rebuild) for many small ones.
          - LoRA adapter path avoids rebroadcasting the frozen base model each step (major RLHF win).
          - ``flush_cache`` is required because RadixAttention entries cached under the old weight
            version would produce stale KVs.
        """
        await self._init_server_adapter()

        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # unload lora
                models_result = await self._engine.available_models()
                exists = any(item["id"] == SGLANG_LORA_NAME for item in models_result["data"])
                if exists:
                    await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)

                # load lora by tensor
                serialize_peft_config, serialize_named_tensors = self.wrap_lora_params(peft_config, weights)
                from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput

                req = LoadLoRAAdapterFromTensorsReqInput(
                    lora_name=SGLANG_LORA_NAME,
                    config_dict=serialize_peft_config,
                    serialized_tensors=serialize_named_tensors,
                )
                # send http request
                await self._engine.load_lora_adapter_from_tensor(req)
        else:
            update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
            if self.config.get("quantization", None) == "fp8":
                from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

                logger.info("Convert bf16 weights to fp8 format before loading")
                fp8_quantizer_helper = SGLangFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
                weights = fp8_quantizer_helper.quant_weights_by_name(
                    weights,
                    dtype=self.model_config.hf_config.dtype,
                )
            else:
                weights = weights

            async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                await sgl_update_weights(
                    engine=self._engine,
                    params_batch=params_batch,
                    device_mesh_key="infer_tp",
                    device_mesh=self.device_mesh,
                )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)

    def wrap_lora_params(self, peft_config: LoraConfig, weights: Generator[tuple[str, torch.Tensor]]):
        """
        What

          - Package a PEFT ``LoraConfig`` + adapter tensors into the (json-dict, serialized-tensors)
            pair that SGLang's ``LoadLoRAAdapterFromTensorsReqInput`` expects.

        Lifecycle

          - Invoked inside ``update_weights`` on the LoRA branch, once per update step.

        Why

          - Enums (task_type, peft_type) and sets (target_modules) are not JSON-serializable; cast
            to value / list so the config survives the HTTP boundary.
          - The serialized tensor payload is replicated ``infer_tp_size`` times because the SGLang
            server fan-out expects one deserializable blob per inference-TP rank (the
            MultiprocessingSerializer payload is rank-agnostic, so the same bytes are duplicated).
        """
        # peft config
        peft_config_json = asdict(peft_config)
        peft_config_json["task_type"] = peft_config_json["task_type"].value
        peft_config_json["peft_type"] = peft_config_json["peft_type"].value
        peft_config_json["target_modules"] = list(peft_config_json["target_modules"])

        # lora weights
        processed_weights: dict[str, torch.Tensor] = {
            name: _preprocess_tensor_for_update_weights(tensor.detach()) for name, tensor in weights
        }

        infer_tp_size = self.device_mesh["infer_tp"].mesh.size()[0]
        serialized_named_tensors = []
        for i in range(infer_tp_size):
            serialized_tensors = MultiprocessingSerializer.serialize(processed_weights, output_str=True)
            serialized_named_tensors.append(serialized_tensors)

        return peft_config_json, serialized_named_tensors
