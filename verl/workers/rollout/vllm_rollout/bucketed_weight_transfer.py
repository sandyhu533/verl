# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """Streams sharded model weights into the colocated rollout engine in fixed-size buckets.

    What:
      - Packs training-side parameter tensors into one fixed-size uint8
        "bucket" buffer and ships them to a `BucketedWeightReceiver`
        running inside the vLLM worker process on the same node.
      - The bucket is transported by CUDA IPC handle (default) or by
        POSIX shared memory (`use_shm=True`, e.g. NPU / non-CUDA devices).
        ZMQ is used only as a tiny REQ/REP control channel that passes the
        IPC/shm handle once and then signals "bucket ready" per round.

    Lifecycle:
      - __init__: record config, lazily allocate socket / buffer.
      - async_send_weights(): init socket -> send buffer handle -> loop
        (fill bucket | flush once full) -> flush last bucket -> cleanup.
      - _cleanup(): close socket, drop buffer, unlink shm, force
        `ipc_collect` + `empty_cache` so the engine side can reclaim VRAM
        before generation resumes.

    Called by:
      - verl.workers.rollout.vllm_rollout.vllm_rollout -- the HybridEngine
        sharding manager instantiates one sender per rollout worker when
        resharding trained weights back into the vLLM engine after each
        PPO/GRPO update step.

    Call graph:
      - async_send_weights() -> ensure_async_iterator(weights) ->
        device.synchronize() + socket.send_pyobj({bucket_meta, is_last})
        -> receiver rebuilds tensors from the same IPC buffer and calls
        `on_bucket_received` (typically `model.load_weights`).

    Branches:
      - Bucket overflow (`offset + nbytes > bucket_size`): synchronize,
        flush the current bucket, wait for ACK, then start a fresh one.
        Final `is_last=True` message drains the tail bucket.
      - CUDA IPC vs shared memory: IPC is zero-copy on-GPU; shm path
        materializes on CPU and the receiver does a `.to(device)` copy.
      - One weight exceeding `bucket_size` hard-asserts -- there is no
        per-tensor chunking yet (embedding layer is the likely culprit;
        see the TODO in the loop).

    Why:
      - Canonical reference: HybridFlow / 3D-HybridEngine (section 4 of
        the veRL paper) -- "amortized weight reshard IPC". Sending one
        tensor per message would be dominated by Python pickling and ZMQ
        syscall overhead (thousands of params); a single mega-buffer
        would spike peak memory on both sides and starve overlap between
        copy-in and copy-out. Bucketing amortizes serialization while
        bounding extra VRAM to `bucket_size_mb`.
      - Interview-ready knob: `bucket_size_mb` is the throughput / memory
        trade-off -- too small wastes syscalls and hurts copy-engine
        overlap; too large inflates peak device memory and delays the
        first bucket (latency to first token after a step).
      - ZMQ here is deliberately NOT the data plane; it is only a
        rendezvous for the handle and a per-bucket barrier. The real
        transport is the OS shared page (CUDA IPC or POSIX shm), which
        is why this file sits next to the HybridEngine sharding manager
        rather than in a generic networking layer.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock").
        bucket_size_mb: Communication buffer size in MB (throughput knob).
        use_shm: Use POSIX shared memory instead of CUDA IPC (NPU-compatible path).
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        """Pack and stream (name, tensor) pairs across bucket boundaries.

        What:
          - Drains a (possibly async) iterator of trained weights, copies
            each tensor's raw bytes into `self.buffer` at the current
            offset, and accumulates a `bucket_meta` dict describing its
            layout (shape, dtype, offset) for the receiver.

        Lifecycle:
          - _init_socket() + _init_buffer() run once (handshake passes
            the IPC/shm handle), then a fill-or-flush loop, then a final
            flush with `is_last=True`, then `_cleanup()` in `finally`.

        Called by:
          - `vllm_rollout.update_weights` / the HybridEngine sharding
            manager after the training step has reshaped / gathered
            FSDP or TP shards into generator form.

        Branches:
          - Overflow path: device.synchronize() before send to make sure
            all async H2D/D2D copies into the buffer have landed, then
            REQ-send the metadata and block on receiver ACK before
            reusing the buffer for the next bucket.
          - Non-overflow path: pure memcpy into the buffer; zero ZMQ
            traffic until the bucket fills up.

        Why:
          - The commented-out `weight.to(dtype, non_blocking=True)` is a
            deliberate decision (see vermouth1992 note): some layers
            (e.g. MoE gate) must stay fp32 for numerical stability, so
            casting is pushed to the rollout side on demand -- at the
            cost of larger per-bucket bytes. This is a trade the
            sharding manager owns, not the transport.
          - The REQ/REP handshake around every bucket gives implicit
            back-pressure: the sender cannot race ahead of the
            receiver's `load_weights` callback, which would otherwise
            corrupt the single shared buffer.

        Args:
            weights: Sync generator or async iterator yielding (name, tensor).
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    get_torch_device().synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
