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
import inspect
import logging
import os
import socket
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import ray
from ray.experimental.state.api import get_actor
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.utils.device import get_device_name, is_torch_npu_available
from verl.utils.py_functional import temp_env_var

__all__ = ["Worker"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_random_string(length: int) -> str:
    import random
    import string

    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    """What:
      - Factory returning a callable Functor that wraps one @register-decorated worker
        method into a scatter-gather pipeline.
      - Pipeline stages: dispatch(shard) -> execute(remote on all workers) -> collect(gather).
      - Returns a class-named instance (type(method_name, (Functor,), {})()) so stack
        traces surface the user-facing method name rather than a generic "Functor".

    Lifecycle:
      - Invoked once per @register method at WorkerGroup construction time from
        _bind_worker_method.
      - Each invocation installs one bound method on self; the returned Functor is
        reused on every trainer-side call.
      - Every wg.generate_sequences(...) / wg.update_actor(...) from trainer code
        enters this Functor.__call__ — it is the hot path of single-controller RPC.

    Called by:
      - verl/single_controller/base/worker_group.py::WorkerGroup._bind_worker_method
      - re-entered from RayWorkerGroup.__init__, RayWorkerGroup.spawn_fused, and
        RayWorkerGroup.fuse (all in this file) whenever methods need (re)binding.

    Call graph:
      bash/entry_point.sh                                [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                [layer: entry]
          -> TaskRunner.remote(...)                      [layer: driver actor]
            -> RayPPOTrainer.init_workers()              [layer: trainer]
              -> RayWorkerGroup.__init__                 [layer: dispatch]
                -> _bind_worker_method(cls, func_generator)
                  -> FUNC_GENERATOR                      [layer: dispatch]
                    -> Functor.__call__                  [layer: dispatch]
                      -> dispatch_fn(self, *a, **kw)     [layer: dispatch]
                        -> execute_fn(method, *a, **kw)  [layer: ray rpc]
                          -> worker.method.remote(...)   [layer: workers]
                            -> collect_fn(self, output)  [layer: dispatch]

    Branches:
      - blocking=True
          -> ray.get(output) materializes remote refs before collect_fn runs.
      - blocking=False
          -> returns raw ObjectRefs; caller is responsible for ray.get later
            (used for async rollout / overlap paths).
      - padding_count > 0 and output is DataProto
          -> trim padded tail via output.select_idxs(indices[:-padding_count]).
      - padding_count > 0 and output is list
          -> slice out the padded tail: output = output[:-padding_count].
      - padding_count == 0
          -> output returned as-is.

    Why:
      - This is HybridFlow's single-controller multi-worker abstraction (HybridFlow §3):
        one driver process issues one Python call, dispatch/collect encode SPMD
        data-parallel semantics, and user worker code stays vanilla.
      - Padding trim preserves the user-visible batch size despite DP-divisibility
        padding inserted by dispatch_fn — a subtle correctness requirement when the
        batch is not a multiple of world_size.
      - type(method_name, (Functor,), {})() costs one extra class per method but buys
        readable stack traces, which matters when debugging rollout/training stalls.
    """

    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """What:
      - Deterministically order a list of Ray PlacementGroups by the node IP hosting them.
      - Input: list[PlacementGroup]; output: the same list sorted by NodeManagerAddress.
      - Guarantees rank_i always lands on the same physical host across job restarts.

    Lifecycle:
      - Called after placement_group().ready() completes, before any rank is assigned.
      - Runs once per pool in RayResourcePool.get_placement_groups, and again from
        _init_with_resource_pool to make the WG's rank<->host mapping reproducible.

    Called by:
      - RayResourcePool.get_placement_groups (this file).
      - RayWorkerGroup._init_with_resource_pool (this file).

    Call graph:
      bash/entry_point.sh                                    [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                    [layer: entry]
          -> TaskRunner.remote(...)                          [layer: driver actor]
            -> RayPPOTrainer.init_workers()                  [layer: trainer]
              -> RayWorkerGroup._init_with_resource_pool     [layer: dispatch]
                -> SORT_PLACEMENT_GROUP_BY_NODE_IP           [layer: placement]
                  -> ray.nodes()                             [layer: ray rpc]
                  -> placement_group_table(pg.id)            [layer: ray rpc]

    Branches:
      - none (single deterministic code path: read NodeID->IP map, sort by IP).

    Why:
      - FSDPCheckpointManager shards model/optimizer state to per-rank local storage.
      - Resume requires rank_i -> host_i to be stable across Ray job restarts;
        Ray's PG scheduler does not guarantee bundle order by itself.
      - Folklore pattern for FSDP resume with local-disk sharded checkpoints —
        no direct paper citation, but widely relied on in production RLHF/SFT stacks.
      - Assumption: each PG's bundles are all on one node (STRICT_PACK in the caller),
        so bundles_to_node_id[0] is representative.

    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


@ray.remote
def get_master_addr_port(master_port_range: Optional[list[int]] = None) -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")

    if master_port_range is None:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
    else:
        port = master_port_range[0]
        while port < master_port_range[1]:
            try:
                with socket.socket() as s:
                    s.bind(("", port))
                    break
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
        else:
            raise RuntimeError(f"Could not find a free port in range {master_port_range}")
    return addr, str(port)


class RayResourcePool(ResourcePool):
    """What:
      - GPU/NPU resource allocation unit.
      - process_on_nodes=[n0, n1, ...] requests one PlacementGroup per node, each with
        n_i bundles; one bundle == one worker slot.
      - Lazy-creates PGs on the first get_placement_groups() call and caches them on
        self.pgs for reuse by subsequent WorkerGroups.

    Lifecycle:
      - Built by ResourcePoolManager.create_resource_pool at trainer startup.
      - Consumed by every RayWorkerGroup that binds to this pool.
      - Typically one pool per role-family; a single pool is shared by
        actor/critic/ref/reward roles when colocated (HybridFlow 3D-HybridEngine).
      - Lives for the full training job (or detached — see branches).

    Called by:
      - verl/trainer/ppo/ray_trainer.py::ResourcePoolManager.create_resource_pool.
      - verl/trainer/main_ppo_sync.py and SFT/diffusion trainer scripts that build
        pools directly.
      - verl/workers/rollout/replica.py and checkpoint_engine setup paths.

    Call graph:
      bash/entry_point.sh                                    [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                    [layer: entry]
          -> TaskRunner.remote(...)                          [layer: driver actor]
            -> RayPPOTrainer.init_workers()                  [layer: trainer]
              -> ResourcePoolManager.create_resource_pool    [layer: trainer]
                -> RAYRESOURCEPOOL(__init__)                 [layer: placement]
                  -> .get_placement_groups(...)              [layer: placement]
                    -> ray.util.placement_group(...)         [layer: ray rpc]
                    -> sort_placement_group_by_node_ip(pgs)  [layer: placement]
              -> RayWorkerGroup(resource_pool=pool)          [layer: dispatch]

    Branches:
      - use_gpu=True + device_name="cuda"
          -> bundle requires 1 GPU.
      - use_gpu=True + device_name="npu"
          -> bundle requires 1 NPU (device_name uppercased internally).
      - accelerator_type set
          -> bundle additionally requires 1e-4 of that label (soft pin to a GPU SKU).
      - detached=True
          -> PGs outlive driver via lifetime="detached"; used for checkpoint_engine
            and any component that must survive trainer crash.
      - self.pgs already cached
          -> early return; get_placement_groups is idempotent.

    Why:
      - STRICT_PACK forces all bundles of one PG onto one node so TP/PP intra-node
        NVLink paths are preserved; cross-node TP would collapse rollout throughput.
      - max_colocate_count CPU slots per bundle gate how many Ray actors
        (WorkerGroups) can coexist on the same GPU:
          - 1 for FSDP (single training engine per GPU).
          - >1 for Megatron where actor/critic/ref can share a device
            (HybridFlow §4 colocation).
      - Fractional num_gpus = 1 / max_colocate_count later in RayClassWithInitArgs
        is the Ray mechanism that actually enforces this sharing on top of the PG.
    """

    def __init__(
        self,
        process_on_nodes: Optional[list[int]] = None,
        use_gpu: bool = True,
        name_prefix: str = None,
        max_colocate_count: int = 10,
        detached=False,
        accelerator_type: Optional[str] = None,
    ) -> None:
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self.pgs = None
        self.detached = detached
        self.accelerator_type = accelerator_type

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = (
            name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        )
        # print(f"pg_name_prefix = {pg_name_prefix}")
        if device_name == "npu":
            device_name = "NPU"
        elif device_name == "cuda":
            device_name = "GPU"

        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = 1
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None

        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = sort_placement_group_by_node_ip(pgs)
        return pgs


class SubRayResourcePool(RayResourcePool):
    """What:
      - A contiguous slice [start_bundle_index, start_bundle_index + subgroup_world_size)
        of an existing parent pool's PG bundles.
      - Reuses the parent's PlacementGroups (does not allocate new ones).
      - Subclass of RayResourcePool; overrides .world_size to report the slice size.

    Lifecycle:
      - Produced by split_resource_pool when a trainer needs to carve one physical
        allocation into N disjoint per-role slices.
      - Consumed by RayWorkerGroup._init_with_subresource_pool, which walks only the
        slice's bundles when spawning actors.
      - Multiple SubRayResourcePools can share the same underlying PGs (one slice per role).

    Called by:
      - split_resource_pool in this file.
      - Trainers that want separate WorkerGroups over one physical allocation
        (non-colocated multi-role layouts).

    Call graph:
      bash/entry_point.sh                                    [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                    [layer: entry]
          -> TaskRunner.remote(...)                          [layer: driver actor]
            -> RayPPOTrainer.init_workers()                  [layer: trainer]
              -> split_resource_pool(pool, split_size)       [layer: placement]
                -> SUBRAYRESOURCEPOOL(__init__)              [layer: placement]
              -> RayWorkerGroup(resource_pool=sub_pool)      [layer: dispatch]
                -> _init_with_subresource_pool(...)          [layer: dispatch]
                  -> _create_worker(...) per bundle          [layer: ray rpc]

    Branches:
      - inherits all RayResourcePool branches (use_gpu, device_name, detached, etc.).
      - world_size property returns subgroup_world_size instead of parent pool's total.

    Why:
      - Enables multi-role non-colocated layouts without paying Ray's PG creation cost
        twice or fragmenting the cluster.
      - Counterpart to FusedWorker colocation:
          - FusedWorker colocation = one actor hosting N roles on one GPU.
          - SubRayResourcePool     = N actors slicing one physical allocation across
            disjoint GPU bundles.
      - Together they cover the full trade space of HybridFlow §4's role placement.
    """

    def __init__(
        self,
        placement_groups: list[PlacementGroup],
        start_bundle_index: int,
        subgroup_world_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pgs = placement_groups
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size

    @property
    def world_size(self):
        return self.subgroup_world_size


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[int, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def extract_pg_from_exist(
    resource_pools: dict[str, RayResourcePool], src_role_names: list[str], resource_pool: RayResourcePool
) -> list:
    src_pgs = [
        pg
        for role_name, resource_pool in resource_pools.items()
        for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: list[tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, (
            f"requesting {request_process} processes, bundle count cannot satisfy"
        )
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


# split a RayResourcePool or SubRayResourcePool into multiple SubRayResourcePool
def split_resource_pool(
    resource_pool: RayResourcePool | SubRayResourcePool, split_size: int | list[int]
) -> list[SubRayResourcePool]:
    """
    Split a RayResourcePool into multiple SubRayResourcePool.
    resouce_pool can also be a SubRayResourcePool (have been splited) for multiple-time spliting.

    Args:
        resource_pool (RayResourcePool | SubRayResourcePool): The resource pool to split.
        split_size (int | list[int]): The size of each split. If int, all splits will have the same size.
            If list[int], each element in the list represents the size of a split.

    Returns:
        list[SubRayResourcePool]: A list of SubRayResourcePool after splitting.
    """
    # convert split_size to list[int]
    if isinstance(split_size, int):
        assert resource_pool.world_size % split_size == 0, "split_size must be a divisor of world_size"
        num_replica = resource_pool.world_size // split_size
        split_size_list = [split_size] * num_replica
    else:
        split_size_list = split_size

    assert sum(split_size_list) == resource_pool.world_size, "split_size must sum up to world_size"

    # judge if this resource pool has been splited
    if isinstance(resource_pool, SubRayResourcePool):
        start_bundle_idx_list = np.cumsum([resource_pool.start_bundle_index] + split_size_list[:-1])
    else:
        start_bundle_idx_list = np.cumsum([0] + split_size_list[:-1])

    # ensure resource_pool.pgs has been initialized
    device = "npu" if is_torch_npu_available(check_device=False) else "cuda"
    placement_groups = resource_pool.get_placement_groups(device_name=device)
    split_resource_pools = [
        SubRayResourcePool(
            process_on_nodes=resource_pool.store,
            use_gpu=resource_pool.use_gpu,
            name_prefix=f"{resource_pool.name_prefix}_split_{split_idx}",
            max_colocate_count=resource_pool.max_colocate_count,
            placement_groups=placement_groups,
            start_bundle_index=start_bundle_idx_list[split_idx],
            subgroup_world_size=split_size_list[split_idx],
        )
        for split_idx in range(len(split_size_list))
    ]
    return split_resource_pools


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, "Both RayResourcePool must either use_gpu or not"
    assert rp1.max_colocate_count == rp2.max_colocate_count, "Both RayResourcePool must has the same max_colocate_count"
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayResourcePool must has the same n_gpus_per_node"
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    new_store = rp1.store + rp2.store

    merged = type(rp1)(
        new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}", rp1.max_colocate_count, rp1.detached
    )
    merged.pgs = rp1.get_placement_groups(device_name=get_device_name()) + rp2.get_placement_groups(
        device_name=get_device_name()
    )

    return merged


class RayClassWithInitArgs(ClassWithInitArgs):
    """What:
      - Deferred Ray-actor spec: remembers (cls, args, kwargs, scheduling options) and
        becomes an actor handle only when __call__(pg, bundle_idx, ...) fires.
      - One instance is reused once per bundle to spawn the full WorkerGroup.
      - Carries mutable _options (Ray actor options) and _additional_resource merged
        in at call time.

    Lifecycle:
      - Constructed by trainer code before ResourcePool is ready (pure spec, no actor).
      - Stored on RayWorkerGroup.ray_cls_with_init after WorkerGroup construction.
      - Invoked from _create_worker per bundle to produce a real Ray actor handle.
      - Re-used inside spawn_fused and fuse to rebind methods across per-role views.

    Called by:
      - verl/trainer/ppo/ray_trainer.py (actor/critic/ref/rollout wrapping).
      - verl/trainer/diffusion/ray_diffusion_trainer.py.
      - verl/workers/rollout/replica.py.
      - create_colocated_worker_cls_fused below (re-wraps the fused class in a cia).

    Call graph:
      bash/entry_point.sh                                    [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                    [layer: entry]
          -> TaskRunner.remote(...)                          [layer: driver actor]
            -> RayPPOTrainer.init_workers()                  [layer: trainer]
              -> RayClassWithInitArgs(cls=ActorWorker, ...)  [layer: dispatch]
                -> RAYCLASSWITHINITARGS.__call__(pg, idx)    [layer: dispatch]
                  -> cls.options(**opts).remote(*a, **kw)    [layer: ray rpc]
                    -> Ray actor materialized on GPU bundle  [layer: workers]

    Branches:
      - sharing_with provided
          -> pin new actor to the same node via NodeAffinitySchedulingStrategy(soft=False)
            and inherit CUDA_VISIBLE_DEVICES from the target actor.
          -> used for rollout engines colocated on a training worker's GPU.
      - use_gpu and device_name == "cuda"
          -> options["num_gpus"] = num_gpus (fractional, 1/max_colocate_count).
      - use_gpu and device_name == "npu"
          -> options["resources"] = {"NPU": num_gpus} (Ray has no first-class NPU key).
      - self._additional_resource has >1 entries
          -> entries merged directly into options (custom labels, memory reservations).
      - default
          -> PlacementGroupSchedulingStrategy pinned to (pg, bundle_idx).

    Why:
      - Splitting "what to run" from "where to run" lets the trainer compose the full
        role graph first and defer scheduling-strategy decisions until PG bundles are
        actually known.
      - Fractional num_gpus is the Ray mechanism enabling multi-role colocation on a
        single device (HybridFlow §4 3D-HybridEngine) — without it, Ray would refuse
        to co-schedule actor+critic+ref on one physical GPU.
      - sharing_with + NodeAffinity is how verl pins a rollout engine to the same
        node (and visible devices) as the training worker whose weights it serves.

    A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        placement_group,
        placement_group_bundle_idx,
        use_gpu: bool = True,
        num_gpus=1,
        sharing_with=None,
        device_name="cuda",
    ) -> Any:
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with
            device_name: Device for training

        Returns:
            A Ray actor handle with the configured options
        """
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=visible_devices, **self.kwargs)

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx
            )
        }
        options.update(self._options)

        if use_gpu and device_name == "cuda":
            options["num_gpus"] = num_gpus
        if use_gpu and device_name == "npu":
            options["resources"] = {"NPU": num_gpus}

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup):
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        worker_handles: list[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        **kwargs,
    ) -> None:
        """What:
          - Bring up (or attach to) a world_size-sized group of Ray actors from one
            RayClassWithInitArgs.
          - Monkey-patch @register-decorated methods onto self so trainer code calls
            wg.method(...) transparently (via func_generator).
          - Mutates self._workers, self._worker_names, self._master_addr, self._master_port.

        Lifecycle:
          - Constructed once per role at trainer init time (or once per fused colocation
            group).
          - After __init__ returns, the WorkerGroup is scatter-gather-ready: trainer
            code can immediately call any @register method.
          - Workers live for the full training job unless detached or explicitly killed.

        Called by:
          - verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.init_workers.
          - verl/trainer/main_ppo_sync.py (sync PPO entry).
          - verl/workers/checkpoint/checkpoint_engine/base.py (detached checkpoint actors).
          - verl/workers/rollout/replica.py (init_hybrid_colocated path).

        Call graph:
          bash/entry_point.sh                                  [layer: launch shell]
            -> verl/trainer/main_ppo.py::main                  [layer: entry]
              -> TaskRunner.remote(...)                        [layer: driver actor]
                -> RayPPOTrainer.init_workers()                [layer: trainer]
                  -> RAYWORKERGROUP.__INIT__                   [layer: dispatch]
                    -> resource_pool.get_placement_groups()    [layer: placement]
                    -> sort_placement_group_by_node_ip(pgs)    [layer: placement]
                    -> _get_master_addr_port(pg, 0)            [layer: ray rpc]
                    -> _create_worker(...) per bundle          [layer: ray rpc]
                      -> ray_cls_with_init(pg, bundle_idx)     [layer: ray rpc]
                        -> Ray actor on GPU bundle             [layer: workers]
                    -> _bind_worker_method(cls, func_generator)[layer: dispatch]

        Branches:
          - worker_names passed and not fused_worker_used
              -> asserts detached-attach mode, skips PG bundle walk.
              -> delegates to _init_with_detached_workers (pure reattach).
          - self._is_init_with_detached_workers
              -> workers already exist elsewhere; just hold handles.
          - resource_pool is SubRayResourcePool
              -> _init_with_subresource_pool (slice layout; bundle indices start at
                resource_pool.start_bundle_index).
          - resource_pool is RayResourcePool (plain)
              -> _init_with_resource_pool: allocate MASTER_ADDR/PORT from pg_idx=0,
                iterate sort_placement_group_by_node_ip(pgs), _create_worker per local_rank.
          - ray_cls_with_init is not None
              -> _bind_worker_method installs dispatch wrappers via func_generator
                for every @register method on the user class.
          - profile_steps set and device_name=="cuda"
              -> runtime_env adds nsight profiler config for each actor.

        Why:
          - This is the single-controller actor materialization step — the point where
            the logical role graph turns into physical Ray actors on specific GPUs
            (HybridFlow §3).
          - Deterministic ordering via sort_placement_group_by_node_ip makes the
            rank<->host mapping reproducible across job restarts, which is load-bearing
            for FSDP checkpoint resume with local-disk sharded state.
          - The fused_worker_used gate routes into the 3D-HybridEngine colocation path
            (HybridFlow §4); non-fused path is the classic one-actor-per-role layout.
          - Method binding happens *after* actor creation so the user class's full
            MRO (including MegatronWorker / FSDPWorker mixins) is visible for the
            @register discovery walk.

        Initialize a RayWorkerGroup.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            name_prefix: Prefix for worker names
            detached: Whether workers should be detached
            worker_names: Names of existing workers to attach to
            ray_wait_register_center_timeout: Timeout for waiting on register center
            **kwargs: Additional keyword arguments
        """
        self._master_addr = kwargs.pop("master_addr", None)
        self._master_port = kwargs.pop("master_port", None)
        self.use_gpu = kwargs.pop("use_gpu", resource_pool.use_gpu if resource_pool is not None else True)
        self._ray_master_port_range = kwargs.pop("master_port_range", None)
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout
        # Whether the WorkerGroup is a Colocate WorkerGroup created by FusedWorker.
        self.fused_worker_used = False if ray_cls_with_init is None else ray_cls_with_init.fused_worker_used
        # if a WorkerGroup is spawned from Colocate WorkerGroup, this indicates which sub-class is binded to
        # this WorkerGroup.
        self.sub_cls_name = ""
        self.device_name = kwargs.get("device_name", "cuda")
        self.profile_steps = kwargs.get("profile_steps", None)
        self.worker_nsight_options = kwargs.get("worker_nsight_options", None)
        self.customized_worker_env = kwargs.get("worker_env", {})
        if self.worker_nsight_options is not None and self.worker_nsight_options["capture-range-end"] is None:
            self.worker_nsight_options["capture-range-end"] = f"repeat-shutdown:{6 * len(self.profile_steps)}"

        if worker_names is not None and (not self.fused_worker_used):
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names, worker_handles=worker_handles)
        elif isinstance(resource_pool, SubRayResourcePool):
            self._init_with_subresource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )
        else:
            self._init_with_resource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

        self.wg_dict = None
        self.method_names = []

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names, worker_handles):
        # ray.get_actor holds a weak reference to the actor, which causes actors garbage collected unexpectedly
        # if we only hold spawn RayWorkerGroup. By passing actor handle explicitly, spawn RayWorkerGroup have
        # strong reference to these actors.
        # https://github.com/ray-project/ray/pull/45699
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(workers)

    def _get_master_addr_port(self, pg, bundle_index=0, master_port_range=None):
        """Get master addr and port for this worker group"""
        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                get_master_addr_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=bundle_index
                    ),
                ).remote(master_port_range=master_port_range)
            )
        elif self._master_addr is not None and self._master_port is not None:
            logger.debug(f"{self._master_addr=} {self._master_port=}")
        else:
            raise ValueError(
                "Both 'master_addr' and 'master_port' must be provided if you intend to manually specify them, "
                "or neither should be provided to use Ray's default assignment."
            )

    def _init_with_resource_pool(
        self,
        resource_pool,
        ray_cls_with_init,
        bin_pack,
        detached,
        worker_env=None,
    ):
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        self.resource_pool = resource_pool
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            if pg_idx == 0:
                self._get_master_addr_port(pg, bundle_index=0, master_port_range=self._ray_master_port_range)

            for local_rank in range(local_world_size):
                rank += 1
                self._create_worker(
                    rank=rank,
                    pg_idx=pg_idx,
                    pg=pg,
                    local_rank=local_rank,
                    resource_pool=resource_pool,
                    ray_cls_with_init=ray_cls_with_init,
                    worker_env=worker_env,
                    detached=detached,
                )

    def _init_with_subresource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None):
        """Initialize the worker group by creating new workers from a resource pool or sub resource pool.
        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size

        rank = -1
        local_world_size = resource_pool.store[0]
        self._get_master_addr_port(
            pgs[resource_pool.start_bundle_index // local_world_size],
            bundle_index=resource_pool.start_bundle_index % local_world_size,
            master_port_range=self._ray_master_port_range,
        )
        for curr_rank in range(resource_pool.start_bundle_index, resource_pool.start_bundle_index + world_size):
            pg_idx = curr_rank // local_world_size
            pg = pgs[pg_idx]
            local_rank = curr_rank % local_world_size
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "

            rank += 1
            self._create_worker(
                rank=rank,
                pg_idx=pg_idx,
                pg=pg,
                local_rank=local_rank,
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                worker_env=worker_env,
                detached=detached,
            )

    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        world_size = resource_pool.world_size
        use_gpu = resource_pool.use_gpu
        if self.use_gpu and not use_gpu:
            raise ValueError("use_gpu is True but resource_pool.use_gpu is False")
        local_world_size = resource_pool.store[0]
        num_gpus = 1 / resource_pool.max_colocate_count

        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "WG_PREFIX": self.name_prefix,
            "WG_BACKEND": "ray",
            "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }
        if worker_env is not None:
            logging.debug(f"Appending ray class env, origin: {env_vars}, customized env: {worker_env}")
            conflict_env_vars = set(env_vars.keys()) & set(worker_env.keys())
            if len(conflict_env_vars) > 0:
                logging.error(
                    f"User customized env vars conflict with system env: {conflict_env_vars} "
                    f"Overriding may cause unexpected behavior."
                )
                raise ValueError(f"Cannot override protected system env: {conflict_env_vars}")
            env_vars.update(worker_env)
        import re

        cia_name = type(ray_cls_with_init.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
        cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
        name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

        if self.profile_steps and self.device_name == "cuda":
            ray_cls_with_init.update_options(
                {
                    "runtime_env": {
                        "env_vars": env_vars,
                        "nsight": self.worker_nsight_options,
                    },
                    "name": name,
                }
            )
        else:
            ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})

        if detached:
            ray_cls_with_init.update_options({"lifetime": "detached"})

        # create a worker
        worker = ray_cls_with_init(
            placement_group=pg,
            placement_group_bundle_idx=local_rank,
            use_gpu=self.use_gpu,
            num_gpus=num_gpus,
            device_name=self.device_name,
        )
        self._workers.append(worker)
        self._worker_names.append(name)

    @property
    def worker_names(self):
        return self._worker_names

    @classmethod
    def from_detached(
        cls,
        name_prefix=None,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
        **kwargs,
    ):
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            ray_cls_with_init: Class with initialization arguments for workers

        Returns:
            A new RayWorkerGroup instance
        """
        worker_group = cls(
            resource_pool=None,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix=name_prefix,
            worker_names=worker_names,
            worker_handles=worker_handles,
            **kwargs,
        )
        return worker_group

    def spawn(self, prefix_set):
        """What:
          - Carve one physical WorkerGroup into per-role view WorkerGroups.
          - Each view exposes only its own prefix's methods (e.g. "actor_update_actor"
            -> "update_actor" on the "actor" view).
          - Returns {prefix: RayWorkerGroup}.
          - No new Ray actors are spawned — actor handles are reused across views.

        Lifecycle:
          - Called by trainer code after RayWorkerGroup.__init__ to get one handle
            per logical role when a single colocated actor hosts multiple roles.
          - Resulting per-role WGs live as long as the underlying actors.

        Called by:
          - fuse (this class) — fuse delegates to spawn then attrsets the results.
          - Trainer code that holds a colocated WorkerGroup and wants per-role API
            surfaces (e.g. wg_dict["actor"].update_actor(...) vs wg_dict["rollout"]).

        Call graph:
          bash/entry_point.sh                                  [layer: launch shell]
            -> verl/trainer/main_ppo.py::main                  [layer: entry]
              -> TaskRunner.remote(...)                        [layer: driver actor]
                -> RayPPOTrainer.init_workers()                [layer: trainer]
                  -> wg = RayWorkerGroup(fused cia)            [layer: dispatch]
                    -> wg.SPAWN({"actor","rollout",...})       [layer: dispatch]
                      -> self.from_detached(name, handles)     [layer: dispatch]
                        -> RayWorkerGroup._init_with_detached_workers
                      -> spawn_fused(prefix_set)               [layer: dispatch]
                        -> _bind_worker_method(raw_cls_dict[k],
                                               func_generator) [layer: dispatch]

        Branches:
          - self.fused_worker_used == True
              -> delegate to spawn_fused; method calls route through
                FusedWorker._fuw_execute via the "{cls}_fwmn_{method}" naming key.
          - self.fused_worker_used == False (legacy create_colocated_worker_cls path)
              -> _rebind_actor_methods strips the "<role>_" prefix installed by
                _bind_workers_method_to_parent so wg.update_actor() aliases
                wg.actor_update_actor().

        Why:
          - Gives trainer code the illusion of N separate WorkerGroups while
            physically they share one set of Ray actors.
          - Key to HybridFlow §4 3D-HybridEngine memory sharing: actor/critic/ref
            can share one GPU process, amortizing CUDA context, NCCL communicator,
            and — in the fused path — model weights when shapes match.
          - Keeping spawn a pure rebinding step (no new actors) means trainer code
            can build/tear down views cheaply without resource-scheduler round trips.

        Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        if self.fused_worker_used:
            return self.spawn_fused(prefix_set)

        def _rebind_actor_methods(worker_group, actor_name):
            prefix: str = actor_name + "_"
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._workers,
                ray_cls_with_init=self.ray_cls_with_init,
                profile_steps=self.profile_steps,
                worker_nsight_options=self.worker_nsight_options,
            )

            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def spawn_fused(self, prefix_set):
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        """What:
          - Inverse-perspective of spawn: attach each per-role sub-WorkerGroup as an
            attribute on self (self.actor, self.critic, ...).
          - Also bind the top-level FusedWorker methods directly on self so
            self.method() routes through _fuw_execute.
          - Populates self.wg_dict (if unset) by internally calling self.spawn first.

        Lifecycle:
          - Called once after constructing a colocated WorkerGroup when trainer code
            prefers self.actor.x() / self.critic.x() / self.x() ergonomics over a dict.
          - Idempotent on self.wg_dict: re-calling fuse does not respawn views.

        Called by:
          - Trainer orchestration code for 3D-HybridEngine layouts (e.g. PPO ray_trainer
            when actor/critic/ref share one FusedWorker actor).
          - Any code path that wants one "super-WG" object with both per-role and
            top-level method surfaces.

        Call graph:
          bash/entry_point.sh                                  [layer: launch shell]
            -> verl/trainer/main_ppo.py::main                  [layer: entry]
              -> TaskRunner.remote(...)                        [layer: driver actor]
                -> RayPPOTrainer.init_workers()                [layer: trainer]
                  -> wg = RayWorkerGroup(fused cia)            [layer: dispatch]
                    -> wg.FUSE({"actor","critic","ref"})       [layer: dispatch]
                      -> self.spawn(prefix_set) if needed      [layer: dispatch]
                        -> self.spawn_fused(...)               [layer: dispatch]
                      -> setattr(self, role, role_wg) per role [layer: dispatch]
                      -> _bind_worker_method(cls, func_generator)
                                                               [layer: dispatch]

        Branches:
          - self.wg_dict is None
              -> call self.spawn(prefix_set) first to populate it.
          - self.wg_dict already set
              -> skip spawn, only re-bind top-level methods.

        Why:
          - Symmetric helper to spawn; neither creates new Ray actors.
          - The combination (per-role attrs + top-level methods) is what makes
            HybridFlow §4's multi-role-per-actor pattern ergonomic at the trainer
            call-site: one object, two usage styles.
          - Binding the top-level FusedWorker methods means cross-role coordination
            calls (e.g. weight sync between actor and rollout) can be issued without
            first picking a role view.

        Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        if self.fused_worker_used and method_name not in self.method_names:
            remote_call = getattr(worker, self.fused_worker_execute_fn_name)
            return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
        # fused worker not used
        remote_call = getattr(worker, method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs)
                    )
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size


"""
Utilities that enables creating workers inside the same ray.Actor,
with code written in separate ray.Actors.
"""


# deprecated, switching to FusedWorker
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name, key=key):
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                async def async_func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                return wrapper

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR)
            setattr(func, MAGIC_ATTR, attrs)
            try:
                # bind direct rollout method to class without prefix
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    assert not hasattr(cls, method_name), (
                        f"conflict direct rollout method {method_name} with role {key}"
                    )
                    setattr(cls, method_name, func)
                    print(f"bind role {key} method {method_name} to class {cls}")
                else:
                    method_name_with_prefix = key + "_" + method_name
                    setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def _determine_fsdp_megatron_base_class(mros: list):
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker":
            return cls
        if cls.__name__ == "Worker":
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_fsdp_megatron_base_class(
        [cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()]
    )
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    print(f"colocated worker base class {worker_cls}")

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls


FusedWorkerCLSName = "FusedWorker"


def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    class FusedWorker(Worker):
        """What:
          - One Ray actor hosting N user worker instances (e.g. ActorWorker +
            CriticWorker + RefWorker) in a single process, sharing the same GPU.
          - Exposes _fuw_execute to route method calls by "{cls_name}_fwmn_{method_name}"
            prefix (fwmn = "fused worker method name").
          - Sub-workers are instantiated locally under DISABLE_WORKER_INIT=1 so
            __init__ work happens once per actor, not once per role.
          - Cross-injects fused_worker_dict into every sub-worker so they can see each
            other (e.g. rollout reading actor weights directly from the same process).

        Lifecycle:
          - Instantiated inside a Ray actor process when the colocated actor is brought up.
          - Lives for the full training job (or detached lifetime if configured).
          - Sub-workers are constructed eagerly in FusedWorker.__init__ (no lazy init).

        Called by:
          - create_colocated_worker_raw_cls (this file) renames this class to
            "FusedWorker_Actor_Critic_..." and returns the renamed type.
          - create_colocated_worker_cls_fused wraps the renamed type in a
            RayClassWithInitArgs with fused_worker_used=True.
          - RayWorkerGroup.__init__ then materializes N physical Ray actors from it.

        Call graph:
          bash/entry_point.sh                                  [layer: launch shell]
            -> verl/trainer/main_ppo.py::main                  [layer: entry]
              -> TaskRunner.remote(...)                        [layer: driver actor]
                -> RayPPOTrainer.init_workers()                [layer: trainer]
                  -> create_colocated_worker_cls_fused(...)    [layer: dispatch]
                    -> create_colocated_worker_raw_cls(...)    [layer: dispatch]
                      -> FUSEDWORKER (class def)               [layer: dispatch]
                  -> RayWorkerGroup(cia)                       [layer: dispatch]
                    -> ray_cls_with_init(pg, bundle_idx)       [layer: ray rpc]
                      -> FusedWorker.__init__ on actor proc    [layer: workers]
                        -> udc(*args, **kwargs) per sub-role   [layer: workers]
                        -> setattr(worker, fused_worker_attr_name, dict)
                                                               [layer: workers]
                  -> wg.method(...) from trainer               [layer: dispatch]
                    -> _execute_remote_single_worker           [layer: dispatch]
                      -> actor._fuw_execute.remote(            [layer: ray rpc]
                           "{cls}_fwmn_{method}", *a, **kw)
                        -> FusedWorker._fuw_execute            [layer: workers]
                          -> self.fused_worker_dict[cls].method(...)
                                                               [layer: workers]

        Branches:
          - method call matches self.method_names (set by _bind_worker_method)
              -> goes through direct remote_call (top-level FusedWorker method).
          - method call does not match
              -> routed via _fuw_execute with "{cls_name}_fwmn_{method_name}" key.
          - sub-worker __init__
              -> DISABLE_WORKER_INIT=1 shortcut skips distributed init; FusedWorker.
                __init__ itself handles the actual dist setup once.

        Why:
          - This is HybridFlow §4's 3D-HybridEngine colocation primitive: actor,
            rollout, and reference models live in one GPU process so weights/KV cache
            can share memory and device communicators, without the OS-level overhead
            of N separate actors.
          - The "{cls}_fwmn_{method}" naming is the routing key that lets a single
            Ray actor disambiguate which inner worker should handle a call without
            needing separate Ray method registrations per role.
          - fused_worker_dict injection into each sub-worker is what unlocks
            same-process weight transfer paths (e.g. rollout engine reading actor
            weights by direct attribute access, bypassing NCCL).
          - Overriding _get_ray_actor_cls_name / _get_ray_method_prefix on each udc
            at construction time keeps the @register dispatch logic consistent
            whether the underlying worker is standalone or fused.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_names = cls_names
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict

            for cls_name, udc, ud_args, ud_kwargs in zip(
                self.cls_names,
                self.raw_cls_dict.values(),
                self.init_args_dict.values(),
                self.init_kwargs_dict.values(),
                strict=True,
            ):
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            for _, worker in self.fused_worker_dict.items():
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            names = method_name.split("_fwmn_")
            cls_name = names[0]
            method_name = names[1]

            assert cls_name in self.fused_worker_dict, (
                f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            )
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            return udc_method(*args, **kwargs)

    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    renamed_fused_worker_cls.is_fused_worker = True
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    return renamed_fused_worker_cls


def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    """What:
      - Public factory. Takes {role_name: RayClassWithInitArgs} and returns one
        RayClassWithInitArgs whose cls is the fused Ray-remote wrapper.
      - Sets fused_worker_used=True on the returned cia so RayWorkerGroup enters
        the FusedWorker code paths (method dispatch via _fuw_execute, spawn/fuse
        via raw_cls_dict).
      - Output cia is ready to be consumed by a single RayWorkerGroup.

    Lifecycle:
      - Trainer calls this during role-graph setup to materialize colocation.
      - The resulting cia is then passed to a single RayWorkerGroup constructor.
      - Per-role handles come from .spawn(prefix_set) or .fuse(prefix_set) on that
        WorkerGroup after construction.

    Called by:
      - verl/trainer/ppo/ray_trainer.py when the role layout maps multiple roles
        to the same physical GPU.
      - Other trainers (sync PPO, diffusion, SFT co-scheduled variants) with
        colocated layouts.

    Call graph:
      bash/entry_point.sh                                    [layer: launch shell]
        -> verl/trainer/main_ppo.py::main                    [layer: entry]
          -> TaskRunner.remote(...)                          [layer: driver actor]
            -> RayPPOTrainer.init_workers()                  [layer: trainer]
              -> CREATE_COLOCATED_WORKER_CLS_FUSED(          [layer: dispatch]
                   {"actor": cia_a, "critic": cia_c, ...})
                -> create_colocated_worker_raw_cls(...)      [layer: dispatch]
                  -> FusedWorker class def                   [layer: dispatch]
                -> ray.remote(raw_cls)                       [layer: ray rpc]
                -> RayClassWithInitArgs(cls=remote_cls)      [layer: dispatch]
                  -> cia.fused_worker_used = True
              -> RayWorkerGroup(resource_pool, cia)          [layer: dispatch]
                -> FusedWorker actor per GPU bundle          [layer: workers]

    Branches:
      - none at this function level (single pipeline: raw_cls -> ray.remote -> cia).
      - downstream branching happens in RayWorkerGroup/FusedWorker based on
        fused_worker_used=True.

    Why:
      - Preferred replacement for the deprecated create_colocated_worker_cls.
      - Keeps one Ray actor per GPU bundle (1:1 actor:GPU) while still giving the
        trainer N independent role APIs — the trade-off is slightly more complex
        method routing (the _fwmn_ naming convention) in exchange for single-process
        memory sharing.
      - Single-process memory sharing is what makes HybridFlow §4 3D-HybridEngine
        practical on commodity GPUs: without it, actor+rollout+ref on one 80GB GPU
        would duplicate parameters and CUDA contexts N times and OOM.
      - Returning a RayClassWithInitArgs (rather than a raw class) keeps the call
        site symmetric with non-colocated paths: trainer code treats all roles
        uniformly as cia -> RayWorkerGroup.

    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict)

    remote_cls = ray.remote(raw_colocated_worker_cls)
    cia = RayClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True

    return cia
