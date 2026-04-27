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
from functools import partial, wraps
from types import FunctionType

from verl.protocol import DataProtoFuture, _padding_size_key
from verl.utils.py_functional import DynamicEnum
from verl.utils.transferqueue_utils import tqbridge

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    """What:
          - Enum class defining different dispatch modes for distributed computation.
          - Each mode represents a specific strategy for distributing data across
            different ranks in a distributed system.
          - DynamicEnum of scatter/gather strategies naming how a trainer call fans out to N workers.
          - Each member is just a tag; the actual dispatch_fn/collect_fn pair lives in DISPATCH_MODE_FN_REGISTRY.

    Lifecycle:
          - Declared at import time with an empty _registry; populated by init_predefined_dispatch_mode() below.
          - Each @register(dispatch_mode=...) stamps the tag into a worker method's MAGIC_ATTR at class-definition time.
          - WorkerGroup._bind_worker_method later resolves the tag against the registry to install a driver-side stub.

    Called by:
          - @register in this file (dispatch_mode kwarg default / explicit overrides).
          - Worker method decorators in verl/workers/fsdp_workers.py, verl/workers/megatron_workers.py,
            verl/workers/reward_manager/{naive,prime,dapo,batch}.py.
          - Experimental agent/reward loops in verl/experimental/agent_loop/, verl/experimental/reward_loop/.
          - Downstream extensions: checkpoint_engine, tqbridge, separation/engine_workers.

    Call graph:
      launch.sh [layer: launch shell]
        -> main_ppo.py [layer: entry]
           -> TaskRunner (ray_trainer driver actor) [layer: driver actor]
              -> RayPPOTrainer.fit / init_workers [layer: trainer]
                 -> worker_group.method(batch)  [layer: dispatch]
                    -> DISPATCH ENUM TAG (via MAGIC_ATTR on the method)
                       -> DISPATCH (Dispatch.<MODE>)
                          -> DISPATCH_MODE_FN_REGISTRY[MODE] -> (dispatch_fn, collect_fn) [layer: dispatch]
                             -> ray actor .remote() per rank [layer: ray rpc]
                                -> Worker.method body (fsdp_workers / megatron_workers / reward_manager) [layer: workers]

    Branches (core modes used across workers):
          - ONE_TO_ALL            -> replicate same arg to every rank (load_model, init_weights, broadcast configs).
          - ALL_TO_ALL            -> pass args through unchanged; caller has already produced a per-rank tuple/list.
          - DP_COMPUTE_PROTO      -> THE workhorse: chunk a DataProto across world_size with auto-padding, concat results.
          - DP_COMPUTE_PROTO_WITH_FUNC -> same split semantics but args[0] is a closure replicated to every rank.
          - DP_COMPUTE / DP_COMPUTE_METRIC -> pre-chunked list-of-N inputs; collect returns list (metrics) or concat.
          - RANK_ZERO (via Execute.RANK_ZERO) -> driver-only I/O (ckpt save, logging) — not dispatch but execute-side.
          - DIRECT_ROLLOUT_METHOD -> escape hatch for vLLM ExternalRayDistributedExecutor; dispatch is forbidden.

    Why:
          - HybridFlow §3.2 models data-flow as a composition of dispatch modes; the enum gives each composition a name.
          - Using DynamicEnum (not a fixed enum.Enum) lets downstream packages (checkpoint_engine, tqbridge,
            experimental/) register new modes without editing this file — the dispatch strategy is pluggable data,
            not hard-coded control flow.
          - The tag-vs-function separation is what keeps Worker classes backend-agnostic: the same @register'd method
            runs under Ray today and could run under a different scheduler tomorrow by swapping the registry entry.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import BatchData

    splitted_args = []
    for arg in args:
        assert BatchData(arg).is_chunkable(), f"arg of type {type(arg)} is not chunkable"
        chunked_arg = BatchData(arg).chunk(chunks=chunks)
        assert len(chunked_arg) == chunks
        splitted_args.append(chunked_arg)

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert BatchData(val).is_chunkable(), f"kwarg '{key}' of type {type(val)} is not chunkable"
        chunked_kwarg = BatchData(val).chunk(chunks=chunks)
        assert len(chunked_kwarg) == chunks
        splitted_kwargs[key] = chunked_kwarg

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    """What:
          - Chunk every DataProto arg/kwarg into `chunks` shards after padding length up to a multiple of `chunks`.
          - Returns (splitted_args, splitted_kwargs); splitted_kwargs carries the padding size under _padding_size_key
            so the collect side can slice the concatenated output back to the original length.

    Lifecycle:
          - Dispatch-side helper for DP_COMPUTE_PROTO / DP_COMPUTE_METRIC.
          - Runs on the driver right before RPCs fan out, inside dispatch_dp_compute_data_proto.
          - Padding size is computed once from the first DataProto arg and reused for every subsequent arg/kwarg in the call.

    Called by:
          - dispatch_dp_compute_data_proto (directly).
          - Transitively DISPATCH_MODE_FN_REGISTRY[DP_COMPUTE_PROTO] / [DP_COMPUTE_METRIC] dispatch_fn.

    Call graph:
      trainer (ray_trainer.py: RayPPOTrainer.fit) [layer: trainer]
        -> worker_group.<method>(proto_batch)  [layer: dispatch]
           -> Functor from func_generator [layer: dispatch]
              -> dispatch_dp_compute_data_proto(worker_group, *args, **kwargs) [layer: dispatch]
                 -> _SPLIT_ARGS_KWARGS_DATA_PROTO_WITH_AUTO_PADDING(world_size, *args, **kwargs)
                    -> DataProto.padding(padding_size=...) + DataProto.chunk(chunks=world_size) per arg
                       -> ray actor .remote() per rank [layer: ray rpc]
                          -> Worker.method body receives its shard + _padding_size_key kwarg [layer: workers]
                             -> collect_dp_compute_data_proto strips padding on the way back [layer: dispatch]

    Branches:
          - obj is DataProto with padding enabled -> compute padding once from first proto, reuse; pad then chunk.
          - obj is DataProto with padding disabled / DataProtoFuture -> chunk directly (caller guarantees divisibility).
          - padding_size resolved > 0 -> inject _padding_size_key into kwargs so the collect side can strip the tail.
          - Length mismatch between DataProto args -> assert; all padded args in a single call must share length.

    Why:
          - Without padding, a batch of N %% world_size != 0 leaves the last rank with a short shard, skewing load
            and breaking allreduce sizes during collective ops inside the worker body.
          - Padding metadata travels in kwargs instead of a side channel so the collect path
            (collect_dp_compute_data_proto) can slice the concatenated output back to the original length — HybridFlow §3.2.
          - Computing padding_size once from the first arg (not per-arg) is what lets multi-arg calls stay aligned across
            ranks; the nonlocal closure is the cheapest way to implement that without threading state through a class.
    """
    from verl.protocol import DataProto, DataProtoFuture

    data_proto_len = None
    padding_size = None

    def _padding_and_split_data(obj, chunks):
        nonlocal data_proto_len, padding_size
        assert isinstance(obj, DataProto | DataProtoFuture)
        if isinstance(obj, DataProto) and obj.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(obj)
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
            else:
                assert data_proto_len == len(obj), (
                    f"expecting all arg share same length of {data_proto_len}, but got {len(obj)}"
                )
            obj.padding(padding_size=padding_size)
        return obj.chunk(chunks=chunks)

    splitted_args = [_padding_and_split_data(arg, chunks) for arg in args]
    splitted_kwargs = {key: _padding_and_split_data(val, chunks) for key, val in kwargs.items()}
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def _concat_data_proto_or_future(output: list):
    from verl.protocol import BatchData

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    return BatchData(output).concat()


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_nd_compute(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    import os

    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)

    max_workers = max(1, min(len(args[0]), os.cpu_count()))

    args = [parallel_put(arg, max_workers=max_workers) for arg in args]
    kwargs = {k: parallel_put(v, max_workers=max_workers) for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_nd_compute(collect_mask: list[bool], worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        collect_dp_rank = collect_mask[global_rank]
        if collect_dp_rank:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_nd_compute_dataproto(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *splitted_args, **splitted_kwargs)


def collect_nd_compute_dataproto(collect_mask: list[bool], worker_group, output):
    output = collect_nd_compute(collect_mask, worker_group, output)

    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )
    return _concat_data_proto_or_future(output)


def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # query dispatch info of the worker group
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    # perform dispatch
    dp_size = max(dp_rank_mapping) + 1
    return dispatch_nd_compute_dataproto(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # the dispatch info is stored in the worker group
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    # a boolean of whether the dp_rank is used for collect
    collect_mask = worker_group._collect_info[mesh_name]
    # perform dispatch
    return collect_nd_compute_dataproto(collect_mask, worker_group, *args, **kwargs)


def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    return {
        "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


# What:
#   - Strategy table mapping Dispatch enum members -> {"dispatch_fn": ..., "collect_fn": ...} pairs.
#   - The single source of truth that turns a Dispatch tag (on a @register'd method) into actual
#     scatter (driver -> workers) and gather (workers -> driver) callables.
#
# Lifecycle:
#   - Built at module import after init_predefined_dispatch_mode() and all dispatch/collect helpers are defined.
#   - Read once per method by WorkerGroup._bind_worker_method via get_predefined_dispatch_fn at WorkerGroup
#     construction, which happens when the trainer calls RayWorkerGroup(...) during init_workers.
#   - Mutated at runtime by register_dispatch_mode (add new mode) and update_dispatch_mode (replace pair);
#     callers include checkpoint_engine/base.py, tqbridge (in verl/utils/transferqueue_utils.py), experimental/.
#
# Called by:
#   - get_predefined_dispatch_fn (this file) -> WorkerGroup._bind_worker_method (worker_group.py L206).
#   - register_dispatch_mode / update_dispatch_mode (this file) -> external extension code.
#   - tests/single_controller/base/test_decorator.py snapshots it for fixture isolation.
#
# Call graph:
#   (import time) init_predefined_dispatch_mode() [layer: entry]
#     -> Dispatch.register("ONE_TO_ALL") ... Dispatch.register("DIRECT_ROLLOUT_METHOD")
#        -> DISPATCH_MODE_FN_REGISTRY = {...}  (populated here)
#
#   (bind time) RayPPOTrainer.init_workers [layer: trainer]
#     -> RayWorkerGroup.__init__ (verl/single_controller/ray/base.py) [layer: driver actor]
#        -> self._bind_worker_method(cls, func_generator) [layer: dispatch]
#           -> get_predefined_dispatch_fn(method.MAGIC_ATTR.dispatch_mode) [layer: dispatch]
#              -> DISPATCH_MODE_FN_REGISTRY[Dispatch.<MODE>]  (reads THIS ENTITY)
#                 -> func_generator(dispatch_fn, collect_fn, ...) -> Functor installed on the group
#                    -> ray actor .remote() per rank at call time [layer: ray rpc]
#                       -> Worker.method body [layer: workers]
#
#   (runtime extension fan-out):
#     register_dispatch_mode / update_dispatch_mode -> mutates DISPATCH_MODE_FN_REGISTRY in place
#     checkpoint_engine, tqbridge, experimental/* -> call the above to plug new (dispatch_fn, collect_fn) pairs
#
# Branches (pair choices worth noting):
#   - ONE_TO_ALL pairs with collect_all_to_all (pass-through), because replicate-in rarely needs a reduce-out;
#     callers that do aggregate use RANK_ZERO's return (Execute.RANK_ZERO) instead.
#   - DP_COMPUTE_METRIC reuses dispatch_dp_compute_data_proto but pairs with collect_dp_compute (list, no concat),
#     so metrics stay a per-rank list rather than being concatenated into one DataProto.
#   - DIRECT_ROLLOUT_METHOD installs dummy_direct_rollout_call on both sides; calling it raises — the mode exists
#     only so vLLM's ExternalRayDistributedExecutor can own dispatch itself and verl stays out of the way.
#
# Why:
#   - HybridFlow §3.2: data-flow is a composition of dispatch modes. Keeping the (dispatch, collect) pair as
#     registry data (not a method on Dispatch, not a subclass) means adding a new fan-out pattern never requires
#     touching worker code or this enum's definition: plug a pair into the registry and @register(dispatch_mode=...)
#     picks it up at the next WorkerGroup bind.
#   - Registry indirection is also how verl decouples the trainer from the scheduler backend — a future non-Ray
#     executor only needs to ship its own func_generator and rely on the same registry contract.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(dispatch_mode, Dispatch | dict), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    """What:
          - Register a function with distributed execution configuration.
          - Decorator factory that tags a worker method with (dispatch_mode, execute_mode, blocking) attributes.
          - Wraps the method so any DataProtoFuture args are .get()'d before the user body runs (when materialize_futures).
          - The tag is stashed on the wrapper under MAGIC_ATTR so the WorkerGroup binding layer can find it by reflection.
          - Handles both synchronous and asynchronous functions, and optionally materializes futures before execution.

    Lifecycle:
          - Applied at class-definition time on Worker subclasses: verl/workers/fsdp_workers.py,
            verl/workers/megatron_workers.py, verl/workers/reward_manager/*, verl/experimental/agent_loop/*,
            verl/experimental/reward_loop/*, verl/experimental/vla/*, verl/experimental/separation/engine_workers.py.
          - At WorkerGroup construction, WorkerGroup._bind_worker_method walks the class, reads MAGIC_ATTR, looks up
            (dispatch_fn, collect_fn) in DISPATCH_MODE_FN_REGISTRY, then calls func_generator to install a driver-side
            stub (Functor) that implements scatter -> ray .remote() fan-out -> gather.
          - blocking=False survives into the Functor, which uses it to choose sync vs async dispatch to Ray actors.

    Called by:
          - All Worker subclass authors across verl/workers/* and verl/experimental/* (dozens of call sites, see grep
            for @register(). Covers fsdp_workers, megatron_workers, reward managers, agent/reward loops, VLA workers).
          - utils/profiler/profile.py uses it to wrap profile_trace calls.
          - MAGIC_ATTR it writes is consumed by WorkerGroup._bind_worker_method (worker_group.py L206) and by
            RayWorkerGroup in verl/single_controller/ray/base.py (which also re-stamps MAGIC_ATTR onto the outer
            stub so nested/fused groups can see the same tag).

    Call graph:
      launch.sh [layer: launch shell]
        -> main_ppo.py [layer: entry]
           -> TaskRunner (ray_trainer driver actor) [layer: driver actor]
              -> RayPPOTrainer.init_workers / RayPPOTrainer.fit [layer: trainer]
                 -> worker_group.method(batch, ...)  [layer: dispatch]
                    -> Functor from func_generator (verl/single_controller/ray/base.py:func_generator) [layer: dispatch]
                       -> dispatch_fn from DISPATCH_MODE_FN_REGISTRY[MAGIC_ATTR.dispatch_mode] [layer: dispatch]
                          -> ray_actor.method.remote(shard_i) per rank [layer: ray rpc]
                             -> REGISTER-WRAPPED WORKER METHOD (inner/async_inner from this decorator)
                                -> _materialize_futures (resolves DataProtoFuture -> DataProto)
                                   -> user func body (fsdp_workers / megatron_workers / reward_manager) [layer: workers]
                                -> return value bubbles back through ray.get
                          -> collect_fn from DISPATCH_MODE_FN_REGISTRY (concat / list / pass-through) [layer: dispatch]

    Branches:
          - inspect.iscoroutinefunction(func) -> install async_inner (awaits func); lets rollout/agent workers stay async.
          - materialize_futures=True (default) -> _materialize_futures resolves DataProtoFuture before the body runs,
            turning the trainer's lazy RPC graph into concrete data at the worker boundary.
          - materialize_futures=False -> pass futures through unchanged (caller consumes them, e.g. chained pipeline ops).
          - blocking flag is NOT acted on here -> it rides in MAGIC_ATTR and is honored by the WorkerGroup stub that
            chooses sync ray.get() vs async ObjectRef return when invoking Ray actors.
          - tqbridge(dispatch_mode=...) wraps func first -> gives TransferQueue a chance to intercept / re-route data
            for the modes it understands; transparent when tqbridge is disabled.

    Why:
          - HybridFlow §3 single-controller/multi-worker: this decorator IS the trainer<->worker boundary. The user
            writes a normal-looking method on Worker; @register + MAGIC_ATTR + func_generator together synthesize the
            scatter-gather RPC so trainer code can call worker_group.method(proto) and get a DataProto back without
            writing dispatch logic by hand.
          - Decoupling the tag (written here) from the binding (done in WorkerGroup) is what lets the same Worker
            class run under Ray today and, in principle, under a non-Ray backend tomorrow — only func_generator and
            the registry entries need to change, never the Worker class.
          - Materializing futures at the wrapper (not at the call site) keeps user code free of DataProto vs
            DataProtoFuture branching; the trainer composes ops lazily and only the wrapper forces evaluation.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.


    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """

    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        func = tqbridge(dispatch_mode=dispatch_mode)(func)

        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
