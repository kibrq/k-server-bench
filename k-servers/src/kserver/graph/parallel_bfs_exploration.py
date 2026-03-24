from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple
import multiprocessing as mp
import gc
from multiprocessing.connection import wait as mp_wait

import numpy as np
from kserver.context.numpy_wf_context import WFContext


HashFn = Callable[[Any], tuple[Hashable, Dict[str, Any]]]
TransitionFn = Callable[[WFContext, Any], Any]
CreateHashFn = Callable[[WFContext], HashFn]
Hook = Callable[..., Any]
HookConstructor = Callable[[Any], Hook]


class HookManager:
    def __init__(self, owner: Any, hooks: Optional[Dict[str, List[HookConstructor]]] = None) -> None:
        self.owner = owner
        self.hooks: Dict[str, List[Hook]] = {}
        for event_name, constructors in (hooks or {}).items():
            self.hooks[event_name] = [ctor(owner) for ctor in constructors]

    def fire_event(self, event_name, *args, **kwargs) -> None:
        for hook in self.hooks.get(event_name, []):
            hook(*args, **kwargs)


class WorkerPool(ABC):
    @abstractmethod
    def get(self) -> list[Any]:
        pass

    @abstractmethod
    def queue_size(self) -> int:
        pass

    @abstractmethod
    def push_for_expansion(self, nodes: list["Node"]) -> None:
        pass

    @abstractmethod
    def push_for_initialization(self, nodes: list["Node"]) -> None:
        pass

    def close(self) -> None:
        return None



@dataclass(slots=True)
class WFPath:
    initial_cfg: Tuple
    requests: Tuple = field(default_factory=tuple)

    def iter_requests(self):
        for step in self.requests:
            yield step

    def get_requests(self) -> Tuple:
        return tuple(self.requests)

    def get_depth(self) -> int:
        return len(self.requests)

    def copy(self) -> "WFPath":
        return WFPath(initial_cfg=tuple(self.initial_cfg), requests=tuple(self.requests))

    def with_appended_request(self, request: Any) -> "WFPath":
        return WFPath(
            initial_cfg=tuple(self.initial_cfg),
            requests=tuple(self.requests) + (request,),
        )

    @property
    def initial(self) -> Tuple:
        return self.initial_cfg


def resolve_wf(
    wf: Optional[np.ndarray] = None,
    wf_path: Optional[WFPath] = None,
    context: Optional[WFContext] = None,
    transitions: Optional[list[TransitionFn]] = None,
) -> np.ndarray:
    if wf is not None:
        return wf

    if wf_path is None:
        raise ValueError("Either wf or wf_path must be provided")

    if context is None or transitions is None:
        raise ValueError("context and transitions are required to reconstruct wf from wf_path")

    out = context.initial_wf(tuple(wf_path.initial_cfg))
    for step in wf_path.iter_requests():
        if callable(step):
            out = step(context, out)
        else:
            out = transitions[step](context, out)
    return out


@dataclass(slots=True)
class Node:
    hsh: Hashable
    wf: Optional[np.ndarray] = None
    path: Optional[WFPath] = None
    metadata: Optional[Dict] = field(default_factory=dict)

    def get_wf(
        self,
        context: WFContext = None,
        transitions: list[TransitionFn] = None,
        cache: bool = False,
    ):
        out = resolve_wf(
            wf=self.wf,
            wf_path=self.path,
            context=context,
            transitions=transitions,
        )
        if cache:
            self.wf = out
            return self.wf
        return out


class NodeBookkeeper(ABC):
    @abstractmethod
    def add(self, u: Node) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class EdgeBookkeeper(ABC):
    @abstractmethod
    def add(self, u: Node, v: Node, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class InMemoryNodeBookkeeper(NodeBookkeeper):
    def __init__(self) -> None:
        self.nodes: dict[Hashable, dict[str, Any]] = {}

    def add(self, u: Node) -> bool:
        if u.hsh in self.nodes:
            return False
        self.nodes[u.hsh] = u
        return True

    def __len__(self) -> int:
        return len(self.nodes)


class InMemoryEdgeBookkeeper(EdgeBookkeeper):
    def __init__(self) -> None:
        self.edges: list[tuple[Node, Node, dict[str, Any]]] = []

    def add(self, u: Node, v: Node, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append((u, v, dict(metadata or {})))

    def __len__(self) -> int:
        return len(self.edges)



@dataclass(slots=True)
class ExpandResult:
    u: Node
    neighbors: List["ExpandNeighbor"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpandNeighbor:
    v: Node
    metadata: dict[str, Any] = field(default_factory=dict)


class Worker:
    def __init__(
        self,
        k: int,
        distance_matrix,
        transitions: list[TransitionFn],
        create_hash_fn: CreateHashFn,
        return_wfs: bool = True,
        return_paths: bool = True,
        hooks: Optional[Dict[str, List[HookConstructor]]] = None,
        normalize_wf: bool = True,
    ) -> None:
        self.k = k
        self.distance_matrix = distance_matrix
        self.transitions = transitions
        self.create_hash_fn = create_hash_fn
        
        self.return_wfs = bool(return_wfs)
        self.return_paths = bool(return_paths)
        self.hook_constructors = hooks
        self.normalize_wf = bool(normalize_wf)
        self.context = WFContext(k=k, distance_matrix=distance_matrix)
        self.hash_fn = create_hash_fn(self.context)
        
        self.hook_manager = HookManager(self, hooks)

    
    def initialize(self, node: Node) -> Node:
        self.hook_manager.fire_event("init_node.before", node=node)
        wf = node.get_wf(self.context, self.transitions, cache=self.return_wfs)
        hsh, hash_metadata = self.hash_fn(wf)
        out = Node(
            hsh=hsh,
            wf=wf if self.return_wfs else None,
            path=(node.path.copy() if (self.return_paths and node.path is not None) else None),
            metadata={**dict(hash_metadata or {}), **dict(node.metadata or {})},
        )
        self.hook_manager.fire_event("init_node.after", node=out)
        return out

    def expand(self, u: Node) -> ExpandResult:
        self.hook_manager.fire_event("expand.start", node=u)

        if u.hsh is None:
            u = self.initialize(u)

        uwf = u.get_wf(self.context, self.transitions, cache=self.return_wfs)
        if self.normalize_wf:
            uwf = uwf - uwf.min()

        neighbors: List[ExpandNeighbor] = []
        for transition_idx, transition in enumerate(self.transitions):

            vwf = transition(self.context, uwf)
            vhash, vhash_metadata = self.hash_fn(vwf)

            vpath = (
                u.path.with_appended_request(transition_idx)
                if u.path is not None
                else None
            )

            v = Node(
                hsh=vhash,
                wf=vwf if self.return_wfs else None,
                path=vpath if self.return_paths else None,
                metadata=vhash_metadata,
            )
            edge_meta: dict[str, Any] = {}
            self.hook_manager.fire_event(
                "expand.after_transition",
                u=u,
                uwf=uwf,
                v=v,
                vwf=vwf,
                vpath=vpath,
                is_new_node=True,
                transition_idx=transition_idx,
                transition=transition,
                edge_meta=edge_meta,
            )
            out_meta = dict(v.metadata or {})
            out_meta.update(edge_meta)
            out_v = Node(
                hsh=v.hsh,
                wf=vwf if self.return_wfs else None,
                path=(v.path.copy() if (self.return_paths and v.path is not None) else None),
                metadata=out_meta,
            )
            neighbors.append(ExpandNeighbor(v=out_v, metadata=edge_meta))

        result = ExpandResult(u=u, neighbors=neighbors)
        self.hook_manager.fire_event("expand.end", result=result)
        return result

    def expand_many(self, nodes: list[Node]) -> list[ExpandResult]:
        return [self.expand(node) for node in nodes]


class OneWorkerPool(WorkerPool):
    def __init__(self, **worker_kwargs) -> None:
        self.worker = Worker(**worker_kwargs)
        self.queue: deque[tuple[str, Node]] = deque()

    def get(self) -> list[Any]:
        if not self.queue:
            return []
        task, payload = self.queue.popleft()
        if task == "expand":
            return [self.worker.expand(payload)]
        if task == "initialize":
            return [self.worker.initialize(payload)]
        raise ValueError(f"Invalid task: {task}")

    def queue_size(self) -> int:
        return len(self.queue)

    def push_for_expansion(self, nodes: list[Node]) -> None:
        self.queue.extend(("expand", node) for node in nodes)

    def push_for_initialization(self, nodes: list[Node]) -> None:
        self.queue.extend(("initialize", node) for node in nodes)


def _subprocess_worker_main(
    task_queue: Any,
    result_queue: Any,
    worker_kwargs: dict[str, Any],
) -> None:
    worker = Worker(**worker_kwargs)
    while True:
        item = task_queue.get()
        if item is None:
            break
        task_kind, payload = item
        if task_kind == "initialize":
            out = worker.initialize(payload)
        elif task_kind == "expand":
            out = worker.expand_many(payload)
        else:
            raise ValueError(f"Invalid task: {task_kind}")
        result_queue.put((task_kind, out))


class SubprocessWorkerPool(WorkerPool):
    def __init__(
        self,
        n_workers: int,
        expand_batch_size: int = 1,
        **worker_kwargs,
    ) -> None:
        self.n_workers = n_workers
        self.expand_batch_size = max(1, int(expand_batch_size))
        

        self.task_queues: list[Any] = [mp.Queue() for _ in range(self.n_workers)]
        self.result_queues: list[Any] = [mp.Queue() for _ in range(self.n_workers)]
        self.queue: deque[tuple[str, Node]] = deque()
        self.inflight_tasks: dict[int, str] = {}
        self.free_workers: deque[int] = deque(range(self.n_workers))
        self._reader_to_wid: dict[Any, int] = {}
        self.processes: list[Any] = []
        for wid in range(self.n_workers):
            proc = mp.Process(
                target=_subprocess_worker_main,
                args=(self.task_queues[wid], self.result_queues[wid], worker_kwargs),
            )
            proc.start()
            self.processes.append(proc)
            self._reader_to_wid[self.result_queues[wid]._reader] = wid

    def _submit_until_full(self) -> int:
        submitted = 0
        while self.free_workers and self.queue:
            wid = self.free_workers.popleft()
            task_kind, payload = self.queue.popleft()
            if task_kind == "expand":
                batch: list[Node] = [payload]
                while self.queue and len(batch) < self.expand_batch_size:
                    next_kind, next_payload = self.queue[0]
                    if next_kind != "expand":
                        break
                    self.queue.popleft()
                    batch.append(next_payload)
                self.task_queues[wid].put(("expand", batch))
            elif task_kind == "initialize":
                self.task_queues[wid].put(("initialize", payload))
            else:
                raise ValueError(f"Invalid task: {task_kind}")
            self.inflight_tasks[wid] = task_kind
            submitted += 1
        return submitted

    def get(self) -> list[Any]:
        if not self.inflight_tasks:
            return []
        active_readers = [self.result_queues[wid]._reader for wid in self.inflight_tasks.keys()]
        ready_readers = mp_wait(active_readers, timeout=None)
        results: list[Any] = []
        for reader in ready_readers:
            wid = self._reader_to_wid[reader]
            if wid not in self.inflight_tasks:
                continue
            task_kind = self.inflight_tasks.pop(wid)
            msg = self.result_queues[wid].get()
            _kind, out = msg
            self.free_workers.append(wid)
            if task_kind == "expand":
                results.extend(out)
            else:
                results.append(out)
        self._submit_until_full()
        return results

    def queue_size(self) -> int:
        return len(self.queue) + len(self.inflight_tasks)

    def push_for_expansion(self, nodes: list[Node]) -> None:
        self.queue.extend(("expand", node) for node in nodes)
        self._submit_until_full()

    def push_for_initialization(self, nodes: list[Node]) -> None:
        self.queue.extend(("initialize", node) for node in nodes)
        self._submit_until_full()

    def close(self) -> None:
        self.queue.clear()
        self.inflight_tasks.clear()
        self.free_workers.clear()
        for task_queue in self.task_queues:
            task_queue.put(None)
        for proc in self.processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        for task_queue in self.task_queues:
            try:
                task_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                task_queue.close()
            except Exception:
                pass
        for result_queue in self.result_queues:
            try:
                result_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                result_queue.close()
            except Exception:
                pass


class RayWorkerPool(WorkerPool):
    def __init__(
        self,
        n_workers: int,
        ray_kwargs: Optional[Dict[str, Any]] = None,
        expand_batch_size: int = 16,
        **worker_kwargs,
    ) -> None:
        try:
            import ray  # type: ignore
        except ImportError as exc:
            raise ImportError("RayWorkerPool requires the `ray` package.") from exc

        self.n_workers = n_workers
        self.ray_kwargs = ray_kwargs
        self.expand_batch_size = max(1, int(expand_batch_size))

        if not ray.is_initialized():
            ray.init(**(ray_kwargs or {}))

        self.ray = ray

        self.actor_cls = self.ray.remote(Worker).options(max_concurrency=1)
        self.actors = [self.actor_cls.remote(**worker_kwargs) for _ in range(self.n_workers)]

        self.inflight_tasks: dict[Any, tuple[int, str]] = {}
        self.free_actors = deque(range(self.n_workers))

        self.queue: deque[tuple[str, Node]] = deque()

    def _submit_until_full(self) -> None:
        submitted = 0
        while self.free_actors and self.queue:
            actor_id = self.free_actors.popleft()
            task, payload = self.queue.popleft()
            if task == "expand":
                batch: list[Node] = [payload]
                while self.queue and len(batch) < self.expand_batch_size:
                    next_task, next_payload = self.queue[0]
                    if next_task != "expand":
                        break
                    self.queue.popleft()
                    batch.append(next_payload)
                fut = self.actors[actor_id].expand_many.remote(batch)
            elif task == "initialize":
                fut = self.actors[actor_id].initialize.remote(payload)
            else:
                raise ValueError(f"Invalid task: {task}")
            self.inflight_tasks[fut] = (actor_id, task)
            submitted += 1
        return submitted

    def get(self) -> list[Any]:
        if len(self.inflight_tasks) == 0:
            return []

        done, not_done = self.ray.wait(list(self.inflight_tasks.keys()))
        done_meta: dict[Any, tuple[int, str]] = {}
        for fut in done:
            actor_id, task = self.inflight_tasks.pop(fut)
            done_meta[fut] = (actor_id, task)
            self.free_actors.append(actor_id)

        self._submit_until_full()

        results = []
        for fut in done:
            _, task = done_meta[fut]
            out = self.ray.get(fut)
            if task == "expand":
                results.extend(out)
            else:
                results.append(out)

        return results
        
    def queue_size(self) -> int:
        return len(self.queue) + len(self.inflight_tasks)

    def push_for_expansion(self, nodes: list[Node]) -> None:
        self.queue.extend(("expand", node) for node in nodes)
        self._submit_until_full()

    def push_for_initialization(self, nodes: list[Node]) -> None:
        self.queue.extend(("initialize", node) for node in nodes)
        self._submit_until_full()


def parallel_bfs_exploration(
    k: int,
    distance_matrix,
    initial_nodes,
    transitions: list[TransitionFn],
    create_hash_fn: CreateHashFn,
    n_workers: int = 1,
    return_wfs: bool = True,
    return_paths: bool = True,
    node_bookkeeper_constructor: Optional[Callable[[], NodeBookkeeper]] = None,
    edge_bookkeeper_constructor: Optional[Callable[[], EdgeBookkeeper]] = None,
    worker_hook_constructors: Optional[Dict[str, List[HookConstructor]]] = None,
    main_hook_constructors: Optional[Dict[str, List[HookConstructor]]] = None,
    ray_kwargs: Optional[Dict[str, Any]] = None,
    expand_batch_size: int = 1,
    pool_backend: str = "auto",
    disable_gc_during_run: bool = False,
):
    """
    Base skeleton for BFS over WF states.

    Parameters requested by user:
    - k
    - distance_matrix
    - initial_nodes
    - transitions(context, wf) -> new_wf
    - create_hash_fn(context) -> hash_fn(wf) returning (hash, metadata)
    """

    assert return_wfs or return_paths, "At least one of return_wfs or return_paths must be True"


    node_bookkeeper = (node_bookkeeper_constructor or InMemoryNodeBookkeeper)()
    edge_bookkeeper = (edge_bookkeeper_constructor or InMemoryEdgeBookkeeper)()

    worker_kwargs = dict(
        k=k,
        distance_matrix=distance_matrix,
        transitions=transitions,
        create_hash_fn=create_hash_fn,
        return_wfs=return_wfs,
        return_paths=return_paths,
        hooks=worker_hook_constructors,
    )
    main_state = {
        "node_bookkeeper": node_bookkeeper,
        "edge_bookkeeper": edge_bookkeeper,
    }
    hook_manager = HookManager(main_state, hooks=main_hook_constructors)
    hook_manager.fire_event("main.start", initial_nodes=tuple(initial_nodes), n_workers=n_workers)

    def handle_expand(result: ExpandResult) -> None:
        hook_manager.fire_event("main.handle_expand.start", result=result)

        new_nodes: list[Node] = []
        old_nodes: list[Node] = []
        new_hashes: set[Hashable] = set()

        is_new = node_bookkeeper.add(result.u)
        if is_new:
            hook_manager.fire_event("main.handle_expand.source_node.new_node", v=result.u)
            new_hashes.add(result.u.hsh)
        else:
            hook_manager.fire_event("main.handle_expand.source_node.existing_node", v=result.u)
            if result.u.hsh not in new_hashes:
                old_nodes.append(result.u)

        for neighbor in result.neighbors:
            v = neighbor.v
            is_new = node_bookkeeper.add(v)
            if is_new:
                hook_manager.fire_event("main.handle_expand.neighbor.new_node", v=v)
                new_nodes.append(v)
                new_hashes.add(v.hsh)
            else:
                hook_manager.fire_event("main.handle_expand.neighbor.existing_node", v=v)
                if v.hsh not in new_hashes:
                    old_nodes.append(v)

            edge_bookkeeper.add(result.u, v, neighbor.metadata)
            hook_manager.fire_event("main.handle_expand.edge_added", u=result.u, v=v, neighbor=neighbor)

        hook_manager.fire_event(
            "main.handle_expand.end",
            result=result,
            new_nodes=tuple(new_nodes),
            old_nodes=tuple(old_nodes),
            num_nodes=len(node_bookkeeper),
            num_edges=len(edge_bookkeeper),
        )
        return new_nodes, old_nodes

    if n_workers <= 1:
        worker_pool: WorkerPool = OneWorkerPool(**worker_kwargs)
        hook_manager.fire_event("main.worker_pool.created", pool_type="one", n_workers=1)
    else:
        backend = "ray" if pool_backend == "auto" else pool_backend
        if backend == "ray":
            worker_pool = RayWorkerPool(
                n_workers,
                ray_kwargs=ray_kwargs,
                expand_batch_size=expand_batch_size,
                **worker_kwargs,
            )
            hook_manager.fire_event("main.worker_pool.created", pool_type="ray", n_workers=n_workers)
        elif backend == "subprocess":
            worker_pool = SubprocessWorkerPool(
                n_workers,
                expand_batch_size=expand_batch_size,
                **worker_kwargs,
            )
            hook_manager.fire_event("main.worker_pool.created", pool_type="subprocess", n_workers=n_workers)
        else:
            raise ValueError(f"Unknown pool_backend: {pool_backend}")

    gc_was_enabled = False
    if disable_gc_during_run:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
    try:
        initial_nodes = [Node(hsh=None, path=WFPath(initial_cfg=tuple(cfg), requests=tuple())) for cfg in initial_nodes]

        hook_manager.fire_event("main.init.start", n_initial=len(initial_nodes))
        hook_manager.fire_event("main.init.before_push", n_initial=len(initial_nodes))
        worker_pool.push_for_initialization(initial_nodes)
        hook_manager.fire_event(
            "main.init.after_push",
            n_initial=len(initial_nodes),
            queue_size=worker_pool.queue_size(),
        )

        frontier, n_handled = [], 0
        while n_handled < len(initial_nodes):
            hook_manager.fire_event("main.init.before_get", queue_size=worker_pool.queue_size())
            nodes = worker_pool.get()
            hook_manager.fire_event(
                "main.init.batch",
                batch_size=len(nodes),
                handled=n_handled,
                n_initial=len(initial_nodes),
            )
            for node in nodes:
                n_handled += 1
                if node_bookkeeper.add(node):
                    frontier.append(node)
        hook_manager.fire_event(
            "main.init.end",
            n_initial=len(initial_nodes),
            n_unique=len(frontier),
        )

        hook_manager.fire_event("main.loop.before_push", batch_size=len(frontier), queue_size=worker_pool.queue_size())
        worker_pool.push_for_expansion(frontier)
        hook_manager.fire_event(
            "main.loop.after_push",
            batch_size=len(frontier),
            queue_size=worker_pool.queue_size(),
        )

        while worker_pool.queue_size() > 0:
            hook_manager.fire_event("main.loop.start", queue_size=worker_pool.queue_size())
            hook_manager.fire_event("main.loop.submitted", queue_size=worker_pool.queue_size())
            hook_manager.fire_event("main.loop.before_get", queue_size=worker_pool.queue_size())
            expand_results = worker_pool.get()
            hook_manager.fire_event(
                "main.loop.after_get",
                queue_size=worker_pool.queue_size(),
                batch_size=len(expand_results),
            )
            hook_manager.fire_event("main.loop.wait_done", expand_results=expand_results)

            for expand_result in expand_results:
                new_nodes, old_nodes = handle_expand(expand_result)
                hook_manager.fire_event("main.loop.handle_expand", expand_result=expand_result, new_nodes=new_nodes, old_nodes=old_nodes)
                hook_manager.fire_event("main.loop.before_push", batch_size=len(new_nodes), queue_size=worker_pool.queue_size())
                worker_pool.push_for_expansion(new_nodes)
                hook_manager.fire_event(
                    "main.loop.after_push",
                    batch_size=len(new_nodes),
                    queue_size=worker_pool.queue_size(),
                )

            hook_manager.fire_event("main.loop.end", queue_size=worker_pool.queue_size())
    finally:
        worker_pool.close()
        if disable_gc_during_run and gc_was_enabled:
            gc.enable()
            gc.collect()

    hook_manager.fire_event(
        "main.end",
        num_nodes=len(node_bookkeeper),
        num_edges=len(edge_bookkeeper),
    )
    hook_manager.fire_event(
        "main.finished",
        num_nodes=len(node_bookkeeper),
        num_edges=len(edge_bookkeeper),
    )

    return {
        "node_bookkeeper": node_bookkeeper,
        "edge_bookkeeper": edge_bookkeeper,
        "hook_manager": hook_manager,
    }
