"""Frame-driven behaviour-tree runtime for generated task intelligence.

The data service produces ordered tasks/subtasks, not executable robot code. This
module supplies the execution boundary: generated primitive names are resolved
through a registry, and every call to :meth:`BehaviourTreeRuntime.tick` advances
the tree once. Explicit trees support sequences, selectors, retry decorators,
parallel-all composites, conditions, and actions.

This module deliberately has no Isaac Sim imports, so its parsing and state
machine can be unit tested with normal Python. Isaac-specific adapters live in
``isaac_adapters.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Status(Enum):
    INVALID = "INVALID"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class StepSpec:
    name: str
    primitive: str
    inputs: dict[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    success_conditions: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    steps: tuple[StepSpec, ...]


@dataclass
class PrimitiveContext:
    """Information shared with a primitive for the duration of one step."""

    step: StepSpec
    blackboard: set[str]
    services: dict[str, Any]


class Primitive(Protocol):
    def initialise(self, context: PrimitiveContext) -> None: ...

    def tick(self, context: PrimitiveContext) -> Status: ...

    def terminate(self, context: PrimitiveContext, new_status: Status) -> None: ...


PrimitiveFactory = Callable[[StepSpec], Primitive]


class TreeNode:
    """Base class for executable behaviour-tree nodes."""

    symbol = "?"

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = Status.INVALID

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        raise NotImplementedError

    def reset(self, runtime: "BehaviourTreeRuntime") -> None:
        self.status = Status.INVALID

    def describe(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.symbol} {self.name}"]


class ActionNode(TreeNode):
    symbol = "→"

    def __init__(self, step: StepSpec) -> None:
        super().__init__(step.name)
        self.step = step
        self.primitive: Primitive | None = None

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        if self.status in (Status.SUCCESS, Status.FAILURE):
            return self.status
        missing = sorted(set(self.step.preconditions) - runtime.blackboard)
        if missing:
            return self._fail(runtime, f"missing preconditions: {', '.join(missing)}")

        if self.primitive is None:
            factory = runtime._primitives.get(self.step.primitive)
            if factory is None:
                available = ", ".join(sorted(runtime._primitives)) or "none"
                return self._fail(
                    runtime,
                    f"unregistered primitive {self.step.primitive!r}; registered: {available}",
                )
            try:
                self.primitive = factory(self.step)
                self.primitive.initialise(runtime._context(self.step))
                runtime.logger(f"[BT ACTION] start: {self.name} ({self.step.primitive})")
            except Exception as exc:
                return self._fail(runtime, f"primitive initialisation raised: {exc}")

        try:
            result = self.primitive.tick(runtime._context(self.step))
        except Exception as exc:
            return self._fail(runtime, f"primitive tick raised: {exc}")
        if not isinstance(result, Status):
            return self._fail(runtime, f"primitive returned invalid status: {result!r}")
        if result is Status.RUNNING:
            self.status = Status.RUNNING
            runtime.feedback = f"running: {self.name}"
            return self.status
        if result is not Status.SUCCESS:
            return self._fail(runtime, "primitive reported failure")

        self.primitive.terminate(runtime._context(self.step), Status.SUCCESS)
        self.primitive = None
        runtime.blackboard.update(self.step.success_conditions)
        runtime.step_index += 1
        self.status = Status.SUCCESS
        runtime.feedback = f"action complete: {self.name}"
        runtime.logger(f"[BT ACTION] SUCCESS: {self.name}")
        return self.status

    def reset(self, runtime: "BehaviourTreeRuntime") -> None:
        if self.primitive is not None:
            self.primitive.terminate(runtime._context(self.step), Status.INVALID)
            self.primitive = None
        super().reset(runtime)

    def _fail(self, runtime: "BehaviourTreeRuntime", reason: str) -> Status:
        if self.primitive is not None:
            try:
                self.primitive.terminate(runtime._context(self.step), Status.FAILURE)
            finally:
                self.primitive = None
        self.status = Status.FAILURE
        runtime.feedback = f"{self.name}: {reason}"
        runtime.logger(f"[BT ACTION] FAILURE: {runtime.feedback}")
        return self.status


class ConditionNode(TreeNode):
    symbol = "?"

    def __init__(self, name: str, fact: str, expected: bool = True) -> None:
        super().__init__(name)
        self.fact = fact
        self.expected = expected

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        actual = self.fact in runtime.blackboard
        self.status = Status.SUCCESS if actual is self.expected else Status.FAILURE
        runtime.logger(
            f"[BT CONDITION] {self.status.value}: {self.name} "
            f"(fact={self.fact!r}, actual={actual})"
        )
        return self.status


class CompositeNode(TreeNode):
    def __init__(self, name: str, children: Iterable[TreeNode]) -> None:
        super().__init__(name)
        self.children = tuple(children)
        if not self.children:
            raise ValueError(f"{name}: composite requires at least one child")
        self.index = 0

    def reset(self, runtime: "BehaviourTreeRuntime") -> None:
        for child in self.children:
            child.reset(runtime)
        self.index = 0
        super().reset(runtime)

    def describe(self, indent: str = "") -> list[str]:
        lines = super().describe(indent)
        for child in self.children:
            lines.extend(child.describe(indent + "  "))
        return lines


class SequenceNode(CompositeNode):
    symbol = "→"

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        while self.index < len(self.children):
            result = self.children[self.index].tick(runtime)
            if result is Status.SUCCESS:
                self.index += 1
                continue
            self.status = result
            return result
        self.status = Status.SUCCESS
        return self.status


class SelectorNode(CompositeNode):
    symbol = "?→"

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        while self.index < len(self.children):
            child = self.children[self.index]
            result = child.tick(runtime)
            if result is Status.FAILURE:
                runtime.logger(f"[BT SELECTOR] {self.name}: branch {self.index + 1} failed; trying fallback")
                self.index += 1
                continue
            self.status = result
            return result
        self.status = Status.FAILURE
        runtime.feedback = f"selector exhausted: {self.name}"
        return self.status


class RetryNode(TreeNode):
    symbol = "↻"

    def __init__(self, name: str, child: TreeNode, max_attempts: int) -> None:
        super().__init__(name)
        self.child = child
        self.max_attempts = max(1, int(max_attempts))
        self.attempt = 1

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        result = self.child.tick(runtime)
        if result is Status.FAILURE and self.attempt < self.max_attempts:
            runtime.logger(
                f"[BT RETRY] {self.name}: attempt {self.attempt}/{self.max_attempts} failed; retrying"
            )
            self.attempt += 1
            self.child.reset(runtime)
            self.status = Status.RUNNING
            return self.status
        self.status = result
        return result

    def reset(self, runtime: "BehaviourTreeRuntime") -> None:
        self.child.reset(runtime)
        self.attempt = 1
        super().reset(runtime)

    def describe(self, indent: str = "") -> list[str]:
        return super().describe(indent) + self.child.describe(indent + "  ")


class ParallelAllNode(CompositeNode):
    symbol = "⇉"

    def tick(self, runtime: "BehaviourTreeRuntime") -> Status:
        results = [child.tick(runtime) for child in self.children]
        if any(result is Status.FAILURE for result in results):
            self.status = Status.FAILURE
        elif all(result is Status.SUCCESS for result in results):
            self.status = Status.SUCCESS
        else:
            self.status = Status.RUNNING
        return self.status


def load_task_intelligence(path: str | Path) -> Any:
    """Load a generated task-intelligence JSON file."""

    with Path(path).expanduser().open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def normalize_task_intelligence(payload: Any) -> tuple[TaskSpec, ...]:
    """Accept both API-wrapped and raw generation output shapes."""

    if isinstance(payload, dict) and isinstance(payload.get("taskIntelligence"), (dict, list)):
        payload = payload["taskIntelligence"]

    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        items = payload["tasks"]
        if items and all(isinstance(item, dict) and _step_items(item) for item in items):
            tasks = tuple(_task_from_mapping(item, index) for index, item in enumerate(items, 1))
        else:
            tasks = (TaskSpec(_text(payload, ("task_name", "taskName", "name"), "Video task"),
                              tuple(_step_from_item(item, index) for index, item in enumerate(items, 1))),)
    elif isinstance(payload, dict):
        tasks = (_task_from_mapping(payload, 1),)
    elif isinstance(payload, list):
        tasks = (TaskSpec("Video task", tuple(_step_from_item(item, index)
                                              for index, item in enumerate(payload, 1))),)
    else:
        raise ValueError("Task intelligence must be a JSON object or array")

    if not tasks:
        raise ValueError("Task intelligence did not contain any tasks")
    empty = [task.name for task in tasks if not task.steps]
    if empty:
        raise ValueError(f"Task(s) missing subtasks: {', '.join(empty)}")
    return tasks


class BehaviourTreeRuntime:
    """Execute an explicit tree or a generated task sequence across sim frames."""

    def __init__(
        self,
        payload: Any,
        primitives: Mapping[str, PrimitiveFactory],
        *,
        initial_facts: Iterable[str] = (),
        services: Mapping[str, Any] | None = None,
        logger: Callable[[str], None] = print,
    ) -> None:
        self._primitives = dict(primitives)
        self._initial_facts = set(initial_facts)
        self.blackboard = set(self._initial_facts)
        self.services = dict(services or {})
        self.logger = logger
        if isinstance(payload, dict) and isinstance(payload.get("tree"), dict):
            self.tasks: tuple[TaskSpec, ...] = ()
            self._steps: tuple[StepSpec, ...] = ()
            self.root = _node_from_mapping(payload["tree"])
        else:
            self.tasks = normalize_task_intelligence(payload)
            self._steps = tuple(step for task in self.tasks for step in task.steps)
            self.root = SequenceNode(
                "Generated task sequence",
                [ActionNode(step) for step in self._steps],
            )
        self.status = Status.INVALID
        self.step_index = 0
        self.feedback = "not started"

    @property
    def active_step(self) -> StepSpec | None:
        return self._steps[self.step_index] if self._steps and self.step_index < len(self._steps) else None

    def reset(self) -> None:
        self.root.reset(self)
        self.blackboard.clear()
        self.blackboard.update(self._initial_facts)
        self.status = Status.INVALID
        self.step_index = 0
        self.feedback = "reset"

    def tick(self) -> Status:
        """Advance the tree once; call this exactly once per playing sim frame."""

        if self.status not in (Status.SUCCESS, Status.FAILURE):
            self.status = self.root.tick(self)
            if self.status is Status.SUCCESS:
                self.feedback = "all tasks complete"
            elif self.status is Status.FAILURE and not self.feedback:
                self.feedback = f"tree failed: {self.root.name}"
        return self.status

    def render_tree(self) -> str:
        return "\n".join(self.root.describe())

    def _context(self, step: StepSpec) -> PrimitiveContext:
        return PrimitiveContext(step=step, blackboard=self.blackboard, services=self.services)



def _node_from_mapping(item: dict[str, Any]) -> TreeNode:
    node_type = _clean(item.get("type"), "action").lower()
    name = _clean(item.get("name"), node_type.replace("_", " ").title())
    if node_type == "action":
        return ActionNode(_step_from_item(item, 1))
    if node_type == "condition":
        fact = _clean(item.get("fact"), "")
        if not fact:
            raise ValueError(f"{name}: condition requires 'fact'")
        return ConditionNode(name, fact, bool(item.get("expected", True)))
    if node_type in ("sequence", "selector", "parallel"):
        children = item.get("children")
        if not isinstance(children, list):
            raise ValueError(f"{name}: {node_type} requires a children array")
        nodes = [_node_from_mapping(child) for child in children if isinstance(child, dict)]
        cls = {"sequence": SequenceNode, "selector": SelectorNode, "parallel": ParallelAllNode}[node_type]
        return cls(name, nodes)
    if node_type == "retry":
        child = item.get("child")
        if not isinstance(child, dict):
            raise ValueError(f"{name}: retry requires a child object")
        return RetryNode(name, _node_from_mapping(child), int(item.get("max_attempts", 2)))
    raise ValueError(f"{name}: unsupported behaviour-tree node type {node_type!r}")


def _task_from_mapping(item: dict[str, Any], index: int) -> TaskSpec:
    steps = _step_items(item)
    if not steps and any(key in item for key in ("primitive", "skill", "action", "subtask_name")):
        steps = [item]
    return TaskSpec(
        _text(item, ("task_name", "taskName", "taskDescription", "name"), f"Video task {index}"),
        tuple(_step_from_item(step, step_index) for step_index, step in enumerate(steps, 1)),
    )


def _step_items(item: dict[str, Any]) -> list[Any]:
    for key in ("subtasks", "subTasks", "steps", "actions", "segments"):
        if isinstance(item.get(key), list):
            return item[key]
    return []


def _step_from_item(item: Any, index: int) -> StepSpec:
    if isinstance(item, str):
        name = _clean(item, f"Step {index}")
        return StepSpec(name, _infer_primitive(name), success_conditions=(f"{_slug(name)}_done",))
    if not isinstance(item, dict):
        raise ValueError(f"Step {index} must be an object or string")
    name = _text(item, ("subtask_name", "subTaskDescription", "sub_task_description",
                        "sub_task", "subtask", "step", "action", "description", "name"),
                 f"Step {index}")
    success = _strings(item.get("success_conditions") or item.get("postconditions"))
    return StepSpec(
        name=name,
        primitive=_text(item, ("primitive", "skill", "robot_primitive"), _infer_primitive(name)),
        inputs=dict(item["inputs"]) if isinstance(item.get("inputs"), dict) else {},
        preconditions=tuple(_strings(item.get("preconditions"))),
        success_conditions=tuple(success or [f"{_slug(name)}_done"]),
        description=_text(item, ("description", "subTaskDescription", "sub_task"), ""),
    )


def _text(item: dict[str, Any], keys: Iterable[str], default: str) -> str:
    for key in keys:
        if key in item:
            return _clean(item[key], default)
    return default


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean(value, "")] if _clean(value, "") else []
    if isinstance(value, dict):
        return [_clean(key, "") for key, enabled in value.items() if enabled and _clean(key, "")]
    if isinstance(value, Iterable):
        return [_clean(item, "") for item in value if _clean(item, "")]
    return [_clean(value, "")] if _clean(value, "") else []


def _clean(value: Any, default: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()) or default


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "step"


def _infer_primitive(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("move", "approach", "navigate")):
        return "navigate_to_workspace"
    if any(token in lowered for token in ("detect", "locate", "identify", "localize")):
        return "perceive_objects"
    if any(token in lowered for token in ("pick", "grasp", "grab")):
        return "grasp_object"
    if any(token in lowered for token in ("place", "insert", "attach", "align")):
        return "manipulate_object"
    if any(token in lowered for token in ("inspect", "scan", "trace", "record")):
        return "inspect_workspace"
    return "execute_subtask"
