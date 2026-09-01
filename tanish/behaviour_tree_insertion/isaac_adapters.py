"""Adapters that turn Isaac Sim controller queues into tree primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .runtime import PrimitiveContext, Status, StepSpec


class IsaacControllerPrimitive:
    """Run one queue-building function through an Isaac articulation controller.

    ``queue`` is called once from ``initialise``. On each later simulation frame,
    the adapter reads robot joints, advances the motion controller, and applies
    its action. A queue that is empty after initialisation is considered a setup
    error rather than instant success.
    """

    def __init__(
        self,
        step: StepSpec,
        queue: Callable[[PrimitiveContext], None],
        *,
        validate: Callable[[PrimitiveContext], bool] | None = None,
        clear_on_terminate: bool = False,
        while_running: Callable[[PrimitiveContext], None] | None = None,
    ) -> None:
        self.step = step
        self.queue = queue
        self.validate = validate
        self.clear_on_terminate = clear_on_terminate
        self.while_running = while_running
        self._queued = False

    def initialise(self, context: PrimitiveContext) -> None:
        controller = _service(context, "motion_controller")
        controller.clear_queue()
        self.queue(context)
        self._queued = not controller.is_done()
        if not self._queued:
            raise RuntimeError(f"{self.step.primitive!r} did not queue any controller commands")

    def tick(self, context: PrimitiveContext) -> Status:
        controller = _service(context, "motion_controller")
        robot = _service(context, "robot")
        articulation_controller = context.services.get("articulation_controller")
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        if self.while_running is not None:
            self.while_running(context)

        if _controller_failed(controller):
            return Status.FAILURE
        if controller.is_done():
            return Status.SUCCESS if self.validate is None or self.validate(context) else Status.FAILURE

        joint_positions = robot.get_joint_positions()
        if joint_positions is None:
            return Status.RUNNING
        if hasattr(joint_positions, "cpu"):
            joint_positions = joint_positions.cpu().numpy()
        action = controller.forward(joint_positions)
        articulation_controller.apply_action(action)
        if _controller_failed(controller):
            return Status.FAILURE
        return Status.RUNNING

    def terminate(self, context: PrimitiveContext, new_status: Status) -> None:
        if self.clear_on_terminate or new_status is not Status.SUCCESS:
            _service(context, "motion_controller").clear_queue()


class FunctionPrimitive:
    """Wrap perception/validation callbacks that complete within one frame."""

    def __init__(self, step: StepSpec, callback: Callable[[PrimitiveContext], bool | Status]) -> None:
        self.step = step
        self.callback = callback

    def initialise(self, context: PrimitiveContext) -> None:
        pass

    def tick(self, context: PrimitiveContext) -> Status:
        result = self.callback(context)
        if isinstance(result, Status):
            return result
        return Status.SUCCESS if result else Status.FAILURE

    def terminate(self, context: PrimitiveContext, new_status: Status) -> None:
        pass


def controller_primitive(
    queue: Callable[[PrimitiveContext], None],
    *,
    validate: Callable[[PrimitiveContext], bool] | None = None,
    clear_on_terminate: bool = False,
    while_running: Callable[[PrimitiveContext], None] | None = None,
):
    """Create a primitive-registry factory for a controller-backed action."""

    return lambda step: IsaacControllerPrimitive(
        step,
        queue,
        validate=validate,
        clear_on_terminate=clear_on_terminate,
        while_running=while_running,
    )


def function_primitive(callback: Callable[[PrimitiveContext], bool | Status]):
    """Create a primitive-registry factory for a callback-backed action."""

    return lambda step: FunctionPrimitive(step, callback)


def _service(context: PrimitiveContext, name: str) -> Any:
    try:
        return context.services[name]
    except KeyError as exc:
        raise RuntimeError(f"Isaac primitive requires service {name!r}") from exc


def _controller_failed(controller: Any) -> bool:
    checker = getattr(controller, "has_failed", None)
    if callable(checker):
        return bool(checker())
    return bool(getattr(controller, "_segment_failed", False))
