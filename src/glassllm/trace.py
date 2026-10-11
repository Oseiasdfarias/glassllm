"""Debug mode: record snapshots of marked model steps for one sequence."""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .autograd import Tensor
from .steps import HEAD_STEPS, STEPS


@dataclass(frozen=True)
class StepRecord:
    id: str
    layer: int | None
    value: np.ndarray
    order: int
    head: int | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def equation(self) -> str:
        return STEPS[self.id].plain

    @property
    def latex(self) -> str:
        return STEPS[self.id].latex


def _validate_selector(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None; got {value!r}")


class Tracer:
    def __init__(self):
        self.records: list[StepRecord] = []

    def record(self, step_id: str, value, layer: int | None = None,
               batched: bool = True, *, head: int | None = None) -> None:
        """Copy a value, removing its single batch axis when requested.

        An explicit head identifies an already sliced value; otherwise head
        steps retain their full head axis for selection with get().
        """
        if step_id not in STEPS:
            raise KeyError(f"unknown step id {step_id!r}; add it to glassllm/steps.py")
        _validate_selector("layer", layer)
        _validate_selector("head", head)
        if head is not None and step_id not in HEAD_STEPS:
            raise ValueError(f"step {step_id!r} has no head axis")
        data = value.data if isinstance(value, Tensor) else np.asarray(value)
        if batched:
            if data.ndim == 0 or data.shape[0] != 1:
                raise ValueError("batched tracing requires batch size 1; use batched=False for unbatched values")
            data = data[0]
        self.records.append(StepRecord(step_id, layer, np.array(data, copy=True), len(self.records), head))

    def get(self, step_id: str, layer: int | None = None, head: int | None = None) -> StepRecord:
        """Return the first matching snapshot, optionally selecting a head.

        Explicit head records match only that head. Full arrays can serve any
        valid head; omitting head returns the first record as stored.
        """
        _validate_selector("layer", layer)
        _validate_selector("head", head)
        if head is not None and step_id in STEPS and step_id not in HEAD_STEPS:
            raise ValueError(f"step {step_id!r} has no head axis")
        for rec in self.records:
            if rec.id != step_id or (layer is not None and rec.layer != layer):
                continue
            if head is None or rec.head == head:
                return rec
            if rec.head is not None:
                continue
            if rec.value.ndim == 0 or head >= rec.value.shape[0]:
                raise ValueError(f"head={head} is out of range for step {step_id!r} with shape {rec.shape}")
            return replace(rec, value=rec.value[head], head=head)
        raise KeyError(f"step {step_id!r} (layer={layer}, head={head}) was not recorded")

    def step_ids(self) -> list[str]:
        return list(dict.fromkeys(rec.id for rec in self.records))


def record(trace: Tracer | None, step_id: str, value, layer: int | None = None,
           batched: bool = True, *, head: int | None = None) -> None:
    """Record only when a tracer was given."""
    if trace is not None:
        trace.record(step_id, value, layer, batched, head=head)
