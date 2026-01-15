from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class AdkStep:
    """Step descriptor for the ADK-style pipeline runner."""
    name: str
    fn: Callable[[object], None]
    skip_if: Optional[Callable[[object], bool]] = None
    always_run: bool = False


class AdkAgent:
    """Lightweight ADK-style step runner for deterministic pipelines."""

    def __init__(self, steps: list[AdkStep]) -> None:
        """Purpose: Initialize the agent with an ordered list of steps.
        Inputs/Outputs: Input is a list of AdkStep; no return value.
        Side Effects / State: Stores the step list for later execution.
        Dependencies: None beyond AdkStep definitions.
        Failure Modes: None; assumes valid callables in steps.
        If Removed: Pipeline steps are never executed and agent does nothing.
        Testing Notes: Provide a minimal step list and ensure order is preserved.
        """
        # Store the pipeline steps for deterministic execution.
        self._steps = steps

    def run(self, context: object) -> None:
        """Purpose: Execute steps in order with optional skip/always-run rules.
        Inputs/Outputs: Input is a mutable context object; no return value.
        Side Effects / State: Invokes step functions that may mutate context.
        Dependencies: Depends on AdkStep.fn and AdkStep.skip_if semantics.
        Failure Modes: Exceptions in step functions propagate to the caller.
        If Removed: The agent pipeline cannot run, breaking request handling.
        Testing Notes: Verify skip_if and always_run logic with simple steps.
        """
        # Iterate steps and honor always_run/skip_if guards.
        for step in self._steps:
            if step.always_run:
                step.fn(context)
                continue
            if step.skip_if and step.skip_if(context):
                continue
            step.fn(context)
