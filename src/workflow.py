"""Simple workflow executor: receive a task, plan it, run it, report back."""

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("workflow")

Tool = Callable[..., Awaitable[str]]


class WorkflowStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str = ""


@dataclass
class WorkflowResult:
    task: str
    status: WorkflowStatus
    steps: list[Step]
    summary: str


Planner = Callable[[str, list[str]], Awaitable[list[Step]]]
Notifier = Callable[[WorkflowResult], Awaitable[None]]


def parse_plan(raw: str) -> list[Step]:
    """Parse an LLM reply (optionally code-fenced) into a list of steps."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
        return [Step(tool=item["tool"], args=item.get("args", {})) for item in data]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Could not parse plan: {e}") from e


def llm_planner(llm_instance) -> Planner:
    """Build a planner that asks a LiveKit LLM to produce a JSON step list."""
    from livekit.agents import llm

    async def plan(task: str, tool_names: list[str]) -> list[Step]:
        ctx = llm.ChatContext()
        ctx.add_message(
            role="system",
            content=(
                "You are a workflow planner. Break the task into ordered steps "
                f"using ONLY these tools: {tool_names}. Reply with a JSON array "
                'only, e.g. [{"tool": "name", "args": {"param": "value"}}].'
            ),
        )
        ctx.add_message(role="user", content=task)
        stream = llm_instance.chat(chat_ctx=ctx)
        parts = [p async for p in stream.to_str_iterable()]
        return parse_plan("".join(parts))

    return plan


class WorkflowRunner:
    def __init__(
        self,
        tools: dict[str, Tool],
        planner: Planner,
        notifier: Notifier | None = None,
        step_timeout: float = 60.0,
    ) -> None:
        self._tools = tools
        self._planner = planner
        self._notifier = notifier
        self._step_timeout = step_timeout

    async def run(self, task: str) -> WorkflowResult:
        result = await self._execute(task)
        if self._notifier:
            try:
                await self._notifier(result)
            except Exception:
                logger.exception("Failed to deliver workflow result")
        return result

    async def _execute(self, task: str) -> WorkflowResult:
        try:
            plan = await self._planner(task, list(self._tools))
        except Exception as e:
            return WorkflowResult(
                task, WorkflowStatus.FAILED, [], f"Planning failed: {e}"
            )

        done: list[Step] = []
        for step in plan:
            done.append(step)
            tool = self._tools.get(step.tool)
            if tool is None:
                step.error = f"unknown tool '{step.tool}'"
            else:
                try:
                    out = tool(**step.args)
                    if inspect.isawaitable(out):
                        out = await asyncio.wait_for(out, self._step_timeout)
                    step.output = str(out)
                except asyncio.TimeoutError:
                    step.error = f"'{step.tool}' timed out after {self._step_timeout}s"
                except Exception as e:
                    step.error = f"'{step.tool}' failed: {e}"
            if step.error:
                return WorkflowResult(
                    task,
                    WorkflowStatus.FAILED,
                    done,
                    f"Stopped at step {len(done)}: {step.error}",
                )

        summary = "; ".join(f"{s.tool}: {s.output}" for s in done) or "Nothing to do."
        return WorkflowResult(task, WorkflowStatus.COMPLETED, done, summary)
