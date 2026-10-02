import asyncio

import pytest

from workflow import (
    Step,
    WorkflowRunner,
    WorkflowStatus,
    parse_plan,
)


async def _echo(text: str) -> str:
    return f"echo:{text}"


async def _boom(**_) -> str:
    raise RuntimeError("kaboom")


def _planner(steps: list[Step]):
    async def plan(task: str, tool_names: list[str]) -> list[Step]:
        return steps

    return plan


async def test_runs_all_steps_and_reports_results() -> None:
    runner = WorkflowRunner(
        tools={"echo": _echo},
        planner=_planner([Step("echo", {"text": "a"}), Step("echo", {"text": "b"})]),
    )
    result = await runner.run("say a then b")

    assert result.status is WorkflowStatus.COMPLETED
    assert [s.output for s in result.steps] == ["echo:a", "echo:b"]
    assert "echo:a" in result.summary and "echo:b" in result.summary


async def test_failed_step_stops_workflow_and_reports_error() -> None:
    runner = WorkflowRunner(
        tools={"echo": _echo, "boom": _boom},
        planner=_planner(
            [Step("echo", {"text": "a"}), Step("boom", {}), Step("echo", {"text": "c"})]
        ),
    )
    result = await runner.run("x")

    assert result.status is WorkflowStatus.FAILED
    assert len(result.steps) == 2  # third step never ran
    assert "kaboom" in result.summary


async def test_unknown_tool_fails_cleanly() -> None:
    runner = WorkflowRunner(tools={}, planner=_planner([Step("nope", {})]))
    result = await runner.run("x")

    assert result.status is WorkflowStatus.FAILED
    assert "nope" in result.summary


async def test_step_timeout() -> None:
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    runner = WorkflowRunner(
        tools={"slow": slow},
        planner=_planner([Step("slow", {})]),
        step_timeout=0.05,
    )
    result = await runner.run("x")

    assert result.status is WorkflowStatus.FAILED
    assert "timed out" in result.summary


async def test_delivers_result_to_notifier() -> None:
    delivered = []

    async def notify(result) -> None:
        delivered.append(result)

    runner = WorkflowRunner(
        tools={"echo": _echo},
        planner=_planner([Step("echo", {"text": "a"})]),
        notifier=notify,
    )
    result = await runner.run("x")

    assert delivered == [result]


async def test_planner_failure_is_reported_and_notified() -> None:
    delivered = []

    async def bad_plan(task, tool_names):
        raise ValueError("no plan")

    async def notify(result) -> None:
        delivered.append(result)

    runner = WorkflowRunner(tools={}, planner=bad_plan, notifier=notify)
    result = await runner.run("x")

    assert result.status is WorkflowStatus.FAILED
    assert delivered and "no plan" in delivered[0].summary


def test_parse_plan_accepts_fenced_json() -> None:
    raw = '```json\n[{"tool": "echo", "args": {"text": "hi"}}]\n```'
    assert parse_plan(raw) == [Step("echo", {"text": "hi"})]


def test_parse_plan_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_plan("I think you should just do it")
