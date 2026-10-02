"""Voice-facing tool that hands a task to the background workflow runner."""

import asyncio
import logging

from livekit.agents import RunContext, function_tool, inference

from workflow import WorkflowResult, WorkflowRunner, llm_planner

logger = logging.getLogger("delegate")

# Register real capabilities here: async callables taking keyword args, returning str.
# Gmail/Calendar/Drive etc. can be wired in the same way.
TOOLS: dict = {}

_background: set[asyncio.Task] = set()


@function_tool()
async def delegate_workflow(context: RunContext, task: str) -> str:
    """Hand a multi-step task to the background worker. It runs on its own and
    the result is reported back to the user when finished.

    Args:
        task: A clear description of the work to complete.
    """
    session = context.session

    async def report(result: WorkflowResult) -> None:
        session.generate_reply(
            instructions=(
                f"Briefly tell the user the background task '{result.task}' "
                f"{result.status.value}. Details: {result.summary}"
            )
        )

    runner = WorkflowRunner(
        tools=TOOLS,
        planner=llm_planner(inference.LLM(model="openai/gpt-4.1-mini")),
        notifier=report,
    )
    job = asyncio.create_task(runner.run(task))
    _background.add(job)
    job.add_done_callback(_background.discard)
    return "Started. Tell the user you're on it and will report back when done."
