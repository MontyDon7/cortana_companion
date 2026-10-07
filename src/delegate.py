"""Voice-facing tool that hands a task to the background workflow runner."""

import asyncio
import logging
import os

from livekit.agents import RunContext, function_tool, inference

from google_tools import build_tools, client_from_env, make_email_notifier
from workflow import WorkflowResult, WorkflowRunner, llm_planner

logger = logging.getLogger("delegate")

# Gmail/Calendar/Drive tools are enabled when GOOGLE_* credentials are set.
# Add other capabilities here: async callables taking keyword args, returning str.
_google = client_from_env(os.environ)
TOOLS: dict = build_tools(_google) if _google else {}
_notify_email = os.environ.get("NOTIFY_EMAIL")
_email_notifier = (
    make_email_notifier(_google, _notify_email) if _google and _notify_email else None
)

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
        if _email_notifier:
            try:
                await _email_notifier(result)
            except Exception:
                logger.exception("Email delivery failed")
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
