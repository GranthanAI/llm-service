"""Follow-up question suggestions (chat.response.suggestions 1.0).

After a SUCCESSFUL chat.response.generated (never an error or a cancelled turn), when the user's preference allows it and the
answer is long enough to be worth following up, a background task asks a small model for 2 or 3 short follow-up questions and
publishes them. It can never delay or fail the answer: it starts after the answer was published, has its own timeout, and every
failure is a metric and a log line.

The call is cost-only: its tokens are published on usage.metered (feature "followups"), so billing sees the cost without charging
the user a message.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.config.logging import get_logger
from app.usage.collector import UsageCollector, usage_purpose
from app.utils.metrics import FOLLOWUPS_TOTAL

logger = get_logger("followups")

MIN_ANSWER_CHARS = 200
ANSWER_TAIL_CHARS = 1500
MAX_QUESTION_CHARS = 120
MAX_SUGGESTIONS = 3
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "followups" / "followups_v1.yaml"
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def load_prompt(path: Path = PROMPT_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_suggestions(reply: str) -> list[str]:
    """The valid suggestions of a model reply: parsed JSON, stripped, with empties, duplicates, anything over 120 characters
    and anything with a newline dropped; at most 3. The caller publishes nothing when fewer than 2 are valid."""
    text = _FENCE.sub("", (reply or "").strip())
    try:
        data = json.loads(text)
    except ValueError:
        return []
    items = data.get("suggestions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        q = item.strip()
        if not q or len(q) > MAX_QUESTION_CHARS or "\n" in q or "\r" in q or q in out:
            continue
        out.append(q)
        if len(out) >= MAX_SUGGESTIONS:
            break
    return out


class FollowupService:
    def __init__(self, publisher: Any, adapter: Any, model_label: str, *, enabled: bool = True, timeout_ms: int = 4000,
                 max_tokens: int = 300, model_resolver: Any = None, prompt: dict[str, Any] | None = None):
        self.publisher = publisher
        self.adapter = adapter                  # anything with `async execute(messages, params) -> GenerationResponse`
        self.model_label = model_label
        self.enabled = enabled and adapter is not None
        self.timeout_s = timeout_ms / 1000.0
        self.max_tokens = max_tokens
        self.model_resolver = model_resolver
        self.prompt = prompt or load_prompt()
        self._tasks: set[asyncio.Task] = set()

    # ---- start -------------------------------------------------------------------------------------------------

    def should_run(self, ctx: Any, answer: str) -> bool:
        return bool(self.enabled and getattr(ctx.preferences, "suggest_followups", True)
                    and len((answer or "").strip()) >= MIN_ANSWER_CHARS and ctx.response_message_id)

    def maybe_start(self, ctx: Any, answer: str) -> asyncio.Task | None:
        """Starts the background task (or counts a skip). Never raises, never awaits."""
        if not self.should_run(ctx, answer):
            FOLLOWUPS_TOTAL.labels(outcome="skipped").inc()
            return None
        task = asyncio.get_running_loop().create_task(self._run(ctx, answer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Waits for running tasks (shutdown and tests)."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ---- the task ------------------------------------------------------------------------------------------------

    def build_messages(self, ctx: Any, answer: str) -> list[dict[str, str]]:
        plan_mode = getattr(ctx.plan, "mode", None)
        mode = getattr(plan_mode, "value", None) or (str(plan_mode) if plan_mode else "default")
        language = getattr(ctx.preferences, "language_name", "") or getattr(ctx.preferences, "language", "auto")
        user = self.prompt["user_template"].format(
            mode=mode, language=language, question=ctx.user_message[:1500], answer_tail=answer.strip()[-ANSWER_TAIL_CHARS:])
        return [{"role": "system", "content": self.prompt["system"]}, {"role": "user", "content": user}]

    async def _run(self, ctx: Any, answer: str) -> None:
        collector = UsageCollector(model_resolver=self.model_resolver)
        outcome = "error"
        try:
            messages = self.build_messages(ctx, answer)
            params = {"max_tokens": self.max_tokens, "temperature": 0.4, "_usage": collector,
                      "response_format": {"type": "json_object"}}
            with usage_purpose("followups"):
                reply = await asyncio.wait_for(self.adapter.execute(messages, params), timeout=self.timeout_s)
            suggestions = parse_suggestions(getattr(reply, "content", ""))
            if len(suggestions) < 2:
                outcome = "invalid"
            else:
                await self.publisher.publish_suggestions(ctx, suggestions, self.model_label)
                outcome = "published"
        except asyncio.TimeoutError:
            outcome = "timeout"
        except Exception as exc:  # noqa: BLE001 - a suggestion can never hurt the answer
            logger.warning("followups_failed", error=type(exc).__name__, conversation_id=ctx.conversation_id)
        finally:
            FOLLOWUPS_TOTAL.labels(outcome=outcome).inc()
            await self._meter(ctx, collector)

    async def _meter(self, ctx: Any, collector: UsageCollector) -> None:
        """The cost of the call (also of one that timed out or returned nothing usable), on usage.metered."""
        if not collector.calls:
            return
        try:
            await self.publisher.publish_usage_metered(
                user_id=ctx.user_id, feature="followups", calls=collector.snapshot(),
                conversation_id=ctx.conversation_id, request_message_id=ctx.message_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("usage_metered_publish_failed", feature="followups", error=type(exc).__name__)
