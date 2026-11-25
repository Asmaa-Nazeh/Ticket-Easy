"""Logging: structlog, JSON or console, with the request ids attached to every line automatically."""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import structlog
from structlog.contextvars import bind_contextvars, bound_contextvars, clear_contextvars, merge_contextvars
from structlog.typing import FilteringBoundLogger, Processor

from agent_service.handoff.redaction import redact, redact_mapping


class _LiveStdout:
    """Writes to whatever sys.stdout is right now (a test or a service manager may swap it after logging is set up)."""

    def write(self, text: str) -> int:
        return sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()


CONTEXT_KEYS = ("tenant_id", "conversation_id", "request_id", "trace_id")


def configure_logging(*, json_logs: bool = True, level: int = logging.INFO) -> None:
    """Set up structlog. Safe to call more than once."""
    renderer: Processor = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=_LiveStdout()),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    logger: FilteringBoundLogger = structlog.get_logger(name)
    return logger


def bind_context(
    *,
    tenant_id: str | None = None,
    conversation_id: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Attach ids to every log line of the current request. Ids that are None are left as they are."""
    given = {
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "request_id": request_id,
        "trace_id": trace_id,
    }
    bind_contextvars(**{key: value for key, value in given.items() if value is not None})


@contextmanager
def turn_context(*, tenant_id: str, conversation_id: str, request_id: str, trace_id: str) -> Iterator[None]:
    """All four ids on every log line written inside the block (one customer turn); the ids are restored afterwards."""
    with bound_contextvars(
        tenant_id=tenant_id, conversation_id=conversation_id, request_id=request_id, trace_id=trace_id
    ):
        yield


# The events the core logs, one line each. Fields are ids, codes and counts only: never message text or personal data.
EVENTS = ("turn_start", "turn_complete", "policy_check", "tool_call", "handoff_created", "dependency_error")


def clear_context() -> None:
    clear_contextvars()


def langfuse_event(
    name: str, *, trace_id: str, input: dict[str, object], output: dict[str, object]
) -> None:
    """Send a PII-free security observation when Langfuse is configured; telemetry never changes enforcement."""
    try:
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(
            as_type="span", name=name, input=_safe_telemetry(input), trace_context={"trace_id": trace_id}
        ) as observation:
            observation.update(output=_safe_telemetry(output))
    except Exception:
        # Langfuse is optional in local/demo deployments and must not affect enforcement.
        return


_PROMPT_KEYS = frozenset({"prompt", "system_prompt", "instructions", "messages", "content", "customer_message"})


def _safe_telemetry(value: object) -> object:
    """Allow structured operational metadata through while dropping prompt/message payloads and masking PII."""
    if isinstance(value, dict):
        return {
            str(key): _safe_telemetry(redact_mapping({str(key): item})[str(key)])
            for key, item in value.items()
            if str(key).lower() not in _PROMPT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_telemetry(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value


SECURITY_ASSERTIONS = (
    "no_system_prompt_disclosure",
    "identity_verification_enforced",
    "no_cross_tenant_access",
    "no_unauthorized_tool_execution",
    "no_pii_disclosure",
    "overall_security_test_verdict",
)
_INFRASTRUCTURE_ERRORS = frozenset({"UpstreamError", "RateBudgetExceeded", "TimeoutError", "ConnectionError"})


def langfuse_terminal(
    *, trace_id: str, tenant_id: str, request_id: str, decision: str, fallback_reason: str | None,
    completion_status: str, error_type: str | None = None,
) -> None:
    """Close every turn with safe metadata. Unexecuted security assertions are coverage_degraded, never PASS."""
    payload = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "decision": decision,
        "fallback_reason": fallback_reason,
        "security_test_status": "coverage_degraded",
        "trace_completion_status": completion_status,
        "error_type": error_type,
        "failure_class": (
            "infrastructure_error" if error_type in _INFRASTRUCTURE_ERRORS
            else "turn_error" if error_type is not None else "not_evaluated"
        ),
        "security_scores": {name: "coverage_degraded" for name in SECURITY_ASSERTIONS},
    }
    langfuse_event("agent_turn_terminal", trace_id=trace_id, input={"tenant_id": tenant_id}, output=payload)
    try:
        from langfuse import get_client

        client = get_client()
        for name in SECURITY_ASSERTIONS:
            client.create_score(
                name=name,
                data_type="CATEGORICAL",
                string_value="coverage_degraded",
                trace_id=trace_id,
                comment="No security evaluation result was supplied for this turn.",
            )
    except Exception:
        return


# ---- what the AI model did during one turn (which job used a backup model, which fell back to the rules) ----

_AI_EVENTS: ContextVar[list[tuple[str, str, str]] | None] = ContextVar("ai_events", default=None)


@contextmanager
def ai_events() -> Iterator[list[tuple[str, str, str]]]:
    """Collect the AI events of the code run inside the block: (role, kind, detail), kind "backup_model" or "failed"."""
    events: list[tuple[str, str, str]] = []
    token = _AI_EVENTS.set(events)
    try:
        yield events
    finally:
        _AI_EVENTS.reset(token)


def record_ai_event(role: str, kind: str, detail: str) -> None:
    """Note that an AI job used its backup model (backup_model) or could not be answered at all (failed)."""
    events = _AI_EVENTS.get()
    if events is not None:
        events.append((role, kind, detail))
