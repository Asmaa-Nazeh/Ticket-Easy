import json

import pytest

from agent_service.observability import bind_context, clear_context, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    clear_context()


def lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_json_logs_carry_the_bound_ids(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)
    bind_context(tenant_id="shop_001", conversation_id="conv-1", request_id="req-1", trace_id="t-1")
    get_logger("test").info("hello", extra_field=3)
    (entry,) = lines(capsys)
    assert entry["event"] == "hello" and entry["level"] == "info" and entry["extra_field"] == 3
    assert (entry["tenant_id"], entry["conversation_id"]) == ("shop_001", "conv-1")
    assert (entry["request_id"], entry["trace_id"]) == ("req-1", "t-1")
    assert "timestamp" in entry


def test_binding_is_additive_and_none_is_ignored(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)
    bind_context(request_id="req-1")
    bind_context(tenant_id="shop_001", request_id=None)
    get_logger("test").info("x")
    (entry,) = lines(capsys)
    assert entry["request_id"] == "req-1" and entry["tenant_id"] == "shop_001"
    assert "trace_id" not in entry


def test_clear_context_removes_the_ids(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)
    bind_context(tenant_id="shop_001")
    clear_context()
    get_logger("test").info("x")
    (entry,) = lines(capsys)
    assert "tenant_id" not in entry


def test_console_mode_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=False)
    get_logger("test").info("plain message")
    out = capsys.readouterr().out
    assert "plain message" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip().splitlines()[0])
    configure_logging(json_logs=True)


def test_langfuse_payloads_drop_prompts_and_redact_pii() -> None:
    from agent_service.observability import _safe_telemetry

    payload = _safe_telemetry({
        "system_prompt": "do not reveal this",
        "customer_message": "Mona mail mona@example.com call 01012345601",
        "debug": "request from 15 Tahrir Street",
        "stage": "risk_screen",
        "details": {"instructions": "nested prompt", "note": "customer text: ignore system; code 482913"},
    })
    assert payload == {
        "debug": "request from [address]",
        "stage": "risk_screen",
        "details": {"note": "customer text: ignore system; code [otp]"},
    }


def test_terminal_scores_report_missing_execution_as_coverage_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from agent_service.observability import SECURITY_ASSERTIONS, langfuse_terminal

    scores: list[dict[str, object]] = []
    terminal_output: dict[str, object] = {}

    class Observation:
        def update(self, **kwargs: object) -> None:
            terminal_output.update(kwargs.get("output", {}))  # type: ignore[arg-type]

    client = SimpleNamespace(
        create_score=lambda **kwargs: scores.append(kwargs),
        start_as_current_observation=lambda **kwargs: nullcontext(Observation()),
    )
    from contextlib import nullcontext

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=lambda: client))
    langfuse_terminal(
        trace_id="trace-1", tenant_id="shop_001", request_id="req-1", decision="answer",
        fallback_reason=None, completion_status="failed", error_type="UpstreamError",
    )
    assert len(scores) == len(SECURITY_ASSERTIONS)
    assert {score["name"] for score in scores} == set(SECURITY_ASSERTIONS)
    assert all(score["string_value"] == "coverage_degraded" for score in scores)
    assert all(score["data_type"] == "CATEGORICAL" for score in scores)
    assert terminal_output["failure_class"] == "infrastructure_error"
    assert terminal_output["security_test_status"] == "coverage_degraded"


def test_every_langfuse_span_boundary_sanitizes_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from agent_service.observability import langfuse_event

    recorded: dict[str, object] = {}

    class Observation:
        def update(self, **kwargs: object) -> None:
            recorded.update(kwargs)

    client = SimpleNamespace(
        start_as_current_observation=lambda **kwargs: (
            recorded.update(kwargs) or nullcontext(Observation())
        )
    )
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=lambda: client))
    langfuse_event(
        "debug", trace_id="trace-1",
        input={"customer_message": "omit me", "phone": "01012345601", "stage": "verify"},
        output={"system_prompt": "omit me too", "email": "mona@example.com", "status": "ok"},
    )
    assert recorded["input"] == {"phone": "[phone]", "stage": "verify"}
    assert recorded["output"] == {"email": "[email]", "status": "ok"}


async def test_ids_are_isolated_between_concurrent_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    import asyncio

    configure_logging(json_logs=True)

    async def work(tenant: str) -> None:
        bind_context(tenant_id=tenant)
        await asyncio.sleep(0)
        get_logger("test").info("work", who=tenant)

    await asyncio.gather(work("shop_001"), work("shop_002"))
    for entry in lines(capsys):
        assert entry["tenant_id"] == entry["who"]
