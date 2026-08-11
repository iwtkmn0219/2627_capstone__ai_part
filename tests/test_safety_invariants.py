"""안전 불변식.

깨지면 곧바로 실제 피해가 되는 것들만 모았다. 일반 회귀 테스트와 달리
"고치는 게 나은 실패" 가 아니라 "절대 실패하면 안 되는" 항목이다.

  1. L4 진술이 학부모 대시보드가 읽는 로그에 남지 않는다
  2. L4 응답이 보호자를 지목해 권하지 않는다
  3. L4 는 알림을 허용하지 않는다 (fail-closed)
"""

from __future__ import annotations

import asyncio
import json

from core.fakes import FakeLLM, FakeSTT, FakeTTS
from core.interfaces import RiskLevel
from core.orchestrator import TurnOrchestrator
from safety.checkers import PROTECT_REPLY, ReadabilityChecker, RuleChecker
from telemetry.logger import TurnLogger

PROFILE = {"name": "지우", "age": 5}

# 아이를 가해자 쪽으로 밀 수 있는 표현. L4 응답에는 하나도 들어가면 안 된다.
GUARDIAN_WORDS = ("엄마", "아빠", "어머니", "아버지", "보호자", "부모", "선생님")


def run_turn(utterance: str, model_reply: str = "그랬구나, 많이 아팠겠다."):
    """가짜 엔진으로 한 턴을 실행한다.

    Args:
        utterance: 아이 발화.
        model_reply: 생성 모델이 뱉었다고 가정할 응답.

    Returns:
        실행 결과 TurnResult.
    """
    orch = TurnOrchestrator(
        stt=FakeSTT(utterance),
        llm=FakeLLM([model_reply]),
        tts=FakeTTS(),
        input_checkers=[RuleChecker()],
        output_checkers=[RuleChecker(), ReadabilityChecker()],
    )
    return asyncio.run(
        orch.run(b"<audio>", profile=PROFILE, history=[], synthesize=False)
    )


def read_jsonl(path):
    """JSONL 파일을 dict 리스트로 읽는다.

    Args:
        path: 읽을 파일 경로.

    Returns:
        레코드 리스트. 파일이 없으면 빈 리스트.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------- 응답


def test_protect_reply_never_names_a_guardian():
    """L4 고정 문장이 특정 어른을 지목해 권하지 않는다.

    문구를 다듬다가 "엄마한테 말해볼까?" 로 되돌아가는 것을 막는다.
    지목된 어른이 누구인지 모르는 상태에서 그 문장은 아이를 가해자에게 보낸다.
    """
    for word in GUARDIAN_WORDS:
        assert word not in PROTECT_REPLY


def test_l4_discards_model_output():
    """L4 에서는 생성 모델이 무슨 말을 하든 고정 문장으로 대체된다."""
    result = run_turn("아빠가 때렸어", model_reply="엄마한테 말해보는 게 어때?")
    assert result.reply_text == PROTECT_REPLY
    # 생성 자체를 건너뛰므로 토큰도 쓰지 않는다.
    assert result.model == ""


def test_l4_reply_has_no_guardian_word():
    """L4 턴의 최종 응답에 보호자 지칭이 없다."""
    reply = run_turn("아빠가 때렸어").reply_text
    assert not any(w in reply for w in GUARDIAN_WORDS)


def test_l3_keeps_generated_reply():
    """L3 는 대화를 끊지 않는다.

    L4 방어를 강화하다가 L3 까지 고정 문장으로 덮어버리면, 일상적인 다침·무서움에도
    대화가 끊긴다. 그건 그 자체로 나쁜 경험이다.
    """
    result = run_turn("어제 넘어져서 다쳤어", model_reply="많이 아팠겠다.")
    assert result.reply_text == "많이 아팠겠다."
    assert result.risk.level is RiskLevel.L3


# --------------------------------------------------------------- 알림


def test_l4_never_allows_notification():
    """L4 는 알림을 허용하지 않는다. 라우터가 해소에 성공한 뒤에만 뒤집을 수 있다."""
    result = run_turn("아빠가 때렸어")
    assert result.risk.level is RiskLevel.L4
    assert result.risk.notify_allowed is False


def test_below_l4_allows_notification():
    """L3 이하는 알림 자체가 안전하므로 막지 않는다."""
    assert run_turn("어제 넘어져서 다쳤어").risk.notify_allowed is True


# --------------------------------------------------------------- 로그


def test_l4_text_absent_from_dashboard_log(tmp_path):
    """L4 진술이 학부모 대시보드의 데이터 소스에 남지 않는다.

    대시보드를 보는 사람이 아이가 지목한 그 어른일 수 있다. 이 테스트가 깨지면
    아이의 진술이 가해자에게 그대로 노출된다.
    """
    logger = TurnLogger(tmp_path / "turns.jsonl", tmp_path / "review.jsonl")
    logger.log(child_id="anon", session_id="s", result=run_turn("아빠가 때렸어"))

    (record,) = read_jsonl(tmp_path / "turns.jsonl")
    assert record["child_text"] is None
    assert record["reply_text"] is None
    assert record["risk"]["alleged_target"] is None
    # 안전 이벤트에도 판정 대상 원문이 실려 있으므로 같이 지워야 한다.
    for event in record["safety_events"]:
        assert event["text"] is None
        assert event["matched_text"] is None
        assert event["alleged_target"] is None


def test_l4_text_present_in_review_queue(tmp_path):
    """검토자는 원문 없이 판단할 수 없으므로 검토 큐에는 남긴다."""
    logger = TurnLogger(tmp_path / "turns.jsonl", tmp_path / "review.jsonl")
    logger.log(child_id="anon", session_id="s", result=run_turn("아빠가 때렸어"))

    (item,) = read_jsonl(tmp_path / "review.jsonl")
    assert item["child_text"] == "아빠가 때렸어"
    assert item["alleged_target"] == "아빠"
    assert item["level"] == int(RiskLevel.L4)


def test_l3_is_not_over_redacted(tmp_path):
    """L3 는 마스킹 대상이 아니다.

    과잉 마스킹으로 흘러가면 대시보드가 비고 P3 분석 원자료도 사라진다.
    """
    logger = TurnLogger(tmp_path / "turns.jsonl", tmp_path / "review.jsonl")
    logger.log(child_id="anon", session_id="s", result=run_turn("친구가 때렸어"))

    (record,) = read_jsonl(tmp_path / "turns.jsonl")
    assert record["child_text"] == "친구가 때렸어"
    assert record["risk"]["alleged_target"] == "친구"
    # L4 가 아니므로 검토 큐에는 올라가지 않는다.
    assert read_jsonl(tmp_path / "review.jsonl") == []


def test_store_text_consent_still_applies(tmp_path):
    """보호자가 원문 저장에 동의하지 않으면 L4 가 아니어도 원문을 남기지 않는다."""
    logger = TurnLogger(tmp_path / "turns.jsonl", tmp_path / "review.jsonl")
    logger.log(
        child_id="anon",
        session_id="s",
        result=run_turn("친구가 때렸어"),
        store_text=False,
    )

    (record,) = read_jsonl(tmp_path / "turns.jsonl")
    assert record["child_text"] is None
    assert all(e["text"] is None for e in record["safety_events"])
