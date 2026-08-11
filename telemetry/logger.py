"""
턴 로그. 두 가지 목적을 동시에 만족.
  1) 학부모 대시보드의 데이터 소스
  2) P3 정량 평가의 원자료 (지연/비용/안전 판정)

JSONL 로 남기면 나중에 pandas 로 바로 분석.
개인정보가 들어가므로 child_id 는 반드시 익명 ID를 쓰고,
원문 저장 여부는 보호자 동의 설정과 연동할 것.

L4(보호 필요)는 위 1)과 같은 파일에 담으면 안 된다. 대시보드를 보는 사람이
아이가 지목한 그 어른일 수 있다. 별도 검토 큐로 분리하고 접근을 통제할 것.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.interfaces import RiskLevel

# 2026-08 기준 표준 요금 (USD / 1M tokens): {모델: (입력, 출력)}
# 모델은 GEMINI_MODEL 환경변수로 갈아끼우므로 단가도 모델별로 들고 있어야 한다.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.125, 0.750),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

# 모르는 모델을 0원으로 치면 비용이 0으로 보여서 아무도 눈치채지 못한다.
# 표에서 가장 비싼 요금으로 계산해 과다 계상 쪽으로 틀리게 둔다.
_FALLBACK_PRICE = max(PRICING.values(), key=lambda p: p[1])


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """모델별 단가로 토큰 비용을 계산한다.

    Args:
        model: 응답을 생성한 모델 이름. PRICING 에 없으면 최고가로 계산한다.
        input_tokens: 입력 토큰 수.
        output_tokens: 출력 토큰 수.

    Returns:
        USD 단위 예상 비용.
    """
    price_in, price_out = PRICING.get(model, _FALLBACK_PRICE)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class TurnLogger:
    """턴 결과를 JSONL 로그 파일에 append 하는 로거.

    싱크가 둘이다. 학부모 대시보드가 읽는 일반 로그와, 접근을 통제해야 하는
    검토 큐를 같은 파일에 담으면 아이가 학대를 진술한 문장이 지목된 어른에게
    그대로 노출된다. L4 는 일반 로그에서 원문을 지우고 검토 큐에만 싣는다.

    Attributes:
        path: 일반 턴 로그 경로. 대시보드의 데이터 소스.
        review_path: 검토 큐 경로. 대시보드에 절대 연결하지 말 것.
    """

    def __init__(
        self,
        path: str = "logs/turns.jsonl",
        review_path: str = "logs/review_queue.jsonl",
    ):
        """로그 파일 경로를 설정하고 부모 디렉터리를 준비.

        Args:
            path: 일반 턴 로그를 저장할 JSONL 파일 경로. 부모 디렉터리가 없으면 생성.
            review_path: 검토 큐를 저장할 JSONL 파일 경로.
        """
        self.path = Path(path)
        self.review_path = Path(review_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.review_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, child_id: str, session_id: str, result, store_text: bool = True):
        """한 턴의 결과를 로그 한 줄로 append.

        L4 턴은 일반 로그에서 원문과 지목 대상을 지우고, 원문을 담은 별도 레코드를
        검토 큐에 함께 적재한다.

        Args:
            child_id: 익명화된 아이 식별자.
            session_id: 세션 식별자.
            result: 기록할 TurnResult.
            store_text: False 면 발화/응답 원문은 저장하지 않음.
                safety_events 안의 원문에도 똑같이 적용된다.

        Returns:
            일반 로그에 실제로 기록된 dict.
        """
        tok = result.tokens or {}
        cost = estimate_cost(result.model, tok.get("input", 0), tok.get("output", 0))
        risk = result.risk
        protect = risk.level >= RiskLevel.L4

        # 안전 이벤트도 판정 대상 원문을 싣고 있으므로 동의 설정을 함께 적용한다.
        # 여기서 지우지 않으면 store_text=False 가 그대로 우회된다.
        redact = not store_text or protect
        events = result.safety_events
        if redact:
            events = [
                {**e, "text": None, "matched_text": None, "alleged_target": None}
                for e in events
            ]

        record = {
            "turn_id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "child_id": child_id,
            "session_id": session_id,
            "child_text": None if redact else result.child_text,
            "reply_text": None if redact else result.reply_text,
            "escalate": result.escalate,
            "risk": {
                "level": int(risk.level),
                "categories": risk.categories,
                "ongoing": risk.ongoing,
                # 지목 대상은 피지목자를 식별하는 정보다. 일반 로그에는 남기지 않는다.
                "alleged_target": None if protect else risk.alleged_target,
                "target_certain": risk.target_certain,
                "notify_allowed": risk.notify_allowed,
            },
            "safety_events": events,
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "total_ms": round(result.total_ms, 1),
            "model": result.model,
            "tokens": tok,
            "est_cost_usd": round(cost, 8),
        }
        _append(self.path, record)

        if protect:
            _append(
                self.review_path,
                {
                    "turn_id": record["turn_id"],
                    "ts": record["ts"],
                    "child_id": child_id,
                    "session_id": session_id,
                    "level": int(risk.level),
                    "categories": risk.categories,
                    "ongoing": risk.ongoing,
                    "alleged_target": risk.alleged_target,
                    "target_certain": risk.target_certain,
                    # TODO: 원문 보관을 store_text(보호자 동의)에 걸어둘지는 미결.
                    # 지금은 동의를 따르지만, 그러면 동의를 끈 계정에서 검토 큐가
                    # 판단 근거 없는 껍데기가 된다. 팀에서 결정할 것.
                    "child_text": result.child_text if store_text else None,
                    "reply_text": result.reply_text if store_text else None,
                },
            )
        return record


def _append(path: Path, record: dict) -> None:
    """JSONL 한 줄을 append.

    Args:
        path: 대상 파일 경로.
        record: 직렬화할 dict.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
