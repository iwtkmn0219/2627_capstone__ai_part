"""
턴 로그. 두 가지 목적을 동시에 만족.
  1) 학부모 대시보드의 데이터 소스
  2) P3 정량 평가의 원자료 (지연/비용/안전 판정)

JSONL 로 남기면 나중에 pandas 로 바로 분석.
개인정보가 들어가므로 child_id 는 반드시 익명 ID를 쓰고,
원문 저장 여부는 보호자 동의 설정과 연동할 것.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

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

    Attributes:
        path: 로그를 append 할 파일 경로.
    """

    def __init__(self, path: str = "logs/turns.jsonl"):
        """로그 파일 경로를 설정하고 부모 디렉터리를 준비.

        Args:
            path: 로그를 저장할 JSONL 파일 경로. 부모 디렉터리가 없으면 생성.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, child_id: str, session_id: str, result, store_text: bool = True):
        """한 턴의 결과를 로그 한 줄로 append.

        Args:
            child_id: 익명화된 아이 식별자.
            session_id: 세션 식별자.
            result: 기록할 TurnResult.
            store_text: False 면 발화/응답 원문은 저장하지 않음.

        Returns:
            로그에 실제로 기록된 dict.
        """
        tok = result.tokens or {}
        cost = estimate_cost(result.model, tok.get("input", 0), tok.get("output", 0))

        record = {
            "turn_id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "child_id": child_id,
            "session_id": session_id,
            "child_text": result.child_text if store_text else None,
            "reply_text": result.reply_text if store_text else None,
            "escalate": result.escalate,
            "safety_events": result.safety_events,
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "total_ms": round(result.total_ms, 1),
            "model": result.model,
            "tokens": tok,
            "est_cost_usd": round(cost, 8),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
