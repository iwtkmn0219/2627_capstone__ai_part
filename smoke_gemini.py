"""
Gemini 실측 스모크 테스트.

    pip install -r requirements.txt
    echo "GEMINI_API_KEY=..." > .env
    python smoke_gemini.py

확인 목적:
  1) 실제 턴 지연 (P3 기준선)
  2) 캐시가 실제로 걸리는지  <- 가장 중요
  3) 실제 토큰 수와 비용
  4) 페르소나가 아이 눈높이로 작동하는지 육안 확인

STT/TTS 는 아직 Fake 를 쓴다. LLM 구간만 순수하게 재기 위해서.
"""

from __future__ import annotations

import asyncio
import statistics

from dotenv import load_dotenv
from core.fakes import FakeSTT, FakeTTS
from core.orchestrator import TurnOrchestrator
from llm.gemini import GeminiEngine
from safety.checkers import ReadabilityChecker, RuleChecker
from telemetry.logger import TurnLogger, estimate_cost

load_dotenv()

PROFILE = {
    "name": "영택",
    "age": 7,
    "interests": ["운동", "축구 보기"],
    "voice": "child_ko",
}

# 아이의 발화 예시 모음
UTTERANCES = [
    "앵쵸야 안녕!",
    "나 오늘 헬스 하고 왔어",
    "티라노사우루스가 제일 세?",
    "앵쵸는 무슨 색 좋아해?",
    "나 오늘 유치원에서 친구랑 싸웠어",
    "심심해",
    "앵쵸 우리 수수께끼 하자",
    "밥 먹기 싫어",
    "엄마 언제 와?",
    "앵쵸는 몇 살이야?",
    "오늘 칼국수 먹었어",
]


async def main():
    llm = GeminiEngine()  # GEMINI_MODEL 환경변수로 지정
    logger = TurnLogger("logs/smoke.jsonl")

    # STT -> LLM -> TTS 파이프라인
    orch = TurnOrchestrator(
        stt=FakeSTT(""),  # 발화마다 교체할 수도
        llm=llm,
        tts=FakeTTS(),
        input_checkers=[RuleChecker()],
        output_checkers=[RuleChecker(), ReadabilityChecker()],
    )

    history: list[tuple[str, str]] = []
    rows = []

    for i, utt in enumerate(UTTERANCES, 1):
        orch.stt = FakeSTT(utt)
        result = await orch.run(
            b"<audio>", profile=PROFILE, history=history, synthesize=False
        )
        logger.log(child_id="anon_smoke", session_id="smoke", result=result)

        history.append(("child", utt))
        history.append(("aengcho", result.reply_text))

        tok = result.tokens
        rows.append(
            {
                "llm_ms": result.timings_ms.get("llm", 0),
                "in": tok.get("input", 0),
                "out": tok.get("output", 0),
                "cached": tok.get("cached", 0),
            }
        )

        flags = [e["checker"] for e in result.safety_events if e["verdict"] != "allow"]
        print(f"\n[{i:02d}] 아이 : {utt}")
        print(f"     앵쵸 : {result.reply_text}")
        print(
            f"     지연 {rows[-1]['llm_ms']:.0f}ms | "
            f"토큰 in {rows[-1]['in']} (캐시 {rows[-1]['cached']}) out {rows[-1]['out']}"
            + (f" | 필터 {flags}" if flags else "")
        )

    _report(rows, llm.model)


def _report(rows: list[dict], model: str):
    """턴별 측정치를 모아 지연/토큰/비용 리포트를 출력.

    Args:
        rows: 턴별 {llm_ms, in, out, cached} 딕셔너리 리스트.
        model: 비용 계산에 사용할 모델 이름.
    """
    lat = [r["llm_ms"] for r in rows]
    total_in = sum(r["in"] for r in rows)
    total_out = sum(r["out"] for r in rows)
    total_cached = sum(r["cached"] for r in rows)
    cost = estimate_cost(model, total_in, total_out)

    print("\n" + "=" * 52)
    print(f"모델             : {model}")
    print(f"턴 수            : {len(rows)}")
    print(f"LLM 지연 평균    : {statistics.mean(lat):.0f}ms")
    print(f"LLM 지연 중앙값  : {statistics.median(lat):.0f}ms")
    print(f"LLM 지연 최대    : {max(lat):.0f}ms")
    print(f"입력 토큰 합     : {total_in}  (턴당 {total_in/len(rows):.0f})")
    print(f"출력 토큰 합     : {total_out}  (턴당 {total_out/len(rows):.0f})")
    print(f"캐시된 토큰 합   : {total_cached}")
    print(f"실측 비용        : ${cost:.6f}  (턴당 ${cost/len(rows):.6f})")
    print(f"→ 1만 턴 환산    : ${cost/len(rows)*10000:.2f}")
    print("=" * 52)

    if total_cached == 0:
        print(
            "\n[주의] 캐시된 토큰이 0입니다.\n"
            "  암묵적 캐싱은 프롬프트가 최소 토큰 수를 넘어야 걸립니다.\n"
            "  페르소나 블록이 짧으면 캐시가 안 잡힐 수 있습니다.\n"
            "  → 대화 이력이 쌓인 뒤에도 계속 0이면 명시적 캐싱(CachedContent) 검토."
        )
    if statistics.mean(lat) > 2000:
        print(
            "\n[주의] LLM 지연이 2초를 넘습니다. 음성 파이프라인에서는 체감이 큽니다."
        )


if __name__ == "__main__":
    asyncio.run(main())
