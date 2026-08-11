"""
턴 오케스트레이터

STT -> 안전(입력) -> LLM -> 안전(출력) -> TTS

[설계 원칙]
- 벤더 SDK는 여기서 import 하지 않고, 전부 인터페이스로만 다룬다.
- 모든 구간의 지연을 계측한다.
- 안전 판정은 전부 로그로 남긴다.
- 위 두가지는 P3 디벨롭에 사용한다.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from core.interfaces import (
    LLMEngine,
    SafetyChecker,
    SafetyResult,
    STTEngine,
    TTSEngine,
    Verdict,
)
from context.persona import PERSONA, build_dynamic_context
from safety.checkers import SAFE_FALLBACK


@dataclass
class TurnResult:
    """한 턴의 최종 산출물과 P3 분석에 필요한 계측값.

    Attributes:
        child_text: 아이 발화. STT 로 받아온 텍스트.
        reply_text: 안전 검사까지 통과한 최종 응답. 차단 시 대체 문장.
        audio: TTS 결과. synthesize=False 면 None.
        escalate: True 면 보호자 알림 필요 (위기 신호 감지).
        model: 응답을 생성한 모델 이름. 생성을 건너뛴 턴은 빈 문자열. (단가 비교용)
        safety_events: 검사 판정 전량 로그.
        timings_ms: 구간별 지연(ms). {stt, safety_in, llm, tts} 키를 가질 수 있음.
        tokens: {input, output, cached} 토큰 수.
    """

    child_text: str
    reply_text: str
    audio: bytes | None = None
    escalate: bool = False
    model: str = ""
    safety_events: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        # 계측한 구간의 합.
        return sum(self.timings_ms.values())


class TurnOrchestrator:
    """구현체를 주입받아 한 턴을 실행.

    전부 인터페이스 타입으로 실행하므로, P3 벤치마크에서 엔진을 교체시 이 클래스는 수정 금지
    """

    def __init__(
        self,
        *,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        input_checkers: list[SafetyChecker],
        output_checkers: list[SafetyChecker],
        max_regenerations: int = 1,
    ):
        """엔진 구현체를 주입받아 오케스트레이터를 구성.

        Args:
            stt: 음성 인식 엔진.
            llm: 응답 생성 엔진.
            tts: 음성 합성 엔진.
            input_checkers: 아이 발화를 검사할 체커들.
            output_checkers: 인공지능 응답을 검사할 체커들.
            max_regenerations: 출력 필터 실패 시 재생성 횟수. 총 시도 횟수는 값 + 1.
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.input_checkers = input_checkers
        self.output_checkers = output_checkers
        self.max_regenerations = max_regenerations

    async def run(
        self,
        audio: bytes,
        *,
        profile: dict,
        history: list[tuple[str, str]],
        memory: list[str] | None = None,
        synthesize: bool = True,
    ) -> TurnResult:
        """오디오 한 덩어리를 받아 응답 오디오까지 생성 후 반환.

        Args:
            audio: 아이 발화 오디오 원본 바이트.
            profile: 아이 프로필. name/age 는 필수, interests/voice 는 선택.
            history: 최근 대화 이력. [(아이 발화, 인공지능 응답), ...] 형식.
            memory: 장기기억에서 검색해온 문장들. 없으면 None.
            synthesize: False 면 TTS 를 건너뜁니다. 텍스트만 필요한 벤치마크에서 사용.

        Returns:
            이번 턴의 결과를 담은 TurnResult.
        """
        timings: dict[str, float] = {}  # 지연률 측정용
        events: list[dict] = []  # 이벤트 확인용

        # 1. STT
        t = time.perf_counter()  # 성능 검증용 타이머
        stt_result = await self.stt.transcribe(audio)
        timings["stt"] = (time.perf_counter() - t) * 1000
        child_text = stt_result.text

        # 인식 실패(무음/잡음)는 정상 흐름으로 취급
        # LLM 을 호출하지 않고 되묻는 문장으로 조기 반환 (API 절약합시다. 돈 아껴야해요. 근.검.절.약.)
        if not child_text:
            return TurnResult(
                child_text="",
                reply_text="잘 못 들었어! 다시 한 번 말해줄래?",
                timings_ms=timings,
            )

        # 2. 입력 안전 검사 (병렬)
        # 체커가 늘어나도 지연이 누적되지 않도록 gather 사용. 가장 느린 체커만큼만 지연.
        t = time.perf_counter()
        escalate = False
        forced_reply = None  # 값이 있으면 LLM 응답을 무시하고 해당 문장 반환
        results = await asyncio.gather(
            *(
                c.check(child_text, context={"stage": "input"})
                for c in self.input_checkers
            )
        )
        for r in results:
            events.append(_event("input", r, child_text))
            if r.verdict is Verdict.ESCALATE:
                # 위기 신호: 대화는 계속하되 보호자 알림 플래그 사용.
                # replacement 가 없으면 LLM 응답을 그대로 사용 (침묵보다 대화 유지가 안전)
                escalate = True
                forced_reply = r.replacement or forced_reply
            elif r.verdict is Verdict.BLOCK:
                forced_reply = r.replacement or SAFE_FALLBACK
        timings["safety_in"] = (time.perf_counter() - t) * 1000

        # 3. LLM 생성 (+ 출력 필터 실패 시 1회 재생성)
        # 입력 검사에서 강제 응답이 정해졌으면(BLOCK, 또는 replacement 를 준 ESCALATE) 생성 스킵
        # 어차피 버릴 응답에 토큰과 지연을 쓸 이유가 없음
        # ESCALATE 이면서 replacement 가 없는 경우는 forced_reply 가 None 이므로 정상 생성
        reply, tokens, model = "", {}, ""
        if forced_reply is None:
            dynamic = build_dynamic_context(
                child_name=profile["name"],
                age=profile["age"],
                interests=profile.get("interests", []),
                recent_turns=history,
                retrieved_memory=memory,
            )

            # 주의: 이 타이머에는 _screen_output(출력 안전 검사) 시간도 포함.
            # 재생성이 일어나면 생성 N회 + 검사 N회가 모두 "llm" 한 칸에 누적.
            t = time.perf_counter()
            for attempt in range(self.max_regenerations + 1):
                llm_out = await self.llm.generate(
                    static_system=PERSONA,
                    dynamic_context=dynamic,
                    user_utterance=child_text,
                    max_output_tokens=200,
                    thinking=False,  # 음성 턴 지연 예산 때문에 사고 비활성
                )
                tokens = {
                    "input": llm_out.input_tokens,
                    "output": llm_out.output_tokens,
                    "cached": llm_out.raw.get("cached_tokens", 0),
                }
                model = llm_out.model
                reply = llm_out.text or SAFE_FALLBACK

                # 벤더 자체 필터에 걸린 경우 즉시 포기.
                if llm_out.blocked_by_vendor:
                    # 직접 dict 를 만들면 스키마가 어긋나므로 SafetyResult 를 거친다.
                    events.append(
                        _event(
                            "output",
                            SafetyResult(verdict=Verdict.BLOCK, checker="vendor"),
                            llm_out.text,
                            attempt=attempt,
                        )
                    )
                    reply = SAFE_FALLBACK
                    break

                verdict, repl = await self._screen_output(reply, events, attempt)
                if verdict is Verdict.ALLOW:
                    break
                if repl:
                    # 체커가 대체 문장을 준 경우(REWRITE): 재생성보다 싸고 빠르므로 그걸 채택.
                    reply = repl
                    break
                # 대체 문장이 없으면 루프를 돌아 재생성. 마지막 시도까지 실패하면 고정 문장으로 마감.
                if attempt == self.max_regenerations:
                    reply = SAFE_FALLBACK
            timings["llm"] = (time.perf_counter() - t) * 1000
        else:
            reply = forced_reply

        # 4. TTS
        audio_out = None
        if synthesize:
            t = time.perf_counter()
            audio_out = await self.tts.synthesize(
                reply, voice=profile.get("voice", "default")
            )
            timings["tts"] = (time.perf_counter() - t) * 1000

        return TurnResult(
            child_text=child_text,
            reply_text=reply,
            audio=audio_out,
            escalate=escalate,
            model=model,
            safety_events=events,
            timings_ms=timings,
            tokens=tokens,
        )

    async def _screen_output(self, text: str, events: list[dict], attempt: int = 0):
        """인공지능 응답을 출력 체커 전체에 병렬로 통과.

        판정은 심각도 순이 아니라 '마지막에 걸린 체커'의 값이고,
        대체 문장은 반대로 '처음 제시한 체커'의 것을 사용(or 단축평가).
        체커 순서가 결과를 바꾸므로, 우선순위가 논의에 따라 정렬을 추가할 것.

        Args:
            text: 검사할 응답 텍스트.
            events: 판정 이벤트를 누적할 리스트.
            attempt: 몇 번째 생성 결과인지. 재생성 초안을 구분하는 데 쓴다.

        Returns:
            (판정, 대체 문장) 튜플. 판정이 ALLOW 가 아니면 호출부가 재생성 여부를 결정.
        """
        results = await asyncio.gather(
            *(c.check(text, context={"stage": "output"}) for c in self.output_checkers)
        )
        worst, replacement = Verdict.ALLOW, None
        for r in results:
            events.append(_event("output", r, text, attempt=attempt))
            if r.verdict is not Verdict.ALLOW:
                worst = r.verdict
                replacement = replacement or r.replacement
        return worst, replacement


def _event(stage: str, r, text: str, attempt: int | None = None) -> dict:
    """SafetyResult 를 로그/분석용 dict 로 변환.

    판정 대상 원문(text)을 체커가 아니라 여기서 싣는다. 체커가 깜빡할 여지가
    없어지고, P2 에서 붙일 분류기도 그대로 혜택을 받는다.

    Args:
        stage: "input" 또는 "output".
        r: 변환할 SafetyResult.
        text: 이 판정이 실제로 검사한 원문.
        attempt: 출력 검사의 생성 시도 회차(0부터). 입력 검사는 None.

    Returns:
        로그 적재용 dict.
    """
    return {
        "stage": stage,
        "checker": r.checker,
        "verdict": r.verdict.value,
        "categories": r.categories,
        "score": r.score,
        "latency_ms": round(r.latency_ms, 2),
        "text": text,
        "matched_text": r.matched_text,
        "attempt": attempt,
    }
