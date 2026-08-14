"""
파이프라인 전 구간의 추상 인터페이스

오케스트레이터가 특정 벤더 SDK를 직접 import 하지 않음

[원칙]
- 이 파일이 프로젝트의 핵심 설계 결정입니다.
- STT/LLM/TTS/Safety 를 모두 인터페이스로 두면, P3 벤치마크가 "구현체 하나 추가"로 끝납니다.
- 절대 오케스트레이터가 특정 벤더 SDK를 직접 import 하지 않도록 유지합니다.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

# ---------------------------------------------------------------- STT


@dataclass
class STTResult:
    """STT 엔진이 어떤 벤더든 동일한 형태로 돌려주는 인식 결과.

    Attributes:
        text: 인식된 최종 텍스트.
        confidence: 0~1 신뢰도. 제공하지 않는 엔진은 None.
        raw: 벤치마크용 엔진별 원본 응답. CER 계산 시 필요.
        engine: 구현체 이름 (STTEngine.name).
    """

    text: str
    confidence: float | None = None
    raw: dict = field(default_factory=dict)
    engine: str = ""


class STTEngine(ABC):
    """음성 -> 텍스트. CLOVA / GPT-4o-transcribe / RTZR / Whisper 를 같은 형태로 반환합니다.

    Attributes:
        name: 구현체 고유 이름.
    """

    name: str = "base"

    @abstractmethod
    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> STTResult:
        """오디오 바이트를 텍스트로 변환합니다.

        Args:
            audio: PCM/인코딩된 오디오 원본 바이트.
            sample_rate: 입력 오디오의 샘플레이트(Hz). 기본 16kHz.

        Returns:
            인식 결과를 담은 STTResult.
        """


# ---------------------------------------------------------------- LLM


@dataclass
class LLMResult:
    """LLM 한 턴의 생성 결과. 토큰 수는 비용 비교(P3)에 사용됩니다.

    Attributes:
        text: 생성된 응답 텍스트.
        input_tokens: 프롬프트 토큰 수. 캐시 히트 효과 측정용.
        output_tokens: 생성 토큰 수.
        model: 실제 호출된 모델 ID.
        blocked_by_vendor: 벤더가 자체 안전 필터로 차단한 경우 True (Gemini safety
            ratings 등).
        raw: 원본 응답 전문. 사후 분석용.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    blocked_by_vendor: bool = False
    raw: dict = field(default_factory=dict)


class LLMEngine(ABC):
    """구현체는 프롬프트를 static/dynamic으로 분리해서 받습니다.

    static = 페르소나 + 안전 지침 (매 턴 동일 -> 캐싱 대상)
    dynamic = 아이 프로필 + 대화 이력 + 현재 발화

    Attributes:
        name: 로그/벤치마크 집계 키.
    """

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        *,
        static_system: str,
        dynamic_context: str,
        user_utterance: str,
        max_output_tokens: int = 200,
        thinking: bool = False,
    ) -> LLMResult:
        """한 턴의 응답을 생성합니다.

        벤더 예외는 구현체에서 흡수하고 결과 객체로 반환하세요.

        Args:
            static_system: 매 턴 동일한 시스템 프롬프트. 프롬프트 캐싱 대상(페르소나).
            dynamic_context: 턴마다 바뀌는 컨텍스트 (프로필 + 이력).
            user_utterance: 이번 턴 아이 발화.
            max_output_tokens: 응답 길이 상한. 아이 대화는 짧게 유지합니다.
            thinking: 확장 사고 사용 여부. 지연이 늘어나므로 기본 off.

        Returns:
            생성된 응답을 담은 LLMResult.
        """


# ---------------------------------------------------------------- TTS


class TTSEngine(ABC):
    """텍스트 -> 음성. 목소리는 voice 파라미터로만 교체합니다.

    Attributes:
        name: 구현체 고유 이름.
    """

    name: str = "base"

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str) -> bytes:
        """text를 완성된 오디오 바이트 한 덩어리로 합성합니다.

        Args:
            text: 합성할 텍스트.
            voice: 벤더별 화자 ID. 매핑은 구현체가 책임집니다.

        Returns:
            합성된 오디오 바이트.
        """

    async def stream(self, text: str, *, voice: str) -> AsyncIterator[bytes]:
        """text를 오디오 청크 스트림으로 합성합니다.

        기본 구현은 논스트리밍입니다. 지연이 문제되면 구현체에서 오버라이드하세요.

        Args:
            text: 합성할 텍스트.
            voice: 벤더별 화자 ID.

        Yields:
            합성된 오디오 바이트 청크.
        """
        yield await self.synthesize(text, voice=voice)


# ---------------------------------------------------------------- Safety


class RiskLevel(int, Enum):
    """발화 하나의 심각도. 정의와 경계 예시는 capstone_documents/severity-levels.md 를 따릅니다.

    int 를 상속하므로 부등호로 임계값을 다룰 수 있고, 그대로 JSON 에 직렬화됩니다.

    Attributes:
        L0: 중립. 감정 신호 없음.
        L1: 일상 부정 감정. 원인이 분명하고 일과성.
        L2: 취약 신호. 외로움/위축/자기비하/무기력.
        L3: 안전 우려. 신체적 위해/공포/방치.
        L4: 보호 필요. 학대/방임 정황, 자해 사고.
    """

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


class Verdict(str, Enum):
    """안전 검사 판정.

    str 을 상속하므로 그대로 로그/JSON 에 직렬화됩니다.

    Attributes:
        ALLOW: 통과.
        REWRITE: 응답을 안전한 문장으로 교체.
        ESCALATE: 보호자에게 알림 (예: 아이의 위기 신호).
        BLOCK: 응답하지 않고 차단.
    """

    ALLOW = "allow"
    REWRITE = "rewrite"
    ESCALATE = "escalate"
    BLOCK = "block"


@dataclass
class SafetyResult:
    """검사 1회의 판정 결과와 판단 근거.

    Attributes:
        verdict: 판정 결과.
        checker: 어떤 checker 가 걸었는지. P3 실험의 원자료가 됩니다.
        categories: 위반 카테고리 목록.
        score: 분류기 점수. 룰 기반 checker 는 None.
        replacement: REWRITE 일 때 대체할 문장.
        matched_text: 오탐 디버깅용. 판정 대상 전문은 checker 가 아니라 오케스트레이터가 로그에 남김.
        latency_ms: 검사에 걸린 시간(ms). 파이프라인 지연 예산 측정용.
        level: 이 발화의 심각도. 출력 검사에서는 의미가 없으므로 L0.
        ongoing: 지금 벌어지는 일인지. 레벨은 그대로 두고 응답 방식만 바꿉니다.
        alleged_target: 아이가 가해 주체로 지목한 호칭. 원문 그대로. 없으면 None.
        target_certain: 지목이 유일하게 특정됐는지. 라우터가 해소 실패를 판단하는 근거.
    """

    verdict: Verdict
    checker: str = ""
    categories: list[str] = field(default_factory=list)
    score: float | None = None
    replacement: str | None = None
    matched_text: str | None = None
    latency_ms: float = 0.0
    level: RiskLevel = RiskLevel.L0
    ongoing: bool = False
    alleged_target: str | None = None
    target_certain: bool = False


@dataclass
class RiskSignal:
    """한 턴의 위험 신호. 라우팅 담당(백엔드)에 넘기는 계약 객체.

    알림 대상 선정은 여기서 하지 않습니다. 등록 어른 관계 그래프가 필요하고
    그건 우리 파트의 데이터가 아니기 때문입니다. 우리는 '무엇을 감지했는가'
    까지만 싣고, '누구에게 알릴 것인가'는 라우터가 정합니다.

    Attributes:
        level: 승격 규칙까지 반영한 최종 심각도.
        categories: 걸린 위험 범주 목록.
        ongoing: 지금 벌어지는 일인지.
        alleged_target: 지목 호칭. 정규화하지 않습니다. 지목이 없으면 None.
        target_certain: 지목이 유일하게 특정됐는지. False 면 라우터는 해소에 실패한
            것으로 보고 억제해야 합니다.
        notify_allowed: 알림을 보내도 되는지. L4 는 항상 False 이며, 라우터가 지목
            해소에 성공한 뒤에만 뒤집을 수 있습니다. 신호가 유실되거나 필드를 읽지
            못했을 때 알림이 나가지 않도록 fail-closed 로 둡니다.
    """

    level: RiskLevel = RiskLevel.L0
    categories: list[str] = field(default_factory=list)
    ongoing: bool = False
    alleged_target: str | None = None
    target_certain: bool = False
    notify_allowed: bool = True


class SafetyChecker(ABC):
    """입력(발화)과 출력(응답) 양쪽에 꽂을 수 있는 필터.

    직접 학습한 KoELECTRA 분류기도 이 인터페이스만 맞추면 바로 투입할 수 있습니다.

    Attributes:
        name: 로그/벤치마크 집계 키. SafetyResult.checker 에 그대로 들어갑니다.
        stage: 꽂을 지점. "input" | "output" | "both".
    """

    name: str = "base"
    stage: str = "both"

    @abstractmethod
    async def check(self, text: str, *, context: dict | None = None) -> SafetyResult:
        """text 하나를 검사합니다.

        Args:
            text: 검사할 문장.
            context: 아이 프로필/직전 대화 등 판정에 참고할 부가 정보. 없으면 None.

        Returns:
            판정 결과를 담은 SafetyResult.
        """
