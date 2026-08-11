"""
안전 계층. 이 프로젝트에서 '직접 만든 것'에 해당하는 핵심 모듈.

3단 방어:
  L1 벤더 필터   (Gemini safety_settings) - 생성 단계에서 차단
  L2 범용 모더레이션 (OpenAI Moderation API, 무료) - 성인 기준 유해성
  L3 아동 특화 분류기 (직접 학습) - "성인 기준으로는 무해하지만 아동에겐 부적절" 을 잡는 레이어.

L3가 존재해야 하는 이유:
범용 모더레이션은 '유해함'을 본다. 하지만 아동 서비스에서 문제가 되는 건
- 어휘 수준 초과 (여섯 살에게 "타협점을 모색해보자")
- 보호자 역할 침범 ("엄마한텐 비밀로 하자")
- 공포/불안 유발 (죽음, 실종, 유괴를 사실적으로 묘사)
- 의료/안전 조언 ("배 아프면 이 약 먹어")
이건 기존 API가 전혀 잡지 못함. 그래서 직접 만들 가치가 있음.
"""

from __future__ import annotations

import re
import time

from core.interfaces import SafetyChecker, SafetyResult, Verdict

SAFE_FALLBACK = (
    "민준아 뭐가 잘못됐다 그지?\n음… 난 잘 모르겠어! 우리 다른 이야기 해볼까?"
)
ESCALATE_REPLY = (
    "그랬구나. 나는 네 얘기 계속 듣고 싶어. 그 얘기, 엄마 아빠한테도 해볼까?"
)


class RuleChecker(SafetyChecker):
    """L3의 부트스트랩 버전. 규칙 기반이라 성능은 낮지만,
    (1) 학습 데이터 라벨링 전에 당장 서비스를 안전하게 굴리고
    (2) 규칙이 잡은 사례가 곧 분류기 학습 데이터가 됨.
    P2에서 KoELECTRA 분류기로 교체하되, 이 규칙은 앙상블로 남겨둘 것.
    """

    name = "rule_v1"
    stage = "both"

    # 보호자로부터 아이를 분리시키는 표현 - 아동 서비스에서 가장 위험한 범주
    ISOLATION = re.compile(r"(비밀로|말하지\s*마|둘만|아무한테도)")
    # 아이의 위기 신호 - 차단이 아니라 보호자 에스컬레이션 대상
    DISTRESS = re.compile(r"(때렸|맞았|아파서\s*울|무서워서|죽고\s*싶|아무도\s*없)")
    # 인공지능 해서는 안 되는 조언
    UNSAFE_ADVICE = re.compile(r"(약을?\s*먹|불을?\s*켜|칼|가위로|혼자\s*나가)")

    async def check(self, text: str, *, context: dict | None = None) -> SafetyResult:
        """정규식 규칙으로 위험 표현을 검사.

        stage 를 주지 않으면 "both" 로 보고 입력/출력 규칙을 모두 적용한다.

        Args:
            text: 검사 대상 문장. 아이 발화 또는 인공지능 응답.
            context: 부가 정보 dict. "stage" 키로 "input"/"output" 구분. 생략하면 "both".

        Returns:
            판정 결과를 담은 SafetyResult.
        """
        t0 = time.perf_counter()
        stage = (context or {}).get("stage", "both")

        def done(verdict, cats, repl=None, match=None):
            """SafetyResult 생성 단축 헬퍼.

            Args:
                verdict: 판정 결과. ALLOW/REWRITE/ESCALATE 등.
                cats: 걸린 위험 범주 이름 목록.
                repl: 원문 대신 내보낼 교체 문장. 없으면 None.
                match: 규칙이 걸린 re.Match. 어느 표현에 걸렸는지 로그에 남긴다.

            Returns:
                조립된 SafetyResult.
            """
            return SafetyResult(
                verdict=verdict,
                checker=self.name,
                categories=cats,
                replacement=repl,
                matched_text=match.group(0) if match else None,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # 아이 발화에서 위기 신호 -> 대화는 계속하되 보호자 알림
        if stage in ("input", "both") and (m := self.DISTRESS.search(text)):
            return done(Verdict.ESCALATE, ["child_distress"], ESCALATE_REPLY, m)

        if stage in ("output", "both"):
            if m := self.ISOLATION.search(text):
                return done(
                    Verdict.REWRITE,
                    ["guardian_isolation"],
                    "type: 1\n" + SAFE_FALLBACK,
                    m,
                )
            if m := self.UNSAFE_ADVICE.search(text):
                return done(
                    Verdict.REWRITE, ["unsafe_advice"], "type: 2\n" + SAFE_FALLBACK, m
                )

        return done(Verdict.ALLOW, [])


class ReadabilityChecker(SafetyChecker):
    """'아이 눈높이'를 정량적으로 강제하는 필터.

    프롬프트만으로는 모델이 종종 어른 말투로 돌아감. 출력단에서 잡을 것.
    임계값은 대상 연령대(4-7세)에 맞춰 실측 후 조정할 것.
    """

    name = "readability_v1"
    stage = "output"

    def __init__(self, max_sentences: int = 3, max_chars_per_sentence: int = 40):
        """문장 수/길이 상한을 설정.

        Args:
            max_sentences: 한 응답에 허용할 최대 문장 수.
            max_chars_per_sentence: 문장 하나에 허용할 최대 글자 수.
        """
        self.max_sentences = max_sentences
        self.max_chars = max_chars_per_sentence

    async def check(self, text: str, *, context: dict | None = None) -> SafetyResult:
        """응답문의 길이와 문장 수를 검사.

        Args:
            text: 길이/문장 수를 검사할 응답문.
            context: 인터페이스 통일용 부가 정보. 이 체커에서는 사용하지 않음.

        Returns:
            판정 결과를 담은 SafetyResult.
        """
        t0 = time.perf_counter()
        sentences = [s for s in re.split(r"[.!?~]\s*", text) if s.strip()]

        too_long = len(sentences) > self.max_sentences
        over = [s for s in sentences if len(s) > self.max_chars]

        verdict = Verdict.REWRITE if (too_long or over) else Verdict.ALLOW
        cats = []
        if too_long:
            cats.append("too_many_sentences")
        if over:
            cats.append("sentence_too_long")

        return SafetyResult(
            verdict=verdict,
            checker=self.name,
            categories=cats,
            # REWRITE 지만 교체문이 아니라 '재생성 요청' 신호로 사용.
            replacement=None,
            # 길이를 넘긴 첫 문장. 임계값을 조정할 때 무엇이 걸렸는지 보려고 남긴다.
            matched_text=over[0] if over else None,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class ClassifierChecker(SafetyChecker):
    """P2에서 여러분이 직접 학습할 KoELECTRA 기반 아동 적합성 분류기 자리.

    RTX 5060 Ti 8GB 에서 충분히 파인튜닝 가능하고, 추론은 CPU로도 싸게 돌아감.

    학습 데이터는 RuleChecker 가 잡은 로그 + 수동 라벨링으로 시작할 것.
    (라벨: safe / age_inappropriate / guardian_isolation / unsafe_advice / distress)
    """

    name = "koelectra_v0"
    stage = "output"

    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        """분류기 경로와 차단 임계값을 설정.

        Args:
            model_path: 학습된 분류기 가중치 경로. None이면 미학습 상태로 항상 통과.
            threshold: 부적합으로 판정할 확률 임계값. 이 값 이상이면 차단.
        """
        self.model_path = model_path
        self.threshold = threshold
        self._model = None  # TODO(P2): transformers 로 로드

    async def check(self, text: str, *, context: dict | None = None) -> SafetyResult:
        """text 하나를 KoELECTRA 분류기로 검사.

        Args:
            text: 분류기에 입력할 검사 대상 문장.
            context: 부가 정보 dict. P2에서 연령·대화 이력 등을 넘길 자리.

        Returns:
            판정 결과를 담은 SafetyResult. 미학습 상태에서는 항상 ALLOW.
        """
        if self._model is None:
            # 미학습 상태에서는 항상 통과 (파이프라인은 지금부터 굴러가야 하므로)
            return SafetyResult(verdict=Verdict.ALLOW, checker=self.name)
        raise NotImplementedError("P2에서 구현")
