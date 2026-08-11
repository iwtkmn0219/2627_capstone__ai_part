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

from core.interfaces import RiskLevel, SafetyChecker, SafetyResult, Verdict

SAFE_FALLBACK = (
    "민준아 뭐가 잘못됐다 그지?\n음… 난 잘 모르겠어! 우리 다른 이야기 해볼까?"
)

# L3 이면서 지금 벌어지는 일. 대화를 이어가는 것보다 곁의 어른에게 연결하는 게 먼저다.
URGENT_ADULT_REPLY = "옆에 어른 있어? 가서 보여주자.\n앵쵸는 여기서 기다릴게."

# L4 전용. 보호자를 특정해서 권하지 않는다.
# 지목된 어른이 누구인지 모르는 상태에서 "엄마 아빠한테 말해볼까?" 는 아이를
# 가해자 쪽으로 밀 수 있다. 알림 대상 선정은 관계 그래프를 가진 라우터의 몫이고,
# 이 자리에서 할 수 있는 건 판정하지 않고, 캐묻지 않고, 계속 듣는 것뿐이다.
# 캐묻지 않는 건 예의 문제가 아니다. 반복·유도 질문은 이후 조사·사법 절차에서
# 아동 진술의 신빙성을 훼손한다.
PROTECT_REPLY = "얘기해줘서 고마워.\n그건 네 잘못이 아니야.\n앵쵸는 계속 듣고 있을게."

# 아이가 부를 만한 양육 어른 호칭. 이 대상이 지목되면 L4 로 본다.
# 부모만이 아니라 교사/학원 강사도 들어간다. 지목 대상이 등록 어른 중 누구인지는
# 여기서 해소하지 않는다. 원문 그대로 실어 보내고 해소는 라우터가 한다.
CAREGIVER_LABELS = (
    "새아빠",
    "새엄마",
    "아빠",
    "아버지",
    "엄마",
    "어머니",
    "할머니",
    "할아버지",
    "선생님",
    "원장님",
    "이모",
    "삼촌",
    "고모",
)
# 또래·형제. 지목되면 L3. 보호 필요 상황이 아니고 등록 어른과 무관하다.
PEER_LABELS = ("친구", "형", "누나", "오빠", "언니", "동생")

_LABEL_RE = re.compile("|".join(CAREGIVER_LABELS + PEER_LABELS))
_PROGRESSIVE = re.compile(r"(고\s*있|는\s*중|지금|방금)")
_PAST = re.compile(r"(었|았)|어제|아까|저번|지난")


def _has_past_marker(text: str) -> bool:
    """과거 시제 표지가 있는지 본다.

    '-었/았' 은 '줬어', '때렸어' 처럼 앞 음절의 ㅆ 받침으로 축약되는 경우가 많아
    문자열 매칭만으로는 절반을 놓친다. 종성이 ㅆ 인 음절을 직접 확인한다.

    Args:
        text: 검사할 문장.

    Returns:
        과거 표지가 있으면 True.
    """
    if _PAST.search(text):
        return True
    for ch in text:
        # '있어' 는 과거가 아니라 존재 표현이므로 제외한다.
        if ch == "있" or not ("가" <= ch <= "힣"):
            continue
        if (ord(ch) - 0xAC00) % 28 == 20:  # 종성 ㅆ
            return True
    return False


def _is_ongoing(text: str) -> bool:
    """지금 벌어지는 일인지 판정.

    다섯 살은 어제 일을 현재형으로 말한다. 반대로 오판하면 지금 위험한 아이를
    위로만 하고 끝내게 되므로, 심각도.md 의 tie-break 대로 애매하면 True 로 둔다.

    Args:
        text: 검사할 문장.

    Returns:
        진행 중으로 보이면 True.
    """
    if _PROGRESSIVE.search(text):
        return True
    return not _has_past_marker(text)


def _extract_target(text: str, before: int) -> tuple[str | None, bool]:
    """가해 주체로 지목된 호칭을 원문 그대로 뽑는다.

    가해 표현 앞에 나온 마지막 호칭을 주체로 본다. P1 규칙 수준의 조악한 추출이라
    문장이 복잡하면 틀린다. 틀린 값을 라우터가 그대로 믿지 않도록 확신 여부를
    함께 돌려주고, 여기서 '선생님'을 담임으로 좁히는 식의 추론은 하지 않는다.

    Args:
        text: 아이 발화 전문.
        before: 가해 표현이 시작된 위치. 이 앞에서만 호칭을 찾는다.

    Returns:
        (호칭, 유일 특정 여부) 튜플. 지목이 없으면 (None, False).
    """
    head = text[:before]
    last = None
    for m in _LABEL_RE.finditer(head):
        last = m
    if last is None:
        return None, False
    # 주격 조사가 붙어 있으면 가해 주체로 볼 근거가 조금 더 강하다.
    return last.group(0), head[last.end() : last.end() + 1] in ("가", "이")


def _signal(
    level: RiskLevel,
    category: str,
    match: re.Match,
    *,
    ongoing: bool = False,
    target: str | None = None,
    certain: bool = False,
) -> SafetyResult:
    """레벨에서 판정과 응답 문장을 유도해 SafetyResult 를 만든다.

    응답 선택 정책을 이 함수 하나에만 둔다. 규칙이 늘어나도 "어느 레벨에서 무슨
    말을 하는가" 가 여러 군데로 흩어지지 않는다.

    Args:
        level: 이 발화의 심각도.
        category: 걸린 위험 범주 이름.
        match: 규칙이 걸린 re.Match. 어느 표현에 걸렸는지 로그에 남긴다.
        ongoing: 지금 벌어지는 일인지.
        target: 지목 호칭. 원문 그대로.
        certain: 지목이 유일하게 특정됐는지.

    Returns:
        조립된 SafetyResult.
    """
    if level >= RiskLevel.L4:
        # 고정 문장을 쓴다. 생성 모델에 맡기면 "엄마한테 말해봐" 로 돌아갈 수 있고,
        # 지목된 어른이 누구인지 모르는 이상 그건 감수할 수 없는 위험이다.
        verdict, reply = Verdict.ESCALATE, PROTECT_REPLY
    elif level >= RiskLevel.L3:
        # 진행 중이면 어른 호출로 끊고, 아니면 생성 모델이 공감 응답을 만들게 둔다.
        # replacement 가 None 이면 오케스트레이터는 LLM 생성을 그대로 진행한다.
        verdict, reply = Verdict.ESCALATE, URGENT_ADULT_REPLY if ongoing else None
    else:
        # L2 이하는 대화를 건드리지 않는다. 레벨만 기록해 누적 판정 재료로 쓴다.
        verdict, reply = Verdict.ALLOW, None

    return SafetyResult(
        verdict=verdict,
        categories=[category],
        replacement=reply,
        matched_text=match.group(0),
        level=level,
        ongoing=ongoing,
        alleged_target=target,
        target_certain=certain,
    )


class RuleChecker(SafetyChecker):
    """L3의 부트스트랩 버전. 규칙 기반이라 성능은 낮지만,
    (1) 학습 데이터 라벨링 전에 당장 서비스를 안전하게 굴리고
    (2) 규칙이 잡은 사례가 곧 분류기 학습 데이터가 됨.
    P2에서 KoELECTRA 분류기로 교체하되, 이 규칙은 앙상블로 남겨둘 것.
    """

    name = "rule_v1"
    stage = "both"

    # --- 아이 발화 (입력) ---------------------------------------------------
    # 자해. 지목과 무관하게 L4 이고 시제도 판정하지 않는다.
    SELF_HARM = re.compile(r"(죽고\s*싶|죽을래|사라지고\s*싶|없어지고\s*싶)")
    # 폭력. 지목 대상에 따라 L3(또래)와 L4(양육 어른)로 갈린다.
    VIOLENCE = re.compile(r"(때리|때렸|때려|맞았|맞아서|꼬집|밀쳤)")
    # 방임. 지목이 없어도 L4.
    NEGLECT = re.compile(r"(밥\s*안\s*줬|굶|안\s*씻|혼자\s*잤|아무도\s*안\s*왔)")
    # 안전 우려. L3.
    SAFETY_CONCERN = re.compile(
        r"(다쳤|다쳐|피가?\s*나|아파서\s*울|무서워|무서웠|혼자\s*있|혼자\s*오래)"
    )
    # 취약 신호. L2. 단발로는 조치가 없고 누적 대상이 된다.
    VULNERABLE = re.compile(
        r"(아무도\s*없|한심|아무것도\s*하기\s*싫|계속\s*기분이\s*나빠)"
    )

    # --- 인공지능 응답 (출력) -----------------------------------------------
    # 보호자로부터 아이를 분리시키는 표현 - 아동 서비스에서 가장 위험한 범주
    ISOLATION = re.compile(r"(비밀로|말하지\s*마|둘만|아무한테도)")
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

        result = None
        if stage in ("input", "both"):
            result = self._screen_child(text)
        if result is None and stage in ("output", "both"):
            result = self._screen_reply(text)
        if result is None:
            result = SafetyResult(verdict=Verdict.ALLOW)

        result.checker = self.name
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _screen_child(self, text: str) -> SafetyResult | None:
        """아이 발화에서 심각도 신호를 찾는다.

        레벨이 높은 규칙부터 본다. 심각도.md 의 '애매하면 높은 쪽' 원칙에 따라
        먼저 걸린 것을 그대로 채택하고 아래 규칙은 보지 않는다.

        Args:
            text: 아이 발화.

        Returns:
            걸린 신호가 있으면 SafetyResult, 없으면 None.
        """
        if m := self.SELF_HARM.search(text):
            return _signal(RiskLevel.L4, "self_harm", m, ongoing=True)

        if m := self.VIOLENCE.search(text):
            target, certain = _extract_target(text, m.start())
            peer = target in PEER_LABELS
            level = RiskLevel.L3 if peer else RiskLevel.L4
            # 양육 어른에 의한 폭력은 아이가 그 환경으로 돌아가므로 진행 중으로 본다.
            # 또래 폭력은 시제대로 판정한다.
            return _signal(
                level,
                "violence",
                m,
                ongoing=_is_ongoing(text) if peer else True,
                target=target,
                certain=certain,
            )

        if m := self.NEGLECT.search(text):
            target, certain = _extract_target(text, m.start())
            return _signal(
                RiskLevel.L4,
                "neglect",
                m,
                ongoing=_is_ongoing(text),
                target=target,
                certain=certain,
            )

        if m := self.SAFETY_CONCERN.search(text):
            return _signal(RiskLevel.L3, "safety_concern", m, ongoing=_is_ongoing(text))

        if m := self.VULNERABLE.search(text):
            return _signal(RiskLevel.L2, "vulnerable", m)

        return None

    def _screen_reply(self, text: str) -> SafetyResult | None:
        """인공지능 응답에서 아동 부적합 표현을 찾는다.

        Args:
            text: 응답문.

        Returns:
            걸린 규칙이 있으면 SafetyResult, 없으면 None.
        """
        if m := self.ISOLATION.search(text):
            return SafetyResult(
                verdict=Verdict.REWRITE,
                categories=["guardian_isolation"],
                replacement="type: 1\n" + SAFE_FALLBACK,
                matched_text=m.group(0),
            )
        if m := self.UNSAFE_ADVICE.search(text):
            return SafetyResult(
                verdict=Verdict.REWRITE,
                categories=["unsafe_advice"],
                replacement="type: 2\n" + SAFE_FALLBACK,
                matched_text=m.group(0),
            )
        return None


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
        # 개행도 문장 구분자로 본다. 문장부호 없이 줄바꿈만으로 나열한 응답이
        # 한 문장으로 묶여 문장 수 상한을 통과하는 것을 막는다.
        sentences = [s for s in re.split(r"[.!?~\n]\s*", text) if s.strip()]

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
