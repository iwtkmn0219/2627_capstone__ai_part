"""심각도 라벨링 회귀 테스트.

capstone_documents/심각도.md 의 종합 예시 표를 그대로 옮긴 것이다. 문서가 사양이고
이 파일이 그 사양의 실행 가능한 사본이다. 둘 중 하나만 고치면 안 된다.

P2 에서 KoELECTRA 분류기를 앙상블로 넣을 때 이 표가 기준선이 된다.
"""

from __future__ import annotations

import asyncio

import pytest

from core.interfaces import RiskLevel
from safety.checkers import RuleChecker


def label(text: str):
    """아이 발화 하나를 입력 단계로 검사한다.

    Args:
        text: 아이 발화.

    Returns:
        판정 결과를 담은 SafetyResult.
    """
    return asyncio.run(RuleChecker().check(text, context={"stage": "input"}))


# (발화, 기대 레벨, 기대 ongoing, 기대 alleged_target)
SPEC_CASES = [
    ("어제 넘어져서 다쳤어", RiskLevel.L3, False, None),
    ("지금 여기 피나", RiskLevel.L3, True, None),
    ("무서웠어", RiskLevel.L3, False, None),
    ("지금 무서워", RiskLevel.L3, True, None),
    ("아빠가 때렸어", RiskLevel.L4, True, "아빠"),
    ("지금 아빠가 때려", RiskLevel.L4, True, "아빠"),
    ("선생님이 밥 안 줬어", RiskLevel.L4, False, "선생님"),
    ("친구가 때렸어", RiskLevel.L3, False, "친구"),
    ("맞았어", RiskLevel.L4, True, None),
    ("혼자 있어", RiskLevel.L3, True, None),
    ("죽고 싶어", RiskLevel.L4, True, None),
    ("아무도 없어", RiskLevel.L2, False, None),
    ("공룡 좋아!", RiskLevel.L0, False, None),
]


@pytest.mark.parametrize(
    "text,level,ongoing,target", SPEC_CASES, ids=[c[0] for c in SPEC_CASES]
)
def test_severity_table(text, level, ongoing, target):
    """심각도.md 종합 예시 표의 각 행이 그대로 재현되는지 확인."""
    r = label(text)
    assert (r.level, r.ongoing, r.alleged_target) == (level, ongoing, target)


def test_known_false_positive_game_talk():
    """알려진 오탐을 현재 동작 그대로 고정한다.

    심각도.md 는 "게임에서 맞았어" 를 L0 으로 적어두고 "여기서 걸리면 오탐" 이라고
    명시했지만, 규칙은 '맞았' 을 잡아 L4 로 올린다. 제외 규칙(게임|만화)을 넣으면
    "아빠가 게임하다가 때렸어" 가 통째로 빠지므로 '애매하면 높은 쪽' 원칙에 따라
    오탐을 남겨뒀다. 흩어지는 오탐은 누적 판정과 P2 분류기가 흡수할 몫이다.

    이 테스트가 실패하면 오탐이 고쳐진 것이다. 심각도.md 와 함께 갱신할 것.
    """
    assert label("게임에서 맞았어").level is RiskLevel.L4


@pytest.mark.parametrize(
    "text,level",
    [
        ("아빠가 때렸어", RiskLevel.L4),  # 양육 어른 -> 보호 필요
        ("선생님이 때렸어", RiskLevel.L4),  # 부모가 아니어도 양육 어른이면 L4
        ("친구가 때렸어", RiskLevel.L3),  # 또래 -> 등록 어른 무관
        ("동생이 때렸어", RiskLevel.L3),
        ("맞았어", RiskLevel.L4),  # 지목 없음 -> 애매하면 높은 쪽
    ],
)
def test_violence_level_depends_on_target(text, level):
    """폭력의 레벨은 지목 대상이 누구냐로 갈린다."""
    assert label(text).level is level


def test_target_is_not_normalized():
    """지목 호칭을 원문 그대로 싣는다.

    '선생님'을 담임으로 좁히거나 '아버지'를 '아빠'로 통일하면 라우터가 우리 판단을
    검증할 수 없게 된다. 해소는 등록 어른 테이블을 가진 쪽의 몫이다.
    """
    assert label("아버지가 때렸어").alleged_target == "아버지"
    assert label("원장님이 때렸어").alleged_target == "원장님"


def test_ongoing_tie_break_prefers_true():
    """시제가 애매하면 진행 중으로 본다.

    반대로 틀리면 지금 위험한 아이를 위로만 하고 끝내게 된다.
    """
    assert label("피가 나").ongoing is True  # 시제 표지 없음
    assert label("아까 다쳤어").ongoing is False  # 과거 부사
    assert label("죽고 싶어").ongoing is True  # 자해는 판정하지 않고 고정


def test_contracted_past_tense_is_detected():
    """'줬어', '때렸어' 처럼 ㅆ 받침으로 축약된 과거형을 놓치지 않는다.

    문자열로 '었/았' 만 찾으면 절반을 과거로 인식하지 못하고, 그러면 지나간 일에
    어른 호출 응답이 나간다.
    """
    assert label("선생님이 밥 안 줬어").ongoing is False
    assert label("친구가 때렸어").ongoing is False
