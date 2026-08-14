# 대화 파이프라인 (P1 스켈레톤)

STT → 안전(입력) → LLM → 안전(출력) → TTS 턴 파이프라인.

## 구조

```
core/interfaces.py    모든 컴포넌트의 추상 인터페이스 (여기가 설계의 핵심)
core/orchestrator.py  턴 실행 + 지연 계측
core/fakes.py         API 키 없이 도는 가짜 구현체
stt/clova.py          CLOVA Speech
llm/gemini.py         Gemini 2.5 Flash-Lite
safety/checkers.py    3단 안전 계층
context/persona.py    앵쵸 페르소나(static) + 개인화 컨텍스트(dynamic)
telemetry/logger.py   P3 평가 + 학부모 대시보드용 로그
```

## 실행

```bash
pip install -r requirements.txt
python demo.py          # API 키 없이 안전 필터 검증
python -m pytest        # 심각도 라벨링 + 안전 불변식 테스트
```

## 테스트

```
tests/test_severity.py           severity-levels.md 종합 예시 표의 실행 가능한 사본
tests/test_safety_invariants.py  깨지면 실제 피해가 되는 항목만 모은 것
```

`test_safety_invariants.py` 는 일반 회귀 테스트가 아닙니다. L4(보호 필요) 진술이
학부모 대시보드 로그에 남지 않는지, L4 응답이 보호자를 지목해 권하지 않는지,
알림이 fail-closed 인지를 고정합니다. 여기가 빨간불이면 다른 건 보지 마세요.

## 설계 규칙 (지키지 않으면 나중에 비쌉니다)

1. **오케스트레이터는 벤더 SDK를 import 하지 않는다.** 전부 인터페이스 경유.
   → STT/LLM 벤치마크가 "구현체 추가"로 끝납니다.
2. **static_system 은 절대 턴마다 바뀌지 않는다.** 캐시 히트가 깨집니다.
   `telemetry` 의 `cached` 토큰 수로 캐시가 실제로 먹는지 매주 확인하세요.
3. **모든 안전 판정을 로그에 남긴다.** 소급 생성이 불가능한 데이터입니다.
4. **BLOCK 보다 REWRITE/ESCALATE 를 우선한다.** 아이 입장에서 대화가
   갑자기 끊기는 건 그 자체로 나쁜 경험입니다.

## 로드맵 연결

- P1: 이 스켈레톤 + RuleChecker 로 안전하게 가동, 로그 축적
- P2: 축적된 로그를 라벨링해 KoELECTRA 분류기 학습 → ClassifierChecker 투입
- P3: 로그 기반 정량 평가 (안전 위반율, 어휘 적합성, 지연, 비용)
- P4: 스트리밍 STT/TTS, 감정 분석, 학부모 대시보드
