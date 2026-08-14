"""
Gemini 구현체.

[중요] 모델 문자열은 절대 하드코딩하지 않음.
Gemini 모델은 1년 이내에 종료되기도 함. (2.5-flash-lite 는 신규 사용자 차단됨)
GEMINI_MODEL 환경변수로 갈아끼울 수 있게 유지할 것.

이 모듈은 환경변수를 읽기만 하고 .env 를 로드하지 않음.
라이브러리가 import 만으로 파일을 읽는 부작용을 갖지 않도록 로드는 진입점의 책임으로 둠.
키 목록은 .env.example 참고.

핵심 포인트:
1) static_system 을 별도로 받아 캐싱이 걸리도록 고정 배치.
   페르소나+안전지침이 매 턴 동일해야 캐시가 먹는다.
2) safety_settings 를 아동 서비스 기준으로 보수적으로 설정.
   벤더 필터는 1차 방어선일 뿐, 우리 SafetyChecker 가 2차.
3) Gemini 3.x 부터 temperature/top_p/top_k 는 deprecated 이므로 보내지 않음.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

from core.interfaces import LLMEngine, LLMResult

# 2026-08 기준 사용 가능한 저비용 모델
#   gemini-3.1-flash-lite  $0.125/$0.750  (종료 예정 2027-05-07)
#   gemini-3.5-flash-lite  $0.30 /$2.50   (종료일 미발표)
# 캡스톤 종료가 2027-06 이므로, 출시 전 반드시 재검증할 것.
#
# get(키, 기본값) 이 아니라 or 로 받는다. 환경변수가 빈 문자열로 설정되어 있으면
# 키가 '존재'하므로 기본값이 발동하지 않고, 빈 모델명이 그대로 API 로 넘어간다.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"

_CHILD_SAFETY = [
    types.SafetySetting(category=c, threshold="BLOCK_LOW_AND_ABOVE")
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class GeminiEngine(LLMEngine):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODEL
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    async def generate(
        self,
        *,
        static_system: str,
        dynamic_context: str,
        user_utterance: str,
        max_output_tokens: int = 200,
        thinking: bool = False,
    ) -> LLMResult:
        kwargs = dict(
            system_instruction=static_system,
            max_output_tokens=max_output_tokens,
            safety_settings=_CHILD_SAFETY,
        )
        # 3.x 는 temperature 계열 미지원. 구형 모델을 쓸 때만 추가.
        if not self.model.startswith("gemini-3"):
            kwargs["temperature"] = 0.8

        # 음성 턴 지연을 줄이기 위해 사고 예산 0. 특수 턴만 켠다.
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=-1 if thinking else 0
            )
        except Exception:
            pass

        config = types.GenerateContentConfig(**kwargs)
        prompt = f"{dynamic_context}\n\n아이의 말: {user_utterance}"

        t0 = time.perf_counter()
        resp = await self._client.aio.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        elapsed = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage_metadata", None)
        text = (resp.text or "").strip()

        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=self.model,
            blocked_by_vendor=not text,
            raw={
                "latency_ms": elapsed,
                "cached_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
                "finish_reason": str(
                    resp.candidates[0].finish_reason if resp.candidates else ""
                ),
            },
        )
