"""
CLOVA Speech STT 구현체 (단문 인식 REST).

실시간 스트리밍(gRPC)은 P1 후반에 붙이되, 인터페이스는 지금 고정.
keyword boosting 은 아동 발화 + 캐릭터 고유명사에 특히 효과가 커서
초기부터 넣어두는 걸 권장.

이 모듈은 환경변수를 읽기만 하고 .env 를 로드하지 않음.
필요한 키는 CLOVA_SPEECH_INVOKE_URL, CLOVA_SPEECH_SECRET 두 개.
.env.example 참고.
"""

from __future__ import annotations

import os

import httpx

from core.interfaces import STTEngine, STTResult

# 아이가 자주 쓰는 단어 + 서비스 고유명사. 오인식 잡히는 대로 계속 추가할 것.
DEFAULT_BOOST = [
    "놀자",
    "수수께끼",
    "역할놀이",
    "유치원",
    "어린이집",
    "선생님",
    "엄마",
    "아빠",
]


class ClovaSTT(STTEngine):
    name = "clova_speech"

    def __init__(self, invoke_url: str | None = None, secret: str | None = None):
        self.invoke_url = invoke_url or os.environ["CLOVA_SPEECH_INVOKE_URL"]
        self.secret = secret or os.environ["CLOVA_SPEECH_SECRET"]
        self._client = httpx.AsyncClient(timeout=15.0)

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> STTResult:
        resp = await self._client.post(
            f"{self.invoke_url}/recognizer/upload",
            headers={"X-CLOVASPEECH-API-KEY": self.secret},
            files={"media": ("turn.wav", audio, "audio/wav")},
            data={
                "params": (
                    '{"language":"ko-KR","completion":"sync",'
                    f'"boostings":{_boost_json()}}}'
                )
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return STTResult(
            text=(data.get("text") or "").strip(),
            confidence=data.get("confidence"),
            raw=data,
            engine=self.name,
        )


def _boost_json() -> str:
    items = ",".join(f'{{"words":"{w}"}}' for w in DEFAULT_BOOST)
    return f"[{items}]"
