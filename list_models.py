"""
내 API 키로 실제 사용 가능한 모델 목록 확인.
404 가 날 때 가장 먼저 돌려볼 스크립트.

    python list_models.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("사용 가능한 생성 모델:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"{m.name}")
