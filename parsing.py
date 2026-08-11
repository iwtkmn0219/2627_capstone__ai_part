"""
PROMPTS.bak 파서.

포맷 (CP949/EUC-KR 인코딩, 탭 구분 2컬럼):
    <wav 경로>\t<음절 단위로 공백이 들어간 문장>
예)
    M5GSH2046-20170801-Korean/wav/M5GSH2046_016\t왕 자 는  별 이  모 두  그 의  것 이 라 고  생 각 했 다

문장 안에서 음절 사이는 공백 1칸, 어절(단어) 사이는 공백 2칸 이상으로
구분되어 있음. 이 스크립트는 원본 줄을 raw_text 로 보존하면서,
음절을 다시 이어 붙여 실제 문장(text)도 함께 복원.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ENCODING = "cp949"  # = EUC-KR. 다른 데이터셋이면 --encoding 으로 바꿔서 재시도.

_WORD_SPLIT_RE = re.compile(r" {2,}")  # 어절 구분: 공백 2칸 이상


@dataclass
class PromptEntry:
    """PROMPTS.bak 한 줄에 대응하는 파싱 결과.

    Attributes:
        utterance_id: 파일명만 뽑아낸 ID. 예: M5GSH2046_016.
        wav_path: 원본 wav 경로 전체.
        text: 음절을 이어붙여 복원한 문장.
        raw_text: 원본 그대로. 음절 사이 공백 보존.
    """

    utterance_id: str
    wav_path: str
    text: str
    raw_text: str


def _rejoin_syllables(word: str) -> str:
    """어절 내부의 '음절 사이 공백'을 제거해 실제 단어로 복원.

    Args:
        word: 음절 사이에 공백이 들어간 어절.

    Returns:
        공백을 제거한 실제 단어.
    """
    return word.replace(" ", "")


def _normalize_text(raw: str) -> str:
    """음절 단위로 공백이 들어간 원본 문장을 실제 문장으로 복원.

    Args:
        raw: 원본 그대로의 문장. 어절은 공백 2칸 이상으로 구분.

    Returns:
        복원된 실제 문장.
    """
    words = _WORD_SPLIT_RE.split(raw.strip())
    return " ".join(_rejoin_syllables(w) for w in words if w)


def parse_line(line: str) -> PromptEntry | None:
    """한 줄을 PromptEntry 로 변환. 빈 줄이면 None.

    Args:
        line: PROMPTS.bak 의 한 줄. '<wav 경로>\t<음절 단위 문장>' 형식.

    Returns:
        파싱된 PromptEntry. 빈 줄이면 None.

    Raises:
        ValueError: 탭으로 구분된 필드가 없는 경우.
    """
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    if "\t" not in line:
        raise ValueError(f"탭으로 구분된 필드가 없습니다: {line!r}")

    wav_path, raw_text = line.split("\t", 1)
    return PromptEntry(
        utterance_id=Path(wav_path).stem,
        wav_path=wav_path,
        text=_normalize_text(raw_text),
        raw_text=raw_text,
    )


def parse_prompts(
    path: str | Path, *, encoding: str = DEFAULT_ENCODING
) -> list[PromptEntry]:
    """PROMPTS.bak 전체를 파싱해 PromptEntry 리스트로 반환.

    Args:
        path: PROMPTS.bak 파일 경로.
        encoding: 파일 인코딩. 기본은 cp949(EUC-KR).

    Returns:
        파싱된 PromptEntry 리스트.

    Raises:
        ValueError: 특정 줄의 파싱에 실패한 경우. 몇 번째 줄인지 포함해 재발생.
    """
    path = Path(path)
    entries: list[PromptEntry] = []
    with path.open("r", encoding=encoding, errors="strict") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                entry = parse_line(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno} 파싱 실패 - {e}") from e
            if entry is not None:
                entries.append(entry)
    return entries


def to_jsonl(entries: list[PromptEntry], out_path: str | Path) -> None:
    """PromptEntry 리스트를 JSONL 파일로 저장.

    Args:
        entries: 저장할 PromptEntry 리스트.
        out_path: 저장할 파일 경로.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def to_csv(entries: list[PromptEntry], out_path: str | Path) -> None:
    """PromptEntry 리스트를 CSV 파일로 저장.

    Args:
        entries: 저장할 PromptEntry 리스트.
        out_path: 저장할 파일 경로.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utterance_id", "wav_path", "text", "raw_text"]
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def main() -> None:
    parser = argparse.ArgumentParser(description="PROMPTS.bak 파서")
    parser.add_argument(
        "input", nargs="?", default="PROMPTS.bak", help="입력 파일 경로"
    )
    parser.add_argument(
        "--encoding", default=DEFAULT_ENCODING, help="입력 파일 인코딩 (기본: cp949)"
    )
    parser.add_argument(
        "--out", help="결과 저장 경로 (.jsonl 또는 .csv). 생략 시 앞 5개만 미리보기"
    )
    args = parser.parse_args()

    entries = parse_prompts(args.input, encoding=args.encoding)
    print(f"{len(entries)}개 문장 파싱 완료 ({args.input})", file=sys.stderr)

    if not args.out:
        for entry in entries[:5]:
            print(f"[{entry.utterance_id}] {entry.text}")
        return

    out_path = Path(args.out)
    if out_path.suffix == ".csv":
        to_csv(entries, out_path)
    else:
        to_jsonl(entries, out_path)
    print(f"저장 완료: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
