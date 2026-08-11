# 커밋 컨벤션 & 코드 스타일

이 프로젝트는 [gitmoji](https://gitmoji.dev/) 컨벤션을 사용한다.

## 메시지 구조

- header, body, footer 세 가지로 구성하며, 각 부분을 빈 행으로 구분한다.

- body, footer는 선택사항이며, 다음과 같은 경우 사용한다.
    - body
        - 권장 사례: 부연 설명이 필요하거나, 커밋을 한 이유가 필요한 경우
        - 내용: 무엇을 변경했는지, 왜 변경했는지를 중심으로 작성한다.(어떻게 변경했는지는 지양한다.)
    - footer
        - 사용 사례: issue tracker id를 작성할 때 사용한다.
        - 형식: "유형: # 이슈 번호"
        - 유형
            1. Fixes : 이슈 수정중 (아직 해결되지 않은 경우)
            2. Resolves : 이슈를 해결했을 때 사용
            3. Ref : 참고할 이슈가 있을 때 사용
            4. Related to : 해당 커밋에 관련된 이슈번호 (아직 해결되지 않은 경우)
        - 예시: "Fixes: #45 Related to: #34, #23"


```
<gitmoji>: <subject>

body

footer
```

- Co-Authored-By 트레일러는 붙이지 않는다.

## 규칙

1. 제목과 본문을 빈 행으로 구분한다.
2. 제목은 50글자 이내로 제한한다.
3. 제목의 첫 글자는 대문자로 작성한다.
4. 제목 끝에는 마침표를 넣지 않는다.
5. 제목은 명령문으로 사용하며 과거형을 사용하지 않는다.
6. 본문의 각 행은 72글자 내로 제한한다.
7. 어떻게 보다는 무엇과 왜를 설명한다

## 자주 쓰는 gitmoji

| gitmoji | 코드 | 의미 |
|---|---|---|
| 🎉 | `:tada:` | 초기 커밋 |
| ✨ | `:sparkles:` | 새 기능 |
| 🐛 | `:bug:` | 버그 수정 |
| ♻️ | `:recycle:` | 리팩토링 |
| 📝 | `:memo:` | 문서 추가/수정 |
| 🔧 | `:wrench:` | 설정 변경 |
| ✅ | `:white_check_mark:` | 테스트 추가/수정 |
| 🔥 | `:fire:` | 코드/파일 삭제 |
| 🚚 | `:truck:` | 파일/디렉토리 이동, 이름 변경 |
| ⚡️ | `:zap:` | 성능 개선 |
| 🔒️ | `:lock:` | 보안 관련 수정 |

## 예시
### 1. Header만 사용하는 경우
```
🎉 Initial commit
✨ CLOVA STT 어댑터 추가
🐛 세이프티 체커의 이스케이프 처리 오류 수정
📝 README에 로드맵 섹션 추가
```
### 2. Header-Body-Footer를 모두 사용하는 경우

```
🐛: Safari에서 모달을 띄웠을 때 스크롤 이슈 수정

모바일 사파리에서 Carousel 모달을 띄웠을 때,
모달 밖의 상하 스크롤이 움직이는 이슈 수정.

resolves: #1137
```

# 코드 스타일

## 포맷터

- Python 코드 포맷팅은 [black](https://black.readthedocs.io/)을 사용한다.
- 커밋 전 `black .`으로 포맷을 맞춘다.

## 주석 / Docstring

- Python 주석(docstring)은 [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)을 따른다.
- 모듈, 클래스, public 함수/메서드에는 docstring을 작성한다.
- 함수 docstring은 `Args`, `Returns`, `Raises` 섹션으로 구성한다.

### 예시

```python
def check_severity(text: str, threshold: float = 0.5) -> bool:
    """입력 텍스트의 위험도를 판정한다.

    Args:
        text: 판정할 원본 텍스트.
        threshold: 위험으로 판단할 최소 점수. 기본값 0.5.

    Returns:
        위험도가 threshold 이상이면 True, 아니면 False.

    Raises:
        ValueError: text가 빈 문자열인 경우.
    """
```
