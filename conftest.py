"""pytest 루트 설정.

내용은 비어 있지만 이 파일이 있어야 한다. pytest 가 이 파일을 찾으면서
저장소 루트를 sys.path 에 넣어주고, 그래야 tests/ 안에서 core, safety 같은
최상위 모듈을 import 할 수 있다. 지우면 테스트가 ImportError 로 죽는다.
"""
