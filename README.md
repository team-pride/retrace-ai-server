# retrace-ai-server

건설 현장 사진 기반 얼굴 인식/판정 AI 서버 (Python, FastAPI).
Spring 서버(retrace-server)가 사진을 넘기면 얼굴 인식 → 판정/정규화를 처리해서 결과를 돌려주는 역할.

## 우선순위 (기획 기준)

1. 얼굴 인식 기본 — 셀피로 기준 벡터 등록, 얼굴 검출 + 본인 여부 판정 *(이번 스켈레톤에서 구현)*
2. 사진 판정 — 각도/표정/블러/품질 판정, 재시도 로직
3. 정규화 — 크기/수평/색조 보정
4. 지표/곡선 — 기하 지표 계산, 시점 곡선 구성
5. 판정/해석 — 예측 생성 및 비교 (AI 핵심)

## 기술 스택

- Python 3.10
- FastAPI + Uvicorn
- DeepFace (얼굴 임베딩 추출) + OpenCV

## 로컬에서 실행하기 (venv)

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate      # Windows는 .venv\Scripts\activate

# 2. 패키지 설치 (개발용: pytest 등 포함)
pip install -r requirements-dev.txt

# 3. 환경변수 파일 준비 (기본값 그대로 써도 됨)
cp .env.example .env

# 4. 서버 실행
uvicorn app.main:app --reload
```

실행 후 http://localhost:8000/docs 에서 Swagger로 API를 바로 테스트해볼 수 있어요.

> ⚠️ DeepFace가 처음 얼굴 인식을 수행할 때 모델 가중치(수십 MB)를 자동으로 다운로드합니다.
> 첫 요청이 느릴 수 있는 건 정상입니다.

### 테스트 실행

```bash
pytest
```

## Docker로 실행하기 (승희님 추천 — 로컬 파이썬 세팅 안 해도 됨)

Python 버전이나 패키지 설치 때문에 머리 아플 필요 없이, Docker만 있으면 바로 실행됩니다.

```bash
# 환경변수 파일 준비
cp .env.example .env

# 빌드 + 실행 (최초 빌드는 DeepFace/OpenCV 설치 때문에 몇 분 걸릴 수 있어요)
docker compose up --build
```

- 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 얼굴 벡터 저장 데이터는 `./data` 폴더에 볼륨으로 유지됩니다.

## 주요 API (우선순위 1)

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| POST | `/api/v1/face/register?user_id={id}` | 셀피 여러 장 업로드 → 평균 임베딩을 기준 벡터로 저장 (원본 이미지는 저장하지 않음) |
| POST | `/api/v1/face/verify?user_id={id}` | 대조 사진 업로드 → 기준 벡터와 코사인 거리 비교 → 본인 여부 판정 |

## 프로젝트 구조

```
app/
  main.py              FastAPI 엔트리포인트
  core/config.py        환경설정 (임계값 등)
  api/routes/           API 라우터 (health, face)
  schemas/               요청/응답 스키마
  services/
    face_service.py      얼굴 임베딩 추출/비교 로직 (DeepFace 래핑)
    vector_store.py       기준 벡터 저장소 (현재는 로컬 JSON, 추후 DB로 교체 예정)
tests/                    pytest 테스트
data/                     로컬 벡터 저장 파일 (git에는 커밋 안 됨)
```

## TODO

- [ ] 우선순위 2: 사진 판정 (각도/표정/블러/품질, 재시도 로직)
- [ ] 우선순위 3: 정규화 (크기/수평/색조 보정)
- [ ] 우선순위 4: 지표/곡선 계산
- [ ] 우선순위 5: 예측/판정 모델
- [ ] `FACE_MATCH_THRESHOLD` 실측 데이터로 튜닝
- [ ] 기준 벡터 저장소를 DB 연동으로 교체 (현재는 로컬 JSON)
