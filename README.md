# retrace-ai-server

RETRACE — 거울이 기억하지 못하는 얼굴의 변화를, 이미 사진첩에 쌓인 과거 사진으로
복원해 보여주는 서비스의 AI 서버 파트 (Python, FastAPI).
사용자 사진에서 본인 얼굴만 골라내고, 비교 가능한 사진만 판정·정규화한 뒤,
기하 지표를 뽑아 시계열 변화 곡선을 만들고, 관리 마커를 기준으로 관리 효과가
관찰되는지까지 판정하는 역할을 한다.

## 우선순위 (기능명세서 기준)

1. 얼굴 인식 기본 — 셀피로 기준 벡터 등록, 얼굴 검출 + 본인 여부 판정 *(구현 완료)*
2. 사진 판정 — 각도/블러/품질 판정, 재시도 로직 *(구현 완료)*
3. 정규화 — 눈동자 간 거리 기준 크기 정렬 + 눈 중심선 수평 정렬 + 흰자 기준 색조 보정 *(구현 완료)*
4. 지표/곡선 — 얼굴 폭·턱선 각도·눈꺼풀 높이·입가 각도 계산 + 시계열 곡선 + 변화점(방향 전환) 자동 탐지 *(구현 완료)*
5. 판정/해석 — 관리 마커 기준 예측선 vs 실제 곡선 비교, 관찰됨/관찰되지 않음/판단 보류 판정 *(구현 완료)*

## 기술 스택

- Python 3.11 (⚠️ 3.12+/3.9- 아님. TensorFlow/DeepFace 휠이 3.11까지만 있음)
- FastAPI + Uvicorn
- DeepFace (얼굴 임베딩 추출, 눈 위치 검출) + OpenCV
- dlib 68점 랜드마크 (턱선/눈꺼풀/입가 지표 계산용, 우선순위 4)
  - DeepFace가 쓰는 retinaface 백엔드는 눈 2개 위치만 주기 때문에 턱선·눈꺼풀·
    입가처럼 더 세밀한 랜드마크가 필요한 우선순위 4는 dlib을 별도로 쓴다.

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

> ⚠️ `pip install` 할 때 dlib이 사전 빌드된 패키지(wheel)가 없어서 **소스에서
> 직접 컴파일**됩니다. 5~15분 정도 걸릴 수 있고, 그동안 터미널이 멈춘 것처럼
> 보여도 정상이니 기다려주세요. macOS에서 컴파일러 관련 에러가 나면
> `xcode-select --install`로 Xcode Command Line Tools를 먼저 설치해야 합니다.

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

## 주요 API

| 우선순위 | Method | Path | 설명 |
|---|---|---|---|
| - | GET | `/health` | 헬스체크 |
| 1 | POST | `/api/v1/face/register?user_id={id}` | 셀피 여러 장 업로드 → 평균 임베딩을 기준 벡터로 저장 (원본 이미지는 저장하지 않음) |
| 1 | POST | `/api/v1/face/verify?user_id={id}` | 대조 사진 업로드 → 기준 벡터와 코사인 거리 비교 → 본인 여부 판정 |
| 2 | POST | `/api/v1/photo/evaluate?photo_key={key}` | 각도/블러/얼굴 검출 신뢰도로 pass/conditional/exclude 판정, 3회 실패 시 자동 최종 제외 |
| 3 | POST | `/api/v1/photo/normalize` | 눈 중심선 수평 정렬 + 눈동자 간 거리 기준 크기 정렬 + 흰자 기준 색보정 → 표준 이미지(base64 PNG) 반환 |
| 4 | POST | `/api/v1/indicator/extract?user_id={id}&photo_key={key}&captured_at={YYYY-MM-DD}` | 사진 한 장에서 얼굴 폭/턱선 각도/눈꺼풀 높이/입가 각도 4종 지표를 계산해 저장 |
| 4 | POST | `/api/v1/indicator/extract-batch?user_id={id}&fallback_captured_at={YYYY-MM-DD}` | 사진 여러 장을 한 번에 업로드 → 각 파일의 EXIF 촬영일을 자동으로 읽어 지표를 일괄 추출/저장. EXIF가 없는 파일은 `fallback_captured_at`을 쓰거나 건너뜀. 곡선 기능 테스트용으로 20장 이상 한 번에 넣을 때 유용 |
| 4 | GET | `/api/v1/indicator/curve?user_id={id}&indicator={지표명}` | 저장된 지표로 시계열 곡선 + 변화점(방향 전환 지점) 계산. 표본 부족 시 `eligible: false`와 부족한 조건 안내 |
| 5 | POST | `/api/v1/marker/register?user_id={id}&marker_date={YYYY-MM-DD}&note={문장}` | 관리 마커(시술/루틴 시작 등) 등록. 종류는 분류하지 않고 자유 문장(note) 그대로 저장 |
| 5 | GET | `/api/v1/marker/list?user_id={id}` | 등록된 관리 마커 목록 조회 |
| 5 | GET | `/api/v1/effect/judge?user_id={id}&indicator={지표명}&marker_id={id}` | 마커 이전 추세를 연장한 예측선과 실제 곡선을 비교해 `observed`/`not_observed`/`pending` 3가지로 변화 관찰 여부 판정 |

## 프로젝트 구조

```
app/
  main.py              FastAPI 엔트리포인트
  core/config.py        환경설정 (임계값 등)
  api/routes/           API 라우터 (health, face, photo, indicator, marker, effect)
  schemas/               요청/응답 스키마
  services/
    face_service.py         얼굴 임베딩 추출/비교 로직 (DeepFace 래핑)
    vector_store.py          기준 벡터 저장소 (현재는 로컬 JSON, 추후 DB로 교체 예정)
    photo_quality_service.py 사진 판정(각도/블러) 로직
    retry_tracker.py         판정 재시도 횟수 추적 (로컬 JSON)
    normalization_service.py 정규화(정렬/크기/색보정) 로직
    indicator_service.py     dlib 68점 랜드마크로 지표 4종 계산 (우선순위 4)
    curve_service.py         지표 시계열 곡선 구성 + 변화점 자동 탐지 (우선순위 4)
    indicator_store.py       사용자별 지표 기록 저장소 (로컬 JSON)
    marker_store.py          사용자별 관리 마커 저장소 (로컬 JSON, 우선순위 5)
    effect_service.py        마커 기준 예측선 vs 실제 곡선 비교 판정 (우선순위 5)
    image_utils.py           이미지 디코딩 공통 유틸 (손상 파일 방어 + EXIF 회전/촬영일 추출)
    model_loader.py          DeepFace 지연 로딩
tests/                    pytest 테스트
data/                     로컬 데이터 파일 (git에는 커밋 안 됨)
```

## TODO

- [x] 우선순위 1: 얼굴 인식 기본
- [x] 우선순위 2: 사진 판정 (각도/블러, 재시도 로직)
- [x] 우선순위 3: 정규화 (크기/수평/색조 보정)
- [x] 우선순위 4: 지표/곡선 계산 (dlib 랜드마크 기반)
- [x] 우선순위 5: 판정/해석 (관리 마커 기준 예측선 vs 실제 곡선 비교)
- [ ] `FACE_MATCH_THRESHOLD` 실측 데이터로 튜닝
- [ ] 기준 벡터/지표/마커 저장소를 DB 연동으로 교체 (현재는 로컬 JSON)
- [ ] 지표 계산 랜드마크 기준점(턱선 4/12번, 입꼬리 48/54번 등)이 실제 사진에서도
      타당한지 실측 데이터로 검증 — 지금은 dlib 68점 표준 인덱스를 기반으로 한
      1차 정의라 팀 리뷰가 필요함
- [ ] 우선순위 5의 "데이터 흔들림 범위"(마커 이전 구간 잔차 표준편차)와
      `EFFECT_NOISE_THRESHOLD_FACTOR`(현재 1.5배) 기준이 실측 데이터로 타당한지
      검증 필요 — 지금은 명세서 문구를 그대로 구현한 1차 정의
- [ ] 기능명세서 4.1(변화 시점 질문 + 자유 문장에서 마커 종류/날짜 자동 추출)과
      4.2(해석 카드 생성)는 자연어 처리/LLM 영역이라 이 AI 서버 범위에서 제외함 —
      지금 `/api/v1/marker/register`는 프론트/사용자가 이미 정한 날짜와 문장을
      그대로 받아 저장만 한다
