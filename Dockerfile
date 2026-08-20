# retrace-ai-server (FastAPI + DeepFace)
FROM python:3.10-slim

# OpenCV / DeepFace 실행에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    AI_DATA_DIR=/app/data

# DeepFace 모델 가중치 캐시 겸 벡터 저장 디렉토리
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--pogit checkout main
rt", "8000"]
