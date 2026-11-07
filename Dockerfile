# Catdex API (Hugging Face Space, Docker SDK, port 8080): FastAPI around the student's own pipeline (catdex_step1-4). The web app in web/ is on Vercel.
FROM python:3.12-slim

WORKDIR /app

# OpenCV (pulled in by ultralytics) needs these shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY catdex_step1.py catdex_step2.py catdex_step3.py catdex_step4.py coat_labels.csv ./
COPY api/ api/

# Bake the model weights into the image (YOLO11s, CLIP ViT-B/16, the pet re-ID model; about 1 GB), so a cold
# start loads them from disk instead of downloading them. Importing the steps downloads everything once.
ENV HF_HOME=/app/.hf YOLO_CONFIG_DIR=/app/.ultralytics PYTHONUNBUFFERED=1
RUN python -c "import catdex_step1, catdex_step2, catdex_step3, catdex_step4"

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser
ENV HF_HUB_OFFLINE=1

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health', timeout=5)" || exit 1

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
