FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    API_URL=http://127.0.0.1:8000

RUN useradd -m -u 1000 app
WORKDIR /home/app/cropdetector

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN test -f models/crop_disease_cnn.tflite \
    || (echo "models/crop_disease_cnn.tflite missing from the build context" && exit 1)

USER app

RUN python -m src.rag.ingest && python -m src.rag.vector_store

EXPOSE 7860
CMD ["bash", "deployment/start.sh"]
