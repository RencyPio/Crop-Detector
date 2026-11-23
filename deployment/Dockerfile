FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PORT=7860 \
    API_URL=http://127.0.0.1:8000

RUN useradd -m -u 1000 app
WORKDIR /home/app/cropdetector

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN test -f models/crop_disease_cnn_ft_final.keras \
    || (echo "models/crop_disease_cnn_ft_final.keras missing from the build context" && exit 1)

USER app

RUN python -m src.rag.ingest && python -m src.rag.vector_store

EXPOSE 7860
CMD ["bash", "deployment/start.sh"]
