# Crop Disease Advisory System

MSc Data Science project. A farmer uploads a leaf photo; a MobileNetV2 model diagnoses it,
RAG retrieves grounded treatment advice, an agent layer makes the advice context-aware
(crop, confidence, confusable diseases, weather), and automation delivers alerts and logs
every detection. Full spec: `Project_Spec_From_Scratch.md`.

## Setup (Windows, PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The trained model is **not** in Git. Put these two files in `models/`:
`crop_disease_cnn_ft_final.keras` and `class_indices.json`.

## Start the app (one command)

```powershell
.\run.ps1
```

Or double-click `run.bat`. It starts the API and the web app, uses another port if 8000 is busy,
and opens the app in your browser. Close the window to stop everything. (`.\run.ps1 -NoBrowser`
starts it without opening a tab.)

## Developer commands

```powershell
python -m pytest                                   # unit tests
python -m scripts.sanity_check data/sanity_check   # local accuracy on known PlantVillage images
python -m scripts.predict_folder data/ood_test_images reports/ood_predictions.csv
```

Always run from the project root, with `python -m`, so `src.` imports resolve.

### RAG knowledge base (Phase 2)

```powershell
python -m src.rag.ingest                         # download sources in data/knowledge_base/sources.csv
python -m src.rag.vector_store                   # chunk + embed + store in ChromaDB
python -m src.rag.retrieve Tomato___Late_blight  # see what retrieval returns
python -m src.rag.advisor --image path\to\leaf.jpg   # image -> CNN -> RAG advice
python -m scripts.rag_all_diseases               # advice + grounding check for every disease (~26 Groq calls)
```

Groq's free tier allows about 8,000 tokens a minute and 200,000 a day (one answer uses ~3,000).
When the limit is hit, the app shows the cited source passages instead of a written answer.

`sources.csv` (committed) lists every document; the downloaded texts in `raw/` and the
`chroma_db/` index are gitignored, so we never redistribute the full texts. If a site blocks
the downloader, open the page in a browser, save its text as `raw/<doc_id>.txt`, and re-run.
Without `GROQ_API_KEY` in `.env` the advisor runs in offline "extractive" mode (top passages, cited).

### Agent + full pipeline (Phase 3)

```powershell
python -m src.pipeline path\to\leaf.jpg --crop Tomato --lat 13.08 --lon 80.27   # image -> decision + advice + weather
python -m src.agent.weather_check 13.08 80.27       # rain probability, next 6 h
python -m scripts.agent_scenarios                   # agent vs bare CNN on real images -> reports/agent_scenarios.csv
```

Weather uses OpenWeatherMap if `OPENWEATHER_API_KEY` is set, otherwise Open-Meteo (free, no key).
Thresholds live in `src/agent/config.py`.

### Alerts + detection log (Phase 4)

```powershell
python -m src.pipeline path\to\leaf.jpg --crop Tomato --lat 13.08 --lon 80.27 --channels telegram
python -m src.automation.logger                     # last 20 detections from data/detections.db
```

Fill `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Optional SMS: add the Twilio fields listed in `src/automation/senders.py` and use `--channels telegram,sms`.
A channel without keys reports `not_configured`; every run is logged with its alert status.

### API + web app (Phase 5)

```powershell
uvicorn src.api.main:app --port 8000      # terminal 1: API, docs at http://localhost:8000/docs
streamlit run frontend/app.py             # terminal 2: web app at http://localhost:8501
```

Docker (one container, API inside, Streamlit on port 7860):
`docker build -f deployment/Dockerfile -t cropdetector .` then `docker run -p 7860:7860 --env-file .env cropdetector`.
Hosting steps: `deployment/HOSTING.md`.

## Layout

```
src/dl/          config, labels, preprocessing, model, train (Kaggle), predict (local)
src/rag/         Phase 2: ingest, chunk, embed, ChromaDB, retrieve, Groq generate
src/agent/       Phase 3: crop mask, confidence gate, confusable check, weather check
src/automation/  Phase 4: Telegram (+ optional Twilio SMS), SQLite logging
src/api/         Phase 5: FastAPI          frontend/  Phase 5: Streamlit
scripts/         runnable checks           tests/     pytest
models/          trained model (gitignored) + reports
data/            images, knowledge base (mostly gitignored)
```

## Build phases

| Phase | Work | Status |
|---|---|---|
| 1 | Clean repo, DL layer as importable modules, verify local inference | Done |
| 2 | RAG: 45 sources for 25 of 26 disease classes, chunk, embed, ChromaDB, Groq | Built |
| 3 | Agent: crop mask, confidence gate, confusable pairs, healthy shortcut, weather | Built |
| 4 | Automation: Telegram alerts (+ optional SMS), SQLite detection log | Built; Telegram tested live |
| 5 | FastAPI `/predict`, Streamlit UI, Docker, free hosting | Built and tested locally; hosting pending |
| 6 | Evaluation (RAG review, agent scenarios, optional PlantDoc), report | |
