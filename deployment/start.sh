#!/usr/bin/env bash
set -e

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "waiting for the API (loading the model)..."
for i in $(seq 1 90); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
    echo "API ready"; break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then echo "API process exited"; exit 1; fi
  sleep 2
done

exec streamlit run frontend/app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false --server.enableCORS false
