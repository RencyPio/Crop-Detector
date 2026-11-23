# Free hosting on Streamlit Community Cloud

Checked 23 Nov 2026: Community Cloud is free, deploys straight from a GitHub repo (public or private),
gives each app up to 2.7 GB of memory, and sleeps after 12 hours without visitors (a visitor wakes it).

The hosted app runs as one process: the Streamlit page calls the pipeline directly (no FastAPI), with
the 9.5 MB TFLite model and ONNX embeddings, so it needs no TensorFlow or PyTorch. On first start it
downloads the model from your Hugging Face model repo and builds the knowledge base (about 2 minutes).

## 1. Put the model on Hugging Face (free model repo)

1. Open https://huggingface.co/new. Owner: you. Name: `crop-doctor-model`. Visibility: **Private**.
2. In the new repo: **Files and versions -> Add file -> Upload files**, drag in
   `models\crop_disease_cnn.tflite`, then **Commit changes**.
3. Create a **Read** token at https://huggingface.co/settings/tokens and copy it.

## 2. Make sure GitHub has the latest code

```powershell
git push origin main
```

## 3. Deploy the app

1. Open https://share.streamlit.io and sign in with GitHub (allow access to your repo).
2. **Create app -> Deploy a public app from GitHub** (the app link can be kept private in settings later).
3. Repository: `RencyPio/Crop-Detector`, branch: `main`, main file path: `frontend/app.py`.
4. App URL: pick a name, for example `crop-doctor`.
5. **Advanced settings**: Python version **3.12**, and paste into **Secrets**:

   ```toml
   MODEL_REPO = "<your-hf-username>/crop-doctor-model"
   HF_TOKEN = "hf_..."
   GROQ_API_KEY = "gsk_..."
   TELEGRAM_BOT_TOKEN = "..."
   TELEGRAM_CHAT_ID = "..."
   OPENWEATHER_API_KEY = "..."
   ```

   The Telegram and OpenWeather lines are optional.
6. **Deploy**. Installing packages takes a few minutes. The first page load then shows
   "Getting ready..." while it downloads the model and builds the knowledge base.

## Notes

- Secrets are never in GitHub; Streamlit exposes them to the app as environment variables.
- The History tab resets when the app restarts or sleeps (the disk is not permanent).
- Under **Settings -> Sharing** you can restrict who can open the app.
- Local use is unchanged: `.\run.ps1` still runs the FastAPI + Streamlit version.
