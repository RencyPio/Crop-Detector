# Hosting on Hugging Face Spaces (Docker)

Checked 23 Nov 2026. Free "CPU basic" Spaces give 2 vCPU, 16 GB RAM and non-persistent disk, and sleep
after 48 hours without visitors (the first visit then takes a minute or two to wake). Hugging Face's docs
are inconsistent about whether Docker Spaces need a PRO subscription on personal accounts: if creating
the Space shows a payment error (HTTP 402), stop and see "If Docker Spaces need PRO" below.

Verified before writing this: all 152 Python packages resolve for Linux / Python 3.12 with the CPU-only
torch build (no NVIDIA libraries), Git LFS is installed on the laptop, and `prepare_space.ps1` was tested.

## Steps

1. Sign in at https://huggingface.co and open https://huggingface.co/new-space
   - Space name: `crop-doctor`
   - SDK: **Docker**, template **Blank**
   - Hardware: **CPU basic (free)**
   - Visibility: **Private**
2. In the new Space, open **Settings -> Variables and secrets -> New secret** and add:
   - `GROQ_API_KEY`
   - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (optional)
   - `OPENWEATHER_API_KEY` (optional; Open-Meteo is used without it)
3. Create an access token at https://huggingface.co/settings/tokens with **Write** access.
   Git asks for it as the password when you push (user name = your Hugging Face user name).
4. In PowerShell:

   ```powershell
   cd C:\Users\rency\Downloads\PDS
   git clone https://huggingface.co/spaces/<your-user>/crop-doctor
   cd cropdetector
   .\deployment\prepare_space.ps1 -SpaceDir C:\Users\rency\Downloads\PDS\crop-doctor
   cd ..\crop-doctor
   git lfs install
   git add .
   git commit -m "Deploy Crop Doctor"
   git push
   ```

   The script copies the tracked project files and the model, puts the Dockerfile at the top level,
   adds the Space settings header to README.md, and sends the model and images through Git LFS.
   It never copies `.env`.
5. Open the Space's **Logs** tab. The first build takes about 15 to 25 minutes (packages, then the
   knowledge base download and ChromaDB build). When the log shows `API ready`, open the **App** tab.

## Updating later

Run steps 4's `prepare_space.ps1`, then `git add . ; git commit -m "Update" ; git push` in the Space folder.

## Notes

- The detection log (`data/detections.db`) is wiped when the Space restarts: fine for a demo, list it as a limitation.
- Telegram alerts go to your own `TELEGRAM_CHAT_ID`.
- Keep the Space private: a public link lets anyone use your Groq quota.
- Photo upload inside the Space needs Streamlit's XSRF protection off; `deployment/start.sh` already sets this.

## If Docker Spaces need PRO

- Run the same container on any Docker host with at least 4 GB RAM (for example Google Cloud Run, which needs a billing card on file).
- Or demo on the laptop with `.\run.ps1`.
