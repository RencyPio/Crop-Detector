# Hosting on Hugging Face Spaces (Docker)

Checked 23 Nov 2026 (re-check before the demo). Free "CPU basic" Spaces give 2 vCPU, 16 GB RAM and
non-persistent disk, and sleep after inactivity; the first visit then takes a minute or two to wake.
Hugging Face's docs are inconsistent about whether **Docker** Spaces need a PRO subscription on
personal accounts. Try creating one; if you get a payment error (HTTP 402), see "Fallbacks" below.

## Steps

1. Create a Space at https://huggingface.co/new-space: SDK **Docker**, hardware **CPU basic**.
   Choose **Private** until you have tested it.
2. In the Space **Settings → Variables and secrets**, add secrets (never commit them):
   `GROQ_API_KEY`, `OPENWEATHER_API_KEY` (optional), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
3. Clone the Space repo next to this project and copy the app into it:

   ```powershell
   git clone https://huggingface.co/spaces/<your-user>/<space-name> ..\crop-space
   robocopy . ..\crop-space /E /XD venv .git data\chroma_db data\knowledge_base\raw data\sanity_check data\ood_test_images reports /XF .env *.db Project_Spec_From_Scratch.md
   Copy-Item deployment\Dockerfile ..\crop-space\Dockerfile
   ```

4. Spaces read settings from a YAML header at the top of the Space's `README.md`. Put this at the
   very top of `..\crop-space\README.md`:

   ```yaml
   ---
   title: Crop Disease Advisor
   emoji: 🌱
   colorFrom: green
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```

5. The model (23 MB) is over Hugging Face's 10 MB limit for plain git files, so track it with Git LFS,
   then push:

   ```powershell
   cd ..\crop-space
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes .
   git commit -m "Deploy crop disease advisor"
   git push
   ```

6. Watch the **Logs** tab. The build installs packages, downloads the knowledge-base sources and builds
   ChromaDB (about 10 to 20 minutes the first time). When it says `API ready`, open the Space.

## Notes

- The detection log (`data/detections.db`) is wiped when the Space restarts (non-persistent disk).
  That is fine for a demo; mention it in the report as a limitation.
- Telegram alerts go to `TELEGRAM_CHAT_ID`, which is your chat. A real multi-farmer product would need
  each farmer to register their own chat id.
- A public Space means anyone with the link can use your Groq quota. Keep it private, or share
  the link only for the demo.

## Fallbacks if Docker Spaces need PRO

- Run the same container on any Docker host with at least 4 GB RAM (e.g. Google Cloud Run's free tier,
  which needs a billing card on file).
- Demo locally and expose it temporarily with a tunnel. Note that this is not a permanent URL.
