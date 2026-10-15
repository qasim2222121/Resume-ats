# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An **overall ATS score** (0–100) with a category breakdown
- **Found / missing keywords**
- **Prioritized improvements** with specific fixes
- **Before/after bullet rewrites**
- A downloadable report

Built with **Streamlit** + **Google Gemini Flash**.

> Scores are AI estimates to guide your edits. They are not output from a real ATS.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey, then either:

- paste it in the app sidebar, or
- create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional override
```

**Never commit `secrets.toml`** (it is listed in `.gitignore`).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets / env var / sidebar | none |
| `GEMINI_MODEL` | Streamlit secrets / env var / sidebar | `gemini-2.5-flash` |

If Google retires a model name, change it in the sidebar. No code edit is needed.

## Limits

- Max file size 5 MB; password-protected or scanned (image-only) PDFs can't be read.
- Your resume text is sent to Google's Gemini API for analysis.
