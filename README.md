# Ask your files: RAG on free tiers

- **Frontend** (`frontend/`): static site on Vercel. It starts waking the backend the moment the page opens, keeps it awake while the tab is open, and shows a calm "Looking through your files" state if a question arrives while the server is still starting.
- **Backend** (`backend/`): FastAPI in Docker on Render (free). Embeddings run on the server (ONNX, model baked into the image). Claude writes the answers.
- **Database**: Supabase Postgres with pgvector, so nothing is lost when Render restarts.

**What it does:** reads PDF (including scanned pages, via OCR), Word, PowerPoint, Excel, CSV, HTML, text/code and images; hybrid search (vector and keyword, fused with reciprocal rank fusion) followed by cross-encoder reranking; follow-up questions rewritten into standalone searches; streamed answers with numbered citations and highlighted source passages; per-file scoping; a private workspace per access code.

## Deploy (about 20 minutes)

1. **Supabase**: create a project, open the SQL editor, run `supabase/schema.sql`. (Already deployed the first version? Run it again; it only adds the new columns.) Then Connect, copy the **Session pooler** connection string (it ends in `:5432`) and put your database password in it.
2. **GitHub**: push this folder as a repository.
3. **Render**: New, Blueprint, pick the repo (it reads `render.yaml`). Set these environment variables:
   - `ANTHROPIC_API_KEY`
   - `DATABASE_URL` (from step 1)
   - `ACCESS_CODES`: one or more codes separated by commas. Each code opens its own private workspace.
   - `ALLOWED_ORIGINS`: your Vercel URL, for example `https://your-app.vercel.app`
4. **Vercel**: import the repo, set **Root Directory** to `frontend`, framework **Other**. Before deploying, edit `frontend/config.js` with your Render URL.
5. **Keep-alive** (recommended): in GitHub, Settings, Secrets and variables, Actions, Variables, add `RENDER_URL` (your Render URL). The workflow pings every 10 minutes between 06:00 and 21:00 UTC (edit the `cron` line for your timezone) and also touches the database, which stops Supabase pausing after a week idle. If you prefer, point cron-job.org or UptimeRobot at `https://YOUR-SERVICE.onrender.com/health?deep=1` every 10 minutes instead.
6. Open your Vercel URL, enter an access code, add a file, ask a question.

## How reranking and OCR fit in 512 MB

- **Reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`, about 80 MB on disk): not loaded at startup. It loads on the first question that needs it (a few seconds), is dropped after 4 idle minutes, and memory is returned to the system. Each question fetches 20 candidates and the reranker picks the best 8.
- **OCR** (Tesseract): only pages with almost no text layer are OCR'd, so mixed PDFs stay fast. Tesseract runs as a short-lived process, so it holds no memory between pages.
- **One heavy job at a time:** OCR evicts the reranker before each page. A question that arrives during OCR skips reranking (plain hybrid order) instead of waiting.
- **Background ingestion:** uploads return at once. The file shows "Reading scanned page 3 of 12…" and becomes searchable when done. The page polls while it works and uploads one file at a time. A restart marks unfinished files as failed so they can be uploaded again.

Switches (Render environment variables): `RERANK=off`, `OCR=off`, `OCR_LANG=eng+fra` (add the language package in the Dockerfile), `OCR_DPI` (lower is faster, default 170), `MAX_OCR_PAGES` (default 30), `RERANK_IDLE_SECONDS`.

## Limits to know about

- **OCR is slow on the free CPU.** Render free gives a fraction of a core. A page takes about 1.5 seconds on a normal laptop CPU and can take many times that here. Expect minutes for a long scan, and the 30-page cap keeps that bounded. Large scanned books are better OCR'd elsewhere first.
- **Memory is my estimate, not a measurement.** If Render reports out-of-memory restarts, set `RERANK=off` first.
- If the server restarts or sleeps during a scan, the file is marked failed and needs re-uploading. The pings and the open tab prevent most of this.
- Render free allows 750 instance hours a month **for the whole workspace**, shared by every free web service. A service kept awake 24/7 uses about 744 of them, so other free services would be suspended until the month ends. That's why the keep-alive only runs in daytime hours. Outside that window the first visitor may see a cold start, which the frontend covers.
- Scheduled pings can be delayed, so an occasional cold start can still happen. The frontend covers that by waking the server early and waiting gracefully, but a worst-case wait is still about a minute.
- Scanned PDFs have no text layer and are rejected (OCR isn't included). Photos of text work, because images are read by Claude.
- Uploads are limited to 15 MB and 25 files per workspace (change `MAX_UPLOAD_MB` and `MAX_DOCS_PER_WORKSPACE`). The Supabase free database holds about 500 MB.
- Access codes are shared secrets. They are fine for a personal tool or small team. For public sign-ups, swap in Supabase Auth.
- The embedding model is English-focused. For other languages choose a multilingual 384-dimension model through `EMBED_MODEL` and bake it into the Dockerfile.
- Claude usage is billed to your API key. The per-code rate limits (20 questions a minute, 15 uploads an hour) are there to cap abuse.
