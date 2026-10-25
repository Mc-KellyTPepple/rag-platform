# Ask your files: RAG on free tiers

- **Frontend** (`frontend/`): static site on Vercel. It starts waking the backend the moment the page opens (and again before a request if it has been idle), and shows a calm "Looking through your files" state while the server is still starting. Nothing pings the backend in the background, so it only wakes when someone uses it.
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
5. Open your Vercel URL, enter an access code, add a file, ask a question.

## How reranking and OCR fit in 512 MB

- **Reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`, about 80 MB on disk): not loaded at startup. It loads on the first question that needs it (a few seconds), is dropped after 4 idle minutes, and memory is returned to the system. Each question fetches 20 candidates and the reranker picks the best 8.
- **OCR** (Tesseract): only pages with almost no text layer are OCR'd, so mixed PDFs stay fast. Tesseract runs as a short-lived process, so it holds no memory between pages.
- **One heavy job at a time:** OCR evicts the reranker before each page. A question that arrives during OCR skips reranking (plain hybrid order) instead of waiting.
- **Background ingestion:** uploads return at once. The file shows "Reading scanned page 3 of 12…" and becomes searchable when done. The page polls while it works and uploads one file at a time. A restart marks unfinished files as failed so they can be uploaded again.

Switches (Render environment variables): `RERANK=off`, `OCR=off`, `OCR_LANG=eng+fra` (add the language package in the Dockerfile), `OCR_DPI` (lower is faster, default 170), `MAX_OCR_PAGES` (default 30), `RERANK_IDLE_SECONDS`.

## Using Groq (free) instead of Anthropic for answers

Set these Render environment variables: `LLM_PROVIDER=groq` and `GROQ_API_KEY` (from console.groq.com). Defaults are `openai/gpt-oss-120b` for answers and `openai/gpt-oss-20b` for rewriting follow-ups; override with `GEN_MODEL` and `FAST_MODEL`. Groq changes which models free accounts can use, and its Llama models are now listed as enterprise-only, so check console.groq.com/docs/models if you see "model not found". Groq's free tier has daily token limits, so a busy day can hit them. Reading **images** still needs `ANTHROPIC_API_KEY`. This changes who writes the answer only. It does not reduce the server's memory use. To do that, set `RERANK=off`.

## Demo mode (no access code)

Set `ALLOW_GUESTS=on` in Render. The sign-in page then offers **Try the demo**, which gives each browser its own private workspace (a random id kept in that browser). Guests are limited so a public demo can't use up the free tiers:

| Variable | Default | What it limits |
|---|---|---|
| `GUEST_MAX_DOCS` | 3 | files per demo workspace |
| `GUEST_MAX_MB` | 5 | size of each file |
| `GUEST_MAX_OCR_PAGES` | 5 | scanned pages read per file |
| `GUEST_RETENTION_HOURS` | 24 | demo workspaces older than this are deleted (checked hourly while the server is awake and at every start) |
| `GUEST_TOTAL_DOCS` | 60 | demo files across all visitors (protects the 500 MB database) |
| `GUEST_DAILY_CHATS` | 300 | demo questions per day across all visitors (protects the shared AI quota; resets on restart) |

Guests also get 6 questions a minute and 6 uploads an hour. Access-code workspaces keep their own, higher limits. Set `ALLOW_GUESTS=off` to close the demo at any time.

Visitors' files and questions are sent to the AI provider you configured (Groq or Anthropic). Tell visitors not to upload sensitive documents, as the demo screen does.

## Limits to know about

- **OCR is slow on the free CPU.** Render free gives a fraction of a core. A page takes about 1.5 seconds on a normal laptop CPU and can take many times that here. Expect minutes for a long scan, and the 30-page cap keeps that bounded. Large scanned books are better OCR'd elsewhere first.
- **Memory is my estimate, not a measurement.** If Render reports out-of-memory restarts, set `RERANK=off` first.
- If the server restarts or sleeps during a scan, the file is marked failed and needs re-uploading. Keep the page open until the file finishes, because the page's progress checks keep the server awake while it works.
- Render free allows 750 instance hours a month **for the whole workspace**, shared by every free web service. The backend wakes only when someone uses it and sleeps again after about 15 idle minutes, so hours are used only while it is in use.
- The first visit after a quiet spell takes up to a minute to wake the server. The frontend starts the wake-up as soon as the page opens and waits gracefully.
- Supabase's free plan **pauses a project after a week with no activity**. With no background pings, if nobody uses the app for a week, open the Supabase dashboard and click **Restore project** (it takes a few minutes).
- Scanned PDFs have no text layer and are rejected (OCR isn't included). Photos of text work, because images are read by Claude.
- Uploads are limited to 15 MB and 25 files per workspace (change `MAX_UPLOAD_MB` and `MAX_DOCS_PER_WORKSPACE`). The Supabase free database holds about 500 MB.
- Access codes are shared secrets. They are fine for a personal tool or small team. For public sign-ups, swap in Supabase Auth.
- The embedding model is English-focused. For other languages choose a multilingual 384-dimension model through `EMBED_MODEL` and bake it into the Dockerfile.
- Claude usage is billed to your API key. The per-code rate limits (20 questions a minute, 15 uploads an hour) are there to cap abuse.
