import asyncio
import hashlib
import hmac
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import llm, rag

CODES = [c.strip() for c in os.getenv("ACCESS_CODES", "").split(",") if c.strip()]
ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_DOCS = int(os.getenv("MAX_DOCS_PER_WORKSPACE", "25"))

SYSTEM = (
    "You answer questions using only the numbered sources provided. "
    "Cite every claim with its source number in square brackets, like [1] or [2][3]. "
    "If the sources don't contain the answer, say so plainly and mention what is missing. "
    "Write cite markers only as bracketed numbers like [1], never in any other format. "
    "State only what the sources say. Copy job titles, names and dates exactly as written. "
    "If a source gives only an organization and dates, don't invent a title or duties for it. "
    "When no title is given, write \"Role not stated\". "
    "If a detail isn't there, say it isn't stated rather than guessing. "
    "Be concise: short paragraphs or bullet points, and a table only if the person asks for one."
)

executor = ThreadPoolExecutor(max_workers=1)  # one file at a time keeps memory flat


@asynccontextmanager
async def lifespan(app):
    rag.init()  # loads the embedding model and opens the DB pool before serving
    rag.reset_stale()
    llm.init()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

_hits = defaultdict(deque)


def limit(ns, bucket, n, per):
    q, now = _hits[(ns, bucket)], time.time()
    while q and now - q[0] > per:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, "Too many requests. Please wait a moment and try again.")
    q.append(now)


def auth(authorization: str = Header("")):
    code = authorization.removeprefix("Bearer ").strip()
    if not any(hmac.compare_digest(code, c) for c in CODES):
        raise HTTPException(401, "Invalid access code.")
    return hashlib.sha256(code.encode()).hexdigest()[:16]  # each code gets its own private workspace


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
def health(deep: bool = False):
    if deep:
        rag.ping_db()  # also keeps the free Supabase project from pausing
    return {"ok": True}


@app.get("/documents")
def documents(ns=Depends(auth)):
    return rag.list_docs(ns)


@app.delete("/documents/{doc_id}")
def delete(doc_id: str, ns=Depends(auth)):
    rag.delete_doc(ns, doc_id)
    return {"ok": True}


@app.post("/documents", status_code=202)
async def upload(file: UploadFile = File(...), ns=Depends(auth)):
    """Accepts the file and returns at once. Reading, OCR and indexing happen in the background."""
    limit(ns, "upload", 15, 3600)
    name = (file.filename or "file")[:200]
    if not rag.is_supported(name):
        raise HTTPException(415, "That file type isn't supported yet.")
    data = await file.read()
    if len(data) > MAX_MB * 1_048_576:
        raise HTTPException(413, f"Files must be smaller than {MAX_MB} MB.")
    if await asyncio.to_thread(rag.count_docs, ns) >= MAX_DOCS:
        raise HTTPException(409, f"Your workspace is full ({MAX_DOCS} files). Delete one to add another.")
    if await asyncio.to_thread(rag.count_processing) >= 3:
        raise HTTPException(429, "Still reading earlier files. Try again in a moment.")
    doc = await asyncio.to_thread(rag.create_doc, ns, name, file.content_type or "")
    executor.submit(rag.process, ns, doc["id"], name, data, llm.describe_image)
    return doc


class ChatIn(BaseModel):
    question: str
    history: list[dict] = []
    doc_ids: list[str] | None = None


def rewrite(question, history):
    """Turn a follow-up like 'and the second one?' into a standalone search query."""
    if not history:
        return question
    try:
        convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
        out = llm.complete(
            "Rewrite the user's last message as one standalone search query, resolving pronouns "
            "from the conversation. Output only the query.",
            [{"role": "user", "content": f"{convo}\nuser: {question}"}], max_tokens=120)
        return out or question
    except Exception as e:
        llm.log.warning("query rewrite failed: %r", e)
        return question


@app.post("/chat")
async def chat(body: ChatIn, ns=Depends(auth)):
    limit(ns, "chat", 20, 60)
    q = body.question.strip()[:2000]
    if not q:
        raise HTTPException(400, "Type a question first.")
    hist = [{"role": m["role"], "content": str(m["content"])[:2000]} for m in body.history[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")]
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    query = await asyncio.to_thread(rewrite, q, hist)
    hits = await asyncio.to_thread(rag.search, ns, query, body.doc_ids)

    async def gen():
        yield sse("sources", [{"n": i + 1, **h} for i, h in enumerate(hits)])
        if not hits:
            yield sse("token", "I couldn't find anything relevant in your files. Try rephrasing, or add the file that covers this.")
            yield sse("done", "")
            return
        ctx = "\n\n".join(
            f"[{i+1}] {h['filename']}" + (f", page {h['page']}" if h["page"] else "")
            + (f", {h['section']}" if h["section"] else "") + f"\n{h['content']}"
            for i, h in enumerate(hits))
        messages = hist + [{"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {q}"}]
        try:
            async for t in llm.stream(SYSTEM, messages):
                yield sse("token", t)
        except Exception as e:
            llm.log.exception("answer generation failed")
            yield sse("error", llm.explain(e))
        yield sse("done", "")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
