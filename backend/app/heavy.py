"""Memory-hungry helpers for a 512 MB free instance.

Nothing here is loaded at startup:
- The reranker loads on the first question that needs it and is dropped after a few idle minutes.
- OCR runs Tesseract as a short-lived subprocess, so it holds no memory between pages.
- Only one heavy job runs at a time. OCR evicts the reranker first, and a question that arrives
  while OCR is busy skips reranking instead of waiting.
"""
import ctypes
import gc
import os
import threading
import time

RERANK_ON = os.getenv("RERANK", "on").lower() != "off"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
IDLE_SECONDS = int(os.getenv("RERANK_IDLE_SECONDS", "240"))
OCR_ON = os.getenv("OCR", "on").lower() != "off"
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "170"))

lock = threading.Lock()
_reranker = None
_last_used = 0.0


def _release():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand freed memory back to the OS
    except Exception:
        pass


def _evict():  # caller must hold `lock`
    global _reranker
    if _reranker is not None:
        _reranker = None
        _release()


def rerank(query, hits, top_n):
    """Re-score candidates with a cross-encoder. Falls back to the fused order if unavailable or busy."""
    global _reranker, _last_used
    if not RERANK_ON or len(hits) <= 1:
        return hits[:top_n]
    if not lock.acquire(timeout=1):
        return hits[:top_n]
    try:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        scores = [float(s) for s in _reranker.rerank(query, [h["content"] for h in hits])]
        _last_used = time.time()
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [{**hits[i], "score": scores[i]} for i in order[:top_n]]
    except Exception:
        _evict()
        return hits[:top_n]
    finally:
        lock.release()


def _janitor():
    while True:
        time.sleep(30)
        with lock:
            if _reranker is not None and time.time() - _last_used > IDLE_SECONDS:
                _evict()


threading.Thread(target=_janitor, daemon=True).start()


def ocr_pdf_pages(data, page_numbers, on_page=None):
    """OCR the given 1-based pages of a PDF. Returns {page_number: text}."""
    import pypdfium2 as pdfium
    import pytesseract

    pdf, out = pdfium.PdfDocument(data), {}
    try:
        for k, n in enumerate(page_numbers):
            if on_page:
                on_page(k, len(page_numbers))
            with lock:
                _evict()
                page = pdf[n - 1]
                img = page.render(scale=OCR_DPI / 72).to_pil().convert("L")
                try:
                    out[n] = pytesseract.image_to_string(img, lang=OCR_LANG, config="--psm 3", timeout=150).strip()
                except RuntimeError:  # Tesseract timed out on this page
                    out[n] = ""
                finally:
                    img.close()
                    page.close()
    finally:
        pdf.close()
    return out
