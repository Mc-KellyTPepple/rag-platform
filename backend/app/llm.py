"""One small wrapper so answers can come from Anthropic (default) or Groq.

LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY
LLM_PROVIDER=groq       needs GROQ_API_KEY (free tier available; OpenAI-compatible API)
Images are always read with Anthropic, because that step needs a vision model.
"""
import base64
import logging
import os

log = logging.getLogger("uvicorn.error")

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
GEN_MODEL = os.getenv("GEN_MODEL") or ("openai/gpt-oss-120b" if PROVIDER == "groq" else "claude-sonnet-5-5")
FAST_MODEL = os.getenv("FAST_MODEL") or ("openai/gpt-oss-20b" if PROVIDER == "groq" else "claude-haiku-5-5")
VISION_MODEL = os.getenv("VISION_MODEL", "claude-haiku-5-5")
GROQ_URL = "https://api.groq.com/openai/v1"

_c = {}


def _groq_extra(model):
    # gpt-oss models "think" before answering, and that thinking uses up the token budget.
    # Keep it short so the visible answer isn't cut off.
    return {"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in model else {}


def init():
    if PROVIDER == "groq":
        from openai import AsyncOpenAI, OpenAI
        key = os.environ["GROQ_API_KEY"]
        _c["sync"] = OpenAI(api_key=key, base_url=GROQ_URL)
        _c["async"] = AsyncOpenAI(api_key=key, base_url=GROQ_URL)
    else:
        import anthropic
        _c["sync"] = anthropic.Anthropic()
        _c["async"] = anthropic.AsyncAnthropic()
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        _c["vision"] = _c["sync"] if PROVIDER != "groq" else anthropic.Anthropic()
    log.info("LLM provider: %s, answer model: %s", PROVIDER, GEN_MODEL)


def complete(system, messages, max_tokens=150):
    """Short, non-streaming call (used to rewrite follow-up questions)."""
    if PROVIDER == "groq":
        r = _c["sync"].chat.completions.create(
            model=FAST_MODEL, max_tokens=max(max_tokens, 600), temperature=0, **_groq_extra(FAST_MODEL),
            messages=[{"role": "system", "content": system}] + messages)
        return (r.choices[0].message.content or "").strip()
    r = _c["sync"].messages.create(model=FAST_MODEL, max_tokens=max_tokens, temperature=0, system=system, messages=messages)
    return r.content[0].text.strip()


async def stream(system, messages, max_tokens=1200):
    """Yields the answer text piece by piece."""
    if PROVIDER == "groq":
        s = await _c["async"].chat.completions.create(
            model=GEN_MODEL, max_tokens=max(max_tokens, 2500), stream=True, temperature=0.2, **_groq_extra(GEN_MODEL),
            messages=[{"role": "system", "content": system}] + messages)
        async for ev in s:
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content
    else:
        async with _c["async"].messages.stream(
                model=GEN_MODEL, max_tokens=max_tokens, temperature=0.2, system=system, messages=messages) as s:
            async for t in s.text_stream:
                yield t


def describe_image(data, mime):
    from app import rag
    if "vision" not in _c:
        raise rag.Unsupported("Reading images needs an ANTHROPIC_API_KEY, even when answers come from Groq.")
    r = _c["vision"].messages.create(
        model=VISION_MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime,
                                         "data": base64.b64encode(data).decode()}},
            {"type": "text", "text": "Transcribe all visible text, then describe the image in detail. "
                                     "For charts give the key values and trends; for diagrams the parts and how they connect."},
        ]}])
    return r.content[0].text


def explain(e):
    """Turn a provider error into a message the person can act on."""
    code, text = getattr(e, "status_code", None), str(e).lower()
    if code in (401, 403):
        return "The AI provider rejected the API key. Check it in Render's environment settings."
    if code == 402 or "credit" in text or "billing" in text:
        return "The AI provider says the account is out of credit."
    if code == 404 or "model" in text and "not found" in text:
        return "The AI model name wasn't found. Check the model setting."
    if code == 429:
        return "The AI service is busy or you've hit its rate limit. Try again in a minute."
    return "The AI service didn't respond. Please try again."
