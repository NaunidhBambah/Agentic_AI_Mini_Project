"""Backup web search used when Gemini's own Google Search grounding is unavailable (quota used up).

Order: Tavily (if TAVILY_API_KEY is set; free tier ~1000 searches/month, made for AI apps)
       -> DuckDuckGo via the `ddgs` package (no key; can be rate-limited)
       -> nothing (the caller then falls back to Gemini's general knowledge).
Each result: {"title", "url", "text"}. Gemini then writes the answer FROM these results, with [W1]-style
citations, so the student still sees real web sources.
"""
import json
import urllib.request

from . import config


def _tavily(query: str, n: int) -> list:
    if not config.TAVILY_API_KEY:
        return []
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps({"query": query, "max_results": n, "search_depth": "basic"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.TAVILY_API_KEY}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "text": x.get("content", "")}
            for x in data.get("results", [])]


def _ddg(query: str, n: int) -> list:
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    return [{"title": x.get("title", ""), "url": x.get("href", ""), "text": x.get("body", "")}
            for x in DDGS().text(query, max_results=n)]


def search(query: str, n: int = 5) -> tuple:
    """Returns (results, provider name). Never raises."""
    for name, fn in (("Tavily", _tavily), ("DuckDuckGo", _ddg)):
        try:
            res = [r for r in fn(query, n) if r["url"] and r["text"]]
            if res:
                print(f"[WEB] {name}: {len(res)} results for '{query[:60]}'", flush=True)
                return res[:n], name
        except Exception as exc:          # network error, rate limit, bad key ...
            print(f"[WEB] {name} failed: {str(exc)[:120]}", flush=True)
    return [], ""