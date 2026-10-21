"""Browser front end for the presentation demo.

The page shows a search bar and the reformulated pipeline's top results with
their scores. Every search runs ``PresentationDemo.execute_demo_query`` from
``jury_demo.py``, so this terminal prints the full backend trace (normalization,
LLM rewrites, filtering, baseline vs reformulated retrieval, rank movement)
exactly as ``python jury_demo.py`` does.

Saved benchmark queries (the suggestions under the search bar, and any query
from ``evaluation/golden_queries.json``) use their stored rewrites and need no
LLM call. Other queries call the LLM live; if that fails (e.g. quota), the
search falls back to the original query and the page shows those results.

Usage::

    python demo_server.py                # http://localhost:8001
    python demo_server.py --port 8002
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jury_demo import DEMO_MENU_OPTIONS, PresentationDemo
from reformulation.validator import validate_query

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_demo", "index.html")
TOP_K = 10

app = FastAPI(title="Query Reformulation Demo")
demo: PresentationDemo
# One search at a time, so the terminal traces of two searches never interleave.
_search_lock = threading.Lock()


class SearchRequest(BaseModel):
    query: str


@app.get("/")
def page() -> FileResponse:
    return FileResponse(PAGE_PATH)


@app.get("/api/suggestions")
def suggestions() -> List[str]:
    return [opt["query"] for opt in DEMO_MENU_OPTIONS]


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    query = request.query.strip()
    try:
        validate_query(query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with _search_lock:
        results = demo.execute_demo_query(query)[:TOP_K]

    top_score = results[0].score if results and results[0].score > 0 else 1.0
    return {
        "query": query,
        "results": [
            {
                "rank": rank,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "section": chunk.section,
                "text": chunk.chunk_text,
                "score": round(chunk.score, 5),
                "relative": round(100 * chunk.score / top_score),
            }
            for rank, chunk in enumerate(results, start=1)
        ],
    }


def main() -> None:
    global demo
    parser = argparse.ArgumentParser(description="Browser front end for the query reformulation demo")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically.")
    args = parser.parse_args()

    # Print each trace line as it happens, even when the terminal is redirected.
    sys.stdout.reconfigure(line_buffering=True)
    demo = PresentationDemo(use_cache=True)
    url = f"http://localhost:{args.port}"
    print(f"Demo page: {url}\nSearches typed in the page print their full trace here.\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    # Quiet server logs, so the terminal shows only the demo trace.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
