"""Minimal FastAPI dashboard for browsing Regime Lens artifacts."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Template


UTC = timezone.utc

RUNS_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Regime Lens Runs</title>
    <style>
      body {
        margin: 0;
        padding: 32px 20px 40px;
        background: linear-gradient(135deg, #faf6ea, #efe6d7);
        color: #1f2937;
        font-family: Georgia, "Times New Roman", serif;
      }
      main { max-width: 1080px; margin: 0 auto; }
      .panel {
        background: rgba(255, 250, 242, 0.92);
        border: 1px solid #d6c7ad;
        border-radius: 18px;
        padding: 18px 20px;
      }
      .run { padding: 12px 0; border-bottom: 1px solid #e0d4be; }
      .run:last-child { border-bottom: 0; }
      a { color: #1d4ed8; text-decoration: none; }
      a:hover { text-decoration: underline; }
      code { font-family: Consolas, monospace; }
      .muted { color: #6b7280; }
    </style>
  </head>
  <body>
    <main>
      <h1>Regime Lens Dashboard</h1>
      <p class="muted">Artifact root: <code>{{ artifact_root }}</code></p>
      <section class="panel">
        {% if runs %}
          {% for run in runs %}
          <div class="run">
            <div><a href="/runs/{{ run.pathId }}"><code>{{ run.runId }}</code></a></div>
            <div class="muted">{{ run.agentType or "unknown-agent" }} | {{ run.status }} | episode {{ run.currentEpisode or 0 }}</div>
            <div class="muted"><code>{{ run.runPath }}</code></div>
          </div>
          {% endfor %}
        {% else %}
          <p class="muted">No runs found.</p>
        {% endif %}
      </section>
    </main>
  </body>
</html>"""
)

RUN_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ run.runId }}</title>
    <style>
      body {
        margin: 0;
        padding: 28px 20px 40px;
        background: linear-gradient(180deg, #f8f2e8, #ebe4d5);
        color: #18212f;
        font-family: "Segoe UI", sans-serif;
      }
      main { max-width: 1080px; margin: 0 auto; }
      .panel {
        background: rgba(255, 252, 246, 0.94);
        border: 1px solid #d6c7ad;
        border-radius: 18px;
        padding: 16px 18px;
        margin-top: 18px;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
      }
      a { color: #1d4ed8; text-decoration: none; }
      a:hover { text-decoration: underline; }
      pre, code {
        font-family: Consolas, monospace;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .muted { color: #6b7280; }
    </style>
  </head>
  <body>
    <main>
      <p><a href="/runs">Back to runs</a></p>
      <h1><code>{{ run.runId }}</code></h1>
      <p class="muted">{{ run.agentType or "unknown-agent" }} | {{ run.status }}</p>
      <p class="muted"><code>{{ run.runPath }}</code></p>
      <section class="grid">
        <div class="panel">
          <div class="muted">Current Episode</div>
          <div>{{ run.currentEpisode or 0 }}</div>
        </div>
        <div class="panel">
          <div class="muted">Latest Checkpoint</div>
          <div><code>{{ run.latestCheckpointId or "n/a" }}</code></div>
        </div>
        <div class="panel">
          <div class="muted">Elapsed Seconds</div>
          <div>{{ "%.2f"|format(run.elapsedSeconds or 0.0) }}</div>
        </div>
      </section>
      <section class="panel">
        <h2>Live Snapshot</h2>
        <pre id="live">{{ live_payload }}</pre>
      </section>
      <section class="panel">
        <h2>Checkpoints</h2>
        {% if checkpoints %}
          {% for item in checkpoints %}
          <div style="padding: 10px 0; border-bottom: 1px solid #d8cbb6;">
            <div><a href="/checkpoints/{{ run.pathId }}/{{ item.checkpointId }}"><code>{{ item.checkpointId }}</code></a></div>
            <div class="muted">episode {{ item.episode }} | return {{ item.agentReturn }}</div>
          </div>
          {% endfor %}
        {% else %}
          <p class="muted">No checkpoints yet.</p>
        {% endif %}
      </section>
    </main>
    <script>
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const liveEl = document.getElementById("live");
      const socket = new WebSocket(`${protocol}://${location.host}/live/{{ run.pathId }}`);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          liveEl.textContent = JSON.stringify(payload, null, 2);
        } catch (error) {
          liveEl.textContent = event.data;
        }
      };
    </script>
  </body>
</html>"""
)

CHECKPOINT_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ checkpoint_id }}</title>
    <style>
      body {
        margin: 0;
        padding: 28px 20px 40px;
        background: #f7f7fb;
        color: #18212f;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }
      main { max-width: 960px; margin: 0 auto; }
      .panel {
        background: white;
        border: 1px solid #d8e1f0;
        border-radius: 18px;
        padding: 18px 20px;
      }
      pre, code {
        font-family: Consolas, monospace;
        white-space: pre-wrap;
        word-break: break-word;
      }
      a { color: #1d4ed8; text-decoration: none; }
      a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <main>
      <p><a href="/runs/{{ run_locator }}">Back to run</a></p>
      <div class="panel">
        <h1><code>{{ checkpoint_id }}</code></h1>
        <pre>{{ payload }}</pre>
      </div>
    </main>
  </body>
</html>"""
)


def create_app(artifact_root: Path | str | None = None) -> FastAPI:
    root = Path(artifact_root or (Path(__file__).resolve().parents[1] / "artifacts")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Regime Lens Dashboard")
    app.state.artifact_root = root

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/runs", status_code=307)

    @app.get("/runs", response_model=None)
    async def runs(request: Request):
        items = _list_runs(root)
        if _wants_html(request):
            return HTMLResponse(RUNS_TEMPLATE.render(artifact_root=str(root), runs=items))
        return JSONResponse(items)

    @app.get("/runs/{run_locator}", response_model=None)
    async def run_detail(run_locator: str, request: Request):
        run_dir = _resolve_run_dir(root, run_locator)
        if run_dir is None:
            return JSONResponse({"error": f"Unknown run {run_locator}"}, status_code=404)
        summary = _run_summary(root, run_dir) or {"runId": run_locator}
        checkpoints = _safe_read_json(run_dir / "checkpoints" / "index.json") or {}
        payload = _live_payload(root, run_locator)
        if _wants_html(request):
            return HTMLResponse(
                RUN_TEMPLATE.render(
                    run=summary,
                    checkpoints=checkpoints.get("checkpoints", []),
                    live_payload=json.dumps(payload, indent=2, ensure_ascii=True),
                )
            )
        return JSONResponse({"summary": summary, "checkpoints": checkpoints.get("checkpoints", []), "live": payload})

    @app.get("/checkpoints/{run_locator}/{checkpoint_id}", response_model=None)
    async def checkpoint_detail(run_locator: str, checkpoint_id: str, request: Request):
        run_dir = _resolve_run_dir(root, run_locator)
        if run_dir is None:
            return JSONResponse({"error": f"Unknown run {run_locator}"}, status_code=404)
        checkpoint_root = run_dir / "checkpoints" / checkpoint_id
        payload = {
            "summary": _safe_read_json(checkpoint_root / "summary.json"),
            "episode": _safe_read_json(checkpoint_root / "episode.json"),
            "policy": _safe_read_json(checkpoint_root / "policy.json"),
            "embedding": _safe_read_json(checkpoint_root / "embedding.json"),
            "regime_analysis": _safe_read_json(checkpoint_root / "regime_analysis.json"),
            "expert_analysis": _safe_read_json(checkpoint_root / "expert_analysis.json"),
            "stats": _safe_read_json(checkpoint_root / "stats.json"),
            "resume_state": _safe_read_json(checkpoint_root / "resume_state.json"),
            "data_fit": _safe_read_json(checkpoint_root / "data_fit.json"),
            "explainability": _safe_read_json(checkpoint_root / "explainability.json"),
            "repro": _safe_read_json(checkpoint_root / "repro.json"),
        }
        if _wants_html(request):
            return HTMLResponse(
                CHECKPOINT_TEMPLATE.render(
                    run_locator=run_locator,
                    checkpoint_id=checkpoint_id,
                    payload=json.dumps(payload, indent=2, ensure_ascii=True),
                )
            )
        return JSONResponse(payload)

    @app.websocket("/live/{run_locator}")
    async def live(run_locator: str, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(_live_payload(root, run_locator))
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def run_server(artifact_root: Path | str | None = None, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(create_app(artifact_root), host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Regime Lens artifact dashboard.")
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    run_server(args.artifact_root, host=args.host, port=args.port)
    return 0


def _list_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in _run_dirs(root):
        summary = _run_summary(root, path)
        if not isinstance(summary, dict):
            continue
        runs.append(summary)
    return sorted(
        runs,
        key=lambda item: (str(item.get("updatedAt") or ""), str(item.get("runPath") or "")),
        reverse=True,
    )


def _run_dirs(root: Path) -> list[Path]:
    run_dirs: dict[str, Path] = {}
    for summary_path in root.rglob("summary.json"):
        run_dir = summary_path.parent
        if run_dir.name.startswith("run-"):
            run_dirs[str(run_dir.resolve())] = run_dir.resolve()
    return sorted(run_dirs.values(), key=lambda path: str(path), reverse=True)


def _run_summary(root: Path, run_dir: Path) -> dict[str, Any] | None:
    summary = _safe_read_json(run_dir / "summary.json")
    if not isinstance(summary, dict):
        return None
    relative_path = _relative_run_path(root, run_dir)
    summary["runPath"] = relative_path
    summary["pathId"] = _encode_path_id(relative_path)
    return summary


def _resolve_run_dir(root: Path, locator: str) -> Path | None:
    direct = root / locator
    if direct.exists():
        resolved = direct.resolve()
        if _is_relative_to(resolved, root.resolve()):
            return resolved
    decoded = _decode_path_id(locator)
    if decoded:
        candidate = (root / decoded).resolve()
        if candidate.exists() and _is_relative_to(candidate, root.resolve()):
            return candidate
    matches: list[Path] = []
    for path in _run_dirs(root):
        summary = _safe_read_json(path / "summary.json") or {}
        if path.name == locator or summary.get("runId") == locator:
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    return None


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _live_payload(root: Path, locator: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(root, locator)
    if run_dir is None:
        return {"error": f"Unknown run {locator}"}
    summary = _run_summary(root, run_dir) or {}
    metrics = _safe_read_json(run_dir / "metrics.json") or {}
    checkpoint_index = _safe_read_json(run_dir / "checkpoints" / "index.json") or {}
    latest_checkpoint = summary.get("latestCheckpointId")
    checkpoint_summary = (
        _safe_read_json(run_dir / "checkpoints" / latest_checkpoint / "summary.json")
        if latest_checkpoint
        else None
    )
    series = metrics.get("series", [])
    return {
        "observedAt": datetime.now(tz=UTC).isoformat(),
        "runId": summary.get("runId") or run_dir.name,
        "pathId": summary.get("pathId"),
        "runPath": summary.get("runPath"),
        "summary": summary,
        "latestMetrics": series[-1] if series else None,
        "latestCheckpoint": checkpoint_summary,
        "checkpointCount": len(checkpoint_index.get("checkpoints", [])),
    }


def _relative_run_path(root: Path, run_dir: Path) -> str:
    return run_dir.resolve().relative_to(root.resolve()).as_posix()


def _encode_path_id(relative_path: str) -> str:
    encoded = base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _decode_path_id(path_id: str) -> str | None:
    try:
        padding = "=" * (-len(path_id) % 4)
        return base64.urlsafe_b64decode(f"{path_id}{padding}").decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
