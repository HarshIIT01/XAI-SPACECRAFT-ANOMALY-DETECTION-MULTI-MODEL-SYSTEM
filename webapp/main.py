from __future__ import annotations

import sys
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from spacecraft_anomaly.config import Config
from spacecraft_anomaly.paths import resolve_path


@dataclass
class RunRecord:
    run_id: str
    created_at_utc: str
    status: str  # queued|running|succeeded|failed
    command: List[str]
    out_dir: str
    model: str
    epochs: int
    dataset: str
    error: Optional[str] = None
    return_code: Optional[int] = None
    finished_at_utc: Optional[str] = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runs_dir() -> Path:
    p = resolve_path("results/runs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_json_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def _run_log_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.log"


def _load_run(run_id: str) -> RunRecord:
    path = _run_json_path(run_id)
    if not path.exists():
        raise FileNotFoundError(run_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(**data)


def _save_run(rec: RunRecord) -> None:
    _run_json_path(rec.run_id).write_text(
        json.dumps(asdict(rec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_runs(limit: int = 50) -> List[RunRecord]:
    items: List[RunRecord] = []
    for p in sorted(_runs_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            items.append(RunRecord(**json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items


def calculate_dashboard_stats() -> Dict[str, Any]:
    all_runs = list_runs(limit=1000)

    total = len(all_runs)
    succeeded = sum(1 for r in all_runs if r.status == "succeeded")
    failed = sum(1 for r in all_runs if r.status == "failed")
    running = sum(1 for r in all_runs if r.status == "running")
    queued = sum(1 for r in all_runs if r.status == "queued")

    success_rate = int(succeeded / total * 100) if total > 0 else 0
    failure_rate = int(failed / total * 100) if total > 0 else 0

    model_stats = {}
    dataset_stats = {}
    for r in all_runs:
        model_stats[r.model] = model_stats.get(r.model, 0) + 1
        dataset_stats[r.dataset] = dataset_stats.get(r.dataset, 0) + 1

    return {
        "total_runs": total,
        "succeeded": succeeded,
        "failed": failed,
        "running": running,
        "queued": queued,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "unique_models": len(model_stats),
        "unique_datasets": len(dataset_stats),
        "model_list": ", ".join(sorted(model_stats.keys())[:3]),
    }


_lock = threading.Lock()
_threads: Dict[str, threading.Thread] = {}


def start_pipeline_run(model: str, epochs: int, dataset: str = "SMAP") -> RunRecord:
    run_id = uuid.uuid4().hex[:12]
    out_dir = str(resolve_path(f"results/runs/{run_id}"))
    os.makedirs(out_dir, exist_ok=True)

    script = resolve_path("scripts/run_pipeline.py")
    cmd = ["python", str(script), "--model", model, "--epochs", str(int(epochs)), "--out", out_dir]

    rec = RunRecord(
        run_id=run_id,
        created_at_utc=_now_utc_iso(),
        status="queued",
        command=cmd,
        out_dir=out_dir,
        model=model,
        epochs=int(epochs),
        dataset=dataset,
    )
    _save_run(rec)

    def _runner() -> None:
        nonlocal rec
        rec.status = "running"
        _save_run(rec)

        log_path = _run_log_path(run_id)
        try:
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(resolve_path(".")),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                rc = proc.wait()
            rec.return_code = int(rc)
            rec.finished_at_utc = _now_utc_iso()
            if rc == 0:
                rec.status = "succeeded"
            else:
                rec.status = "failed"
                rec.error = f"Pipeline exited with code {rc}. See log."
        except Exception as e:
            rec.return_code = -1
            rec.finished_at_utc = _now_utc_iso()
            rec.status = "failed"
            rec.error = f"{type(e).__name__}: {e}"
        finally:
            _save_run(rec)

    t = threading.Thread(target=_runner, daemon=True, name=f"pipeline-run-{run_id}")
    with _lock:
        _threads[run_id] = t
    t.start()
    return rec


def _safe_artifact_paths(run: RunRecord) -> List[Path]:
    out = Path(run.out_dir)
    if not out.exists():
        return []
    allowed = (".png", ".jpg", ".jpeg", ".txt", ".npy", ".json", ".csv", ".log")
    paths: List[Path] = []
    for p in sorted(out.glob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            paths.append(p)
    # also include the captured log if present
    lp = _run_log_path(run.run_id)
    if lp.exists():
        paths.append(lp)
    return paths


cfg = Config()
templates = Jinja2Templates(directory=str(resolve_path("webapp/templates")))

app = FastAPI(title="Spacecraft Anomaly Detection", version="1.0")
app.mount("/static", StaticFiles(directory=str(resolve_path("webapp/static"))), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    all_runs = list_runs(limit=1000)
    stats = calculate_dashboard_stats()

    model_stats = {}
    dataset_stats = {}
    for r in all_runs:
        model_stats[r.model] = model_stats.get(r.model, 0) + 1
        dataset_stats[r.dataset] = dataset_stats.get(r.dataset, 0) + 1

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "cfg": cfg,
            "stats": stats,
            "recent_runs": list_runs(limit=8),
            "model_stats": model_stats,
            "dataset_stats": dataset_stats,
        },
    )


@app.get("/run", response_class=HTMLResponse)
def run_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context={
            "cfg": cfg,
            "defaults": {
                "model": "GNN",
                "epochs": 5,
                "dataset": "SMAP",
            },
        },
    )


@app.post("/run")
def run_submit(
    model: str = Form(...),
    epochs: int = Form(5),
    dataset: str = Form("SMAP"),
) -> RedirectResponse:
    if model not in {"LSTM_AE", "LSTM_VAE", "TRANSFORMER", "GNN", "FUSION"}:
        raise HTTPException(status_code=400, detail="Invalid model.")
    if epochs < 1 or epochs > 500:
        raise HTTPException(status_code=400, detail="Invalid epochs.")
    rec = start_pipeline_run(model=model, epochs=int(epochs), dataset=dataset)
    return RedirectResponse(url=f"/results/{rec.run_id}", status_code=303)


@app.get("/results", response_class=HTMLResponse)
def results_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={"runs": list_runs(limit=50)},
    )


@app.get("/results/{run_id}", response_class=HTMLResponse)
def results_detail(request: Request, run_id: str) -> HTMLResponse:
    try:
        run = _load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found.")
    artifacts = _safe_artifact_paths(run)
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "run": run,
            "artifacts": [
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "is_image": p.suffix.lower() in {".png", ".jpg", ".jpeg"},
                }
                for p in artifacts
            ],
        },
    )


@app.get("/results/{run_id}/download/{filename}")
def download_artifact(run_id: str, filename: str) -> FileResponse:
    try:
        run = _load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found.")

    base = Path(run.out_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not target.exists() or not target.is_file():
        # allow download of captured log which lives in runs dir
        log_p = _run_log_path(run_id).resolve()
        if filename == log_p.name and log_p.exists():
            return FileResponse(str(log_p), filename=log_p.name)
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(str(target), filename=target.name)


# -----------------------
# API endpoints
# -----------------------


@app.post("/api/runs")
def api_create_run(payload: Dict[str, Any]) -> JSONResponse:
    model = str(payload.get("model", "GNN"))
    epochs = int(payload.get("epochs", 5))
    dataset = str(payload.get("dataset", "SMAP"))
    rec = start_pipeline_run(model=model, epochs=epochs, dataset=dataset)
    return JSONResponse({"run_id": rec.run_id, "status": rec.status})


@app.get("/api/runs")
def api_list_runs(limit: int = 50) -> JSONResponse:
    runs = list_runs(limit=limit)
    return JSONResponse({"runs": [asdict(r) for r in runs]})


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str) -> JSONResponse:
    try:
        run = _load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found.")
    return JSONResponse(asdict(run))

