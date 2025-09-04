# app.py
from __future__ import annotations

"""
SOP vs Log Comparison API
- Sync:        POST /process
- Async (SSE): POST /process_async  →  GET /stream/{job_id} (SSE) or GET /progress/{job_id}
- Health:      GET /health
- UI (opt):    If ./static/index.html exists → GET /ui and /static/*
"""

import asyncio
import base64
import io
import json
import re
import threading
import traceback
import uuid
import time
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime
from decimal import Decimal
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# === Project utilities you already have ===
from utils import get_rows_between_activities  # noqa: F401
from sop_extractor import get_machine_procedure
from sop_checker import compare_logs_and_sop

app = FastAPI(title="SOP vs Log Comparison API", version="1.2.0")

# (Optional) enable CORS if calling from a separate origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten if you know your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (Optional) serve /ui and /static if a local 'static' folder exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/ui")
    def ui():
        return FileResponse(os.path.join("static", "index.html"))

# =========================
# Markdown table parsing
# =========================

def _find_first_markdown_table_block(md_text: str) -> Optional[List[str]]:
    lines = md_text.splitlines()
    i = 0
    sep_re = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$')

    while i < len(lines) - 1:
        line = lines[i]
        next_line = lines[i + 1]
        if '|' in line and sep_re.match(next_line or ''):
            block = [line.strip(), next_line.strip()]
            j = i + 2
            while j < len(lines) and ('|' in lines[j]) and lines[j].strip():
                block.append(lines[j].strip())
                j += 1
            return block
        i += 1
    return None


def markdown_table_to_dataframe(markdown_text: str) -> Optional[pd.DataFrame]:
    if not markdown_text or not markdown_text.strip():
        return None

    block = _find_first_markdown_table_block(markdown_text)
    if not block or len(block) < 2:
        return None

    header_line = block[0].strip().strip('|')
    headers = [h.strip() for h in header_line.split('|')]

    data_rows: List[List[str]] = []
    for line in block[2:]:
        row = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(row) < len(headers):
            row = row + [''] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[: len(headers)]
        data_rows.append(row)

    if not data_rows:
        return None

    try:
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    except Exception:
        return None

# =========================
# JSON-safe helpers
# =========================

def _json_safe_cell(x: Any) -> Any:
    """Convert scalars to JSON-safe values (fixes TypeError: Object of type time is not JSON serializable)."""
    if x is None or isinstance(x, (str, bool)):
        return x

    # numpy types
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # pandas / datetime / decimal
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.isoformat()
    if isinstance(x, dtime):
        return x.isoformat()  # HH:MM:SS[.ffffff]
    if isinstance(x, Decimal):
        return float(x)

    # int/float
    if isinstance(x, (int, float)):
        return x

    # fallback to string for other objects
    return str(x)


def _df_to_dict_records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Safely convert a DataFrame to list-of-dicts with JSON-safe scalars."""
    if limit is not None:
        df = df.head(limit)
    # pandas >=2.1: DataFrame.map exists; else fall back to applymap
    if hasattr(df, "map"):
        safe_df = df.map(_json_safe_cell)
    else:
        safe_df = df.applymap(_json_safe_cell)
    return safe_df.to_dict(orient='records')


def _df_to_csv_base64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode('utf-8')).decode('utf-8')

# =========================
# Core processing logic
# =========================

def _read_excel_from_bytes(xlsx_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(xlsx_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {exc}")

def _extract_sop_text(pdf_bytes: bytes, machine_name: str, section_identifier: str) -> str:
    try:
        return get_machine_procedure(io.BytesIO(pdf_bytes), machine_name, section_identifier)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract SOP text: {exc}")

def _build_log_text(filtered_df: pd.DataFrame) -> str:
    headers = filtered_df.columns.tolist()
    log_text = "Headers: " + " | ".join(map(str, headers)) + "\nRows:\n"
    log_text += "\n".join(filtered_df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    return log_text

# Weighted step plan (100 total)
STEP_WEIGHTS = [
    ("read_files", 5),
    ("read_excel", 15),
    ("filter_rows", 15),
    ("build_log_text", 5),
    ("extract_sop", 20),
    ("compare", 30),
    ("finalize", 10),
]

def _process_pipeline(
    log_bytes: bytes,
    sop_bytes: bytes,
    machine_name: str,
    start_activity: str,
    end_activity: str,
    section_identifier: str,
    preview_rows: int,
    progress_cb=None,
) -> Dict[str, Any]:
    """Run the pipeline; optionally report progress via callback(name, pct, message)."""
    def bump(step_name: str, pct_accum: int, msg: str):
        if progress_cb:
            progress_cb(step_name, pct_accum, msg)

    pct = 0

    # Step 1: read files
    bump("read_files", pct, "Validating uploads")
    if not log_bytes:
        raise HTTPException(status_code=400, detail="The uploaded log file appears to be empty.")
    if not sop_bytes:
        raise HTTPException(status_code=400, detail="The uploaded SOP file appears to be empty.")
    pct += STEP_WEIGHTS[0][1]
    bump("read_files", pct, "Files validated")

    # Step 2: read excel
    bump("read_excel", pct, "Reading Excel log")
    df = _read_excel_from_bytes(log_bytes)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Could not read any data from the Excel log file.")
    pct += STEP_WEIGHTS[1][1]
    bump("read_excel", pct, f"Excel loaded: {len(df)} rows")

    # Step 3: filter rows
    bump("filter_rows", pct, "Filtering rows between activities")
    try:
        filtered_df = get_rows_between_activities(df, start_activity, end_activity)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to filter rows: {exc}")
    filtered_count = 0 if filtered_df is None else len(filtered_df)
    pct += STEP_WEIGHTS[2][1]
    bump("filter_rows", pct, f"Filtered rows: {filtered_count}")

    if filtered_df is None or filtered_df.empty:
        pct = 100
        bump("finalize", pct, "No matching rows; finishing")
        return {
            "found_rows": filtered_count,
            "filtered_log_preview": [],
            "sop_text": "",
            "comparison": {
                "parsed_table": False,
                "table": None,
                "csv_base64": None,
                "raw_text": "",
            },
            "message": "No rows matched the start/end activity window. Nothing to compare.",
        }

    # Step 4: build log text
    bump("build_log_text", pct, "Structuring log text")
    log_text = _build_log_text(filtered_df)
    pct += STEP_WEIGHTS[3][1]
    bump("build_log_text", pct, "Log text ready")

    # Step 5: extract SOP text
    bump("extract_sop", pct, "Extracting SOP section")
    sop_text = _extract_sop_text(sop_bytes, machine_name, section_identifier)
    if not sop_text or not sop_text.strip():
        raise HTTPException(status_code=404, detail="Could not find the specified SOP section. Please verify your settings.")
    pct += STEP_WEIGHTS[4][1]
    bump("extract_sop", pct, "SOP extracted")

    # Step 6: compare
    bump("compare", pct, "Analyzing compliance")
    try:
        comparison_results = compare_logs_and_sop(sop_text, log_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed during comparison: {exc}")
    pct += STEP_WEIGHTS[5][1]
    bump("compare", pct, "Comparison complete")

    # Step 7: parse + package
    bump("finalize", pct, "Packaging results")
    comparison_df = markdown_table_to_dataframe(comparison_results)

    if comparison_df is not None and not comparison_df.empty:
        parsed_table = True
        table_records = _df_to_dict_records(comparison_df)
        csv_b64 = _df_to_csv_base64(comparison_df)
        raw_text_out = None
    else:
        parsed_table = False
        table_records = None
        csv_b64 = None
        raw_text_out = comparison_results or "(no output)"

    pct = 100
    bump("finalize", pct, "Done")

    return {
        "found_rows": len(filtered_df),
        "filtered_log_preview": _df_to_dict_records(filtered_df, limit=preview_rows),
        "sop_text": sop_text,
        "comparison": {
            "parsed_table": parsed_table,
            "table": table_records,
            "csv_base64": csv_b64,
            "raw_text": raw_text_out,
        },
        "notes": {
            "csv_field": "If 'parsed_table' is true, 'csv_base64' contains a UTF-8 CSV you can save.",
            "preview_limit": preview_rows,
        },
    }

# =========================
# Routes (sync)
# =========================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "SOP vs Log Comparison API is running",
        "usage_sync": "POST /process (multipart/form-data)",
        "usage_async": "POST /process_async then stream /stream/{job_id} or poll /progress/{job_id}",
        "docs": "/docs",
    }

@app.post("/process")
async def process(
    log_file: UploadFile = File(..., description="Excel log file (.xlsx)"),
    sop_file: UploadFile = File(..., description="SOP PDF file (.pdf)"),
    machine_name: str = Form("PAM GLATT"),
    start_activity: str = Form("PAM GLATT started via SCADA. (Status changed from Off to On)"),
    end_activity: str = Form("PAM GLATT stopped via SCADA. (Status changed from On to Off)"),
    section_identifier: str = Form(""),
    preview_rows: int = Form(50, description="How many filtered log rows to include in the JSON preview"),
):
    log_bytes = await log_file.read()
    sop_bytes = await sop_file.read()

    payload = _process_pipeline(
        log_bytes, sop_bytes, machine_name,
        start_activity, end_activity, section_identifier,
        preview_rows,
    )
    return JSONResponse(content=payload)

# =========================
# Async jobs with SSE & polling
# =========================

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 1800  # keep finished jobs for 30 minutes

def _set_job(job_id: str, **updates):
    """Update job dict and purge old finished jobs."""
    with _jobs_lock:
        now = time.time()
        _jobs.setdefault(job_id, {})
        _jobs[job_id].update(updates)
        _jobs[job_id]["_t"] = now

        # GC old finished jobs
        dead = [
            jid for jid, j in list(_jobs.items())
            if j.get("status") in {"done", "error"} and now - j.get("_t", now) > JOB_TTL_SECONDS
        ]
        for jid in dead:
            _jobs.pop(jid, None)

def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None

def _run_job(
    job_id: str,
    log_bytes: bytes,
    sop_bytes: bytes,
    machine_name: str,
    start_activity: str,
    end_activity: str,
    section_identifier: str,
    preview_rows: int
):
    def progress_cb(step_name: str, pct: int, msg: str):
        _set_job(job_id, progress=pct, step=step_name, message=msg, status="running")

    try:
        _set_job(job_id, status="running", progress=0, step="start", message="Started")
        result = _process_pipeline(
            log_bytes, sop_bytes, machine_name,
            start_activity, end_activity, section_identifier,
            preview_rows,
            progress_cb=progress_cb,
        )
        _set_job(job_id, status="done", progress=100, step="finalize", message="Done", result=result)
    except HTTPException as he:
        _set_job(job_id, status="error", progress=100, step="error",
                 message=he.detail, error={"code": he.status_code, "detail": he.detail})
    except Exception as e:
        _set_job(job_id, status="error", progress=100, step="error",
                 message=str(e), error={"code": 500, "detail": str(e), "trace": traceback.format_exc()})

@app.post("/process_async")
async def process_async(
    log_file: UploadFile = File(..., description="Excel log file (.xlsx)"),
    sop_file: UploadFile = File(..., description="SOP PDF file (.pdf)"),
    machine_name: str = Form("PAM GLATT"),
    start_activity: str = Form("PAM GLATT started via SCADA. (Status changed from Off to On)"),
    end_activity: str = Form("PAM GLATT stopped via SCADA. (Status changed from On to Off)"),
    section_identifier: str = Form(""),
    preview_rows: int = Form(50),
):
    log_bytes = await log_file.read()
    sop_bytes = await sop_file.read()

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="queued", progress=0, step="queued", message="Queued")

    _executor.submit(
        _run_job, job_id, log_bytes, sop_bytes, machine_name,
        start_activity, end_activity, section_identifier, preview_rows
    )
    return {"job_id": job_id}

@app.get("/progress/{job_id}")
async def progress(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job

@app.get("/stream/{job_id}")
async def stream(job_id: str):
    async def event_gen():
        # Ask browser to auto-retry if the SSE connection drops
        yield "retry: 2000\n\n"
        yield f"event: hello\ndata: {json.dumps({'job_id': job_id})}\n\n"
        last_progress = -1
        while True:
            job = _get_job(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'error': 'Unknown job_id'})}\n\n"
                return

            payload = {k: job.get(k) for k in ["status", "progress", "step", "message"]}
            # always send heartbeat
            yield f"event: heartbeat\ndata: {json.dumps(payload)}\n\n"

            if job.get("progress", 0) != last_progress:
                last_progress = job.get("progress", 0)
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

            if job.get("status") in {"done", "error"}:
                final_key = "result" if job.get("status") == "done" else "error"
                final_payload = job.get(final_key) or {}
                yield f"event: {final_key}\ndata: {json.dumps(final_payload)}\n\n"
                return

            await asyncio.sleep(1)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Nginx: disable buffering
            "Connection": "keep-alive",
        },
    )

# =========================
# Health
# =========================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )

