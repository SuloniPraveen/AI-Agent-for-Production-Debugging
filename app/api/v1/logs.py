"""Log upload and batch status (Phase 1 ingestion API)."""

import asyncio
import os
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
)
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.log_batch import LogBatchStatus
from app.models.user import User
from app.schemas.logs import (
    LogBatchStatusResponse,
    LogLinesUploadBody,
    LogUploadResponse,
)
from app.services.log_ingestion import (
    create_pending_batch,
    get_batch_for_user,
    run_ingestion_sync,
)

router = APIRouter()


async def _ingest_task(batch_id: UUID, path: str) -> None:
    try:
        await asyncio.to_thread(run_ingestion_sync, batch_id, path)
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/upload", response_model=LogUploadResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["logs_upload"][0])
async def upload_logs(
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    """Accept UTF-8 logs via multipart (`file` field) or JSON `{"lines": [...], "filename": "..."}`.

    Ingestion runs in a background task; poll GET /logs/batches/{batch_id}.
    """
    tmp_path: str | None = None
    filename = "upload.log"

    try:
        ct = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in ct:
            form = await request.form()
            f = form.get("file")
            if f is None or not isinstance(f, UploadFile):
                raise HTTPException(status_code=422, detail="multipart uploads require form field `file`")
            if not f.filename:
                raise HTTPException(status_code=422, detail="file must have a filename")
            suffix = Path(f.filename).suffix or ".log"
            filename = f.filename[:512]
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="log_ingest_")
            os.close(fd)
            Path(tmp_path).write_bytes(await f.read())
        elif "application/json" in ct:
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(status_code=422, detail="invalid JSON body") from e
            try:
                parsed = LogLinesUploadBody.model_validate(body)
            except ValidationError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            filename = parsed.filename
            fd, tmp_path = tempfile.mkstemp(suffix=".log", prefix="log_ingest_")
            os.close(fd)
            Path(tmp_path).write_text("\n".join(parsed.lines), encoding="utf-8")
        else:
            raise HTTPException(
                status_code=415,
                detail="Use multipart/form-data with `file` or application/json with `lines`",
            )

        batch = create_pending_batch(user_id=user.id, filename=filename)
        assert tmp_path is not None
        background_tasks.add_task(_ingest_task, batch.id, tmp_path)
        logger.info("log_upload_accepted", batch_id=str(batch.id), user_id=user.id, filename=filename)
        return LogUploadResponse(batch_id=batch.id, status=LogBatchStatus.PENDING.value, message="Ingestion started")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("log_upload_failed", error=str(e))
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/batches/{batch_id}", response_model=LogBatchStatusResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["logs_batch_status"][0])
async def get_batch_status(
    request: Request,
    batch_id: UUID,
    user: User = Depends(get_current_user),
):
    """Return ingestion status for a batch owned by the current user."""
    batch = get_batch_for_user(batch_id, user.id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return LogBatchStatusResponse(
        batch_id=batch.id,
        status=batch.status,
        filename=batch.filename,
        line_count=batch.line_count,
        chunk_count=batch.chunk_count,
        error_message=batch.error_message,
    )
