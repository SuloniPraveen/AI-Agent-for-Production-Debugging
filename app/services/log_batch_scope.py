"""Resolve which log batch(es) vector search should use for a user."""

from __future__ import annotations

from typing import List
from uuid import UUID

from sqlmodel import Session, select

from app.core.logging import logger
from app.models.log_batch import LogBatch, LogBatchStatus
from app.services.database import database_service


def resolve_log_search_batch_ids(user_id: int, focus_batch_id: UUID | None = None) -> List[UUID]:
    """Return batch UUIDs to search.

    - If ``focus_batch_id`` is set: that batch only when it exists, belongs to the user, and is completed.
    - Otherwise: the user's most recently created **completed** batch (typical \"last upload\").

    Returns an empty list if there is nothing to search.
    """
    engine = database_service.engine
    with Session(engine) as session:
        if focus_batch_id is not None:
            b = session.get(LogBatch, focus_batch_id)
            if b is None or b.user_id != user_id:
                logger.warning(
                    "log_search_focus_batch_denied",
                    user_id=user_id,
                    batch_id=str(focus_batch_id),
                    found=b is not None,
                )
                return []
            if b.status != LogBatchStatus.COMPLETED.value:
                logger.info(
                    "log_search_focus_batch_not_ready",
                    user_id=user_id,
                    batch_id=str(focus_batch_id),
                    status=b.status,
                )
                return []
            return [focus_batch_id]

        stmt = (
            select(LogBatch)
            .where(LogBatch.user_id == user_id)
            .where(LogBatch.status == LogBatchStatus.COMPLETED.value)
            .order_by(LogBatch.created_at.desc())
        )
        latest = session.exec(stmt).first()
        return [latest.id] if latest else []
