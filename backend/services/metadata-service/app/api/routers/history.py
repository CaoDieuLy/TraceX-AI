from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from ...database import get_session
from shared.models import QueryHistory, User

router = APIRouter(tags=["history"])


@router.get("")
def get_history(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict:
    rows = session.scalars(
        select(QueryHistory)
        .where(QueryHistory.user_id == current_user.id)
        .order_by(QueryHistory.created_at.desc())
        .limit(20)
    ).all()
    items = [
        {
            "query_id": r.query_id,
            "query_text": r.query_text,
            "video_id": r.video_id,
            "storage_path": None,
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(),
        }
        for r in rows
    ]
    return {"count": len(items), "items": items}


class SelectRequest(BaseModel):
    query: str = ""
    selectedIndex: int = 0


@router.post("/select", status_code=200)
def select_history(
    body: SelectRequest,
    _: User = Depends(get_current_user),
) -> dict:
    return {"ok": True}
