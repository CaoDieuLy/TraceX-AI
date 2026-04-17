from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from .database import Base, engine, get_session
from .schemas import CandidateListResponse, CandidateResponse, ImportResponse, OverviewResponse
from .service import get_candidate, get_overview, import_legacy_metadata, search_candidates

app = FastAPI(title="MCPT Metadata Service", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "metadata-service"}


@app.get("/api/v1/overview", response_model=OverviewResponse)
def overview(session: Session = Depends(get_session)) -> dict:
    return get_overview(session)


@app.get("/api/v1/candidates", response_model=CandidateListResponse)
def list_candidates(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict:
    items = search_candidates(session=session, query=query, limit=limit)
    return {"count": len(items), "items": items}


@app.get("/api/v1/candidates/{candidate_id}", response_model=CandidateResponse)
def candidate_detail(candidate_id: str, session: Session = Depends(get_session)) -> dict:
    candidate = get_candidate(session, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@app.post("/api/v1/candidates/import-legacy", response_model=ImportResponse)
def import_candidates(session: Session = Depends(get_session)) -> dict:
    return import_legacy_metadata(session)
