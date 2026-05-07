from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from ...core.models import User
from ...core.schemas import AuthTokenResponse, UserLoginRequest
from ...core.auth import create_access_token
from ...services.user_service import authenticate_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=AuthTokenResponse)
def login(payload: UserLoginRequest, session: Session = Depends(lambda: None)) -> dict:
    from ...database import SessionLocal
    session = SessionLocal()
    try:
        user = authenticate_user(session, identifier=payload.identifier, password=payload.password)
        if user is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username, email, or password")
        token = create_access_token(user_id=user.id, email=user.email, role=user.role)
        return {"access_token": token, "user": user}
    finally:
        session.close()
