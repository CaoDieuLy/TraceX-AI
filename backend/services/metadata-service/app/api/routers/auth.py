from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from shared.models import User
from ...core.schemas import AuthTokenResponse, UserLoginRequest
from ...core.auth import create_access_token
from ...services.user_service import authenticate_user

router = APIRouter(tags=["auth"])


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


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": getattr(current_user, "full_name", None),
        "role": current_user.role,
        "is_active": getattr(current_user, "is_active", True),
    }
