from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from ...core.auth import create_access_token, hash_password, verify_password
from ...database import SessionLocal
from ...services.user_service import authenticate_user
from shared.models import User
from ...core.schemas import AuthTokenResponse, UserLoginRequest

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


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=255)


@router.post("/change-password", status_code=200)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    session = SessionLocal()
    try:
        user = session.get(User, current_user.id)
        user.hashed_password = hash_password(payload.new_password)
        session.commit()
    finally:
        session.close()
    return {"message": "Password changed successfully"}
