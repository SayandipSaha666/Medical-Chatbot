from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from .. import models
from ..database import get_db
from ..dependencies.clerk_auth import get_current_user
from ..Schemas import ChatCreate, ChatResponse,ChatDetail
from .. import models

router = APIRouter(
    prefix="/api/chats",
    tags=["Chats"]
)


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    data: ChatCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chat = models.Chat(
        user_id=current_user.id,
        title=data.title
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/", response_model=list[ChatResponse],status_code=status.HTTP_200_OK)
def get_chats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chats = (
        db.query(models.Chat)
        .filter(models.Chat.user_id == current_user.id)
        .order_by(models.Chat.updated_at.desc())
        .all()
    )
    return chats


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chat = (
        db.query(models.Chat)
        .filter(
            models.Chat.id == chat_id,
            models.Chat.user_id == current_user.id
        )
        .first()
    )

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )

    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chat = (
        db.query(models.Chat)
        .filter(
            models.Chat.id == chat_id,
            models.Chat.user_id == current_user.id
        )
        .first()
    )

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )

    db.delete(chat)
    db.commit()
    return

@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: int,
    data: ChatCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
    ):
    chat_query = db.query(models.Chat).filter(
        models.Chat.id == chat_id,
        models.Chat.user_id == current_user.id
    )

    if not chat_query.first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    
    chat_query.update(data.model_dump(exclude_unset=True), synchronize_session=False)
    db.commit()
    return chat_query.first()