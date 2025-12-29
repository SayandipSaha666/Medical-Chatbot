from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from app.database import get_db
from app.dependencies.clerk_auth import get_current_user
from app.services.imagekit_service import upload_image
from app import models

router = APIRouter(
    prefix="/api/chats/{chat_id}/images",
    tags=["Images"]
)

@router.post("")
async def upload_chat_image(
    chat_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    # Validate chat ownership
    chat = db.query(models.Chat).filter(
        models.Chat.id == chat_id,
        models.Chat.user_id == current_user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Read image bytes
    image_bytes = await file.read()

    # Upload to ImageKit
    upload_result = upload_image(
        file_bytes=image_bytes,
        filename=file.filename,
        user_id=str(current_user.id)
    )

    # Save metadata to DB
    image = models.ChatImage(
        user_id=current_user.id,
        chat_id=chat_id,
        imagekit_file_id=upload_result.file_id,
        image_url=upload_result.url,
        thumbnail_url=upload_result.thumbnail_url
    )

    db.add(image)
    db.commit()
    db.refresh(image)

    return {
        "id": image.id,
        "image_url": image.image_url,
        "thumbnail_url": image.thumbnail_url
    }
