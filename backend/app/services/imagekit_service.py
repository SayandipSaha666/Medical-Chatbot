from imagekitio import ImageKit
from ..config import settings
import uuid
# imagekit = ImageKit(
#     settings.imagekit_public_key,
#     settings.imagekit_private_key,
#     settings.imagekit_url_endpoint
# )
import os

imagekit = ImageKit()

imagekit.public_key = os.getenv("IMAGEKIT_PUBLIC_KEY")
imagekit.private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")
imagekit.url_endpoint = os.getenv("IMAGEKIT_URL_ENDPOINT")

# app/services/imagekit_service.py
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

def upload_image(file_bytes: bytes, filename: str, user_id: str):
    if len(file_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image size exceeds 5MB")
    unique_name = f"{uuid.uuid4()}_{filename}"
    return imagekit.upload(
        file=file_bytes,
        file_name=unique_name,
        folder=f"/users/{user_id}/chat-images"
    )
