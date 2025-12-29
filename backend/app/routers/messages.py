from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
import markdown
from ..database import get_db
from ..dependencies.clerk_auth import get_current_user
from ..Schemas import MessageCreate, MessageOut
from .. import models   
from ..services.langchain_chain import get_chain,invoke_chain_async
from fastapi.responses import StreamingResponse
import asyncio

router = APIRouter(
    prefix="/api/chats/{chat_id}/messages",
    tags=["Messages"]
)

@router.get("/", response_model=list[MessageOut])
def get_messages(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chat = db.query(models.Chat).filter(
        models.Chat.id == chat_id,
        models.Chat.user_id == current_user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return (
        db.query(models.Message)
        .filter(models.Message.chat_id == chat_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )


@router.post("/", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
async def send_message(
    chat_id: int,
    data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Validate chat ownership
    chat = db.query(models.Chat).filter(
        models.Chat.id == chat_id,
        models.Chat.user_id == current_user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if not data.content.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1️⃣ Save user message
    user_message = models.Message(
        chat_id=chat_id,
        role="user",
        content=data.content
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # 2️⃣ Initialize chain (lazy)
    chain = get_chain()

    # 3️⃣ Invoke chain
    try:
        # raw_response = chain.invoke(data.content)
        raw_response = await invoke_chain_async(chain, data.content) # Updated
        answer_text = raw_response["answer"]
        sources = raw_response.get("sources", [])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM error: {str(e)}"
        )

    # 4️⃣ Markdown → HTML (same as Flask)
    formatted_response = markdown.markdown(
        # raw_response,
        answer_text,
        extensions=["fenced_code", "tables"]
    )

    # 5️⃣ Save assistant message
    assistant_message = models.Message(
        chat_id=chat_id,
        role="assistant",
        content=formatted_response,
        sources=sources,  # optional: add retriever metadata later
        parent_message_id=user_message.id
    )

    db.add(assistant_message)

    # Update chat timestamp
    chat.updated_at = assistant_message.created_at

    db.commit()
    db.refresh(assistant_message)

    return assistant_message

@router.post("/stream")
async def stream_message(
    chat_id: int,
    data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # 1️⃣ Validate chat ownership
    chat = db.query(models.Chat).filter(
        models.Chat.id == chat_id,
        models.Chat.user_id == current_user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if not data.content.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 2️⃣ Save user message
    user_message = models.Message(
        chat_id=chat_id,
        role="user",
        content=data.content
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # 3️⃣ Get chain
    chain = get_chain()

    async def token_generator():
        """
        Streams tokens to frontend and saves final response to DB
        """
        collected_text = ""
        collected_sources = []

        try:
            # LangChain async streaming
            async for chunk in chain.astream(data.content):

                # If your chain returns dict chunks
                if isinstance(chunk, dict):
                    token = chunk.get("answer", "")
                    collected_sources = chunk.get("sources", collected_sources)
                else:
                    token = str(chunk)

                collected_text += token
                yield token

        except Exception as e:
            yield f"\n\n[ERROR]: {str(e)}"
            return

        # 4️⃣ Markdown → HTML
        formatted_response = markdown.markdown(
            collected_text,
            extensions=["fenced_code", "tables"]
        )

        # 5️⃣ Save assistant message AFTER streaming ends
        assistant_message = models.Message(
            chat_id=chat_id,
            role="assistant",
            content=formatted_response,
            sources=collected_sources,
            parent_message_id=user_message.id
        )

        db.add(assistant_message)
        chat.updated_at = assistant_message.created_at
        db.commit()

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
