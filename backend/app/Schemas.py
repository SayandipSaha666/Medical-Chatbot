from pydantic import BaseModel,ConfigDict,EmailStr
from typing import List,Dict,Any
from typing import Optional,Literal
from datetime import datetime
from pydantic.types import conint

class UserIn(BaseModel):
    name: Optional[str]

class UserResponse(BaseModel):
    id: int
    clerk_user_id: str
    name: str
    email: EmailStr
    model_config = ConfigDict(from_attributes=True) 

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ChatResponse(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatDetail(ChatResponse):
    messages: List["MessageOut"]


class MessageCreate(BaseModel):
    content: str


class MessageOut(BaseModel):
    id: int
    chat_id: int
    role: str            # "user" | "assistant"
    content: str
    sources: List[Dict[str, Any]]
    parent_message_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

ChatDetail.model_rebuild()
