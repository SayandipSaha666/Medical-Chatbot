from sqlalchemy import Column, Integer, String, ForeignKey, Text, JSON, TIMESTAMP, text
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    clerk_user_id = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    chats = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    images = relationship(
        "ChatImage",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        onupdate=text("now()")
    )

    user = relationship("User", back_populates="chats")

    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan"
    )

    images = relationship(
        "ChatImage",
        back_populates="chat",
        cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=lambda: [])
    parent_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    chat = relationship("Chat", back_populates="messages")

    parent_message = relationship(
        "Message",
        remote_side=[id],
        backref="child_messages"
    )

    images = relationship(
        "ChatImage",
        back_populates="message",
        cascade="all, delete-orphan"
    )


class ChatImage(Base):
    __tablename__ = "chat_images"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(
        Integer,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True
    )

    imagekit_file_id = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    thumbnail_url = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    user = relationship("User", back_populates="images")
    chat = relationship("Chat", back_populates="images")
    message = relationship("Message", back_populates="images")
