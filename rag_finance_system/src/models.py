"""
models.py
SQLAlchemy ORM 模型: Conversation / Message / Favorite
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Enum, ForeignKey, JSON,
)
from sqlalchemy.orm import relationship
from .database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    title = Column(String(500), nullable=False, default="新对话")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow, nullable=False)

    messages = relationship("Message", back_populates="conversation",
                            cascade="all, delete-orphan",
                            order_by="Message.created_at")
    favorites = relationship("Favorite", back_populates="conversation",
                             cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id",
                             ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Enum("user", "assistant"), nullable=False)
    content = Column(Text, nullable=False)
    question = Column(Text, nullable=True)
    rewritten_query = Column(Text, nullable=True)
    sources = Column(JSON, nullable=True)
    confidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    favorites = relationship("Favorite", back_populates="message",
                             cascade="all, delete-orphan")


class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(String(36), primary_key=True)
    fav_type = Column(Enum("conversation", "source"), nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversations.id",
                             ondelete="CASCADE"), nullable=True, index=True)
    message_id = Column(String(36), ForeignKey("messages.id",
                        ondelete="CASCADE"), nullable=True, index=True)
    source_data = Column(JSON, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="favorites")
    message = relationship("Message", back_populates="favorites")
