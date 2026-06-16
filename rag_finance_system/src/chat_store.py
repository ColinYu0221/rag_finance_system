"""
chat_store.py
对话历史 + 收藏 业务逻辑层
"""

import uuid
import json
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .models import Conversation, Message, Favorite


def _new_id() -> str:
    return str(uuid.uuid4())


class ChatStore:
    def __init__(self, db: Session):
        self.db = db

    # ── 对话 ──

    def create_conversation(self, title: str = "新对话") -> Conversation:
        conv = Conversation(id=_new_id(), title=title)
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        return (
            self.db.query(Conversation)
            .order_by(desc(Conversation.updated_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(Conversation.id == conv_id)
            .first()
        )

    def update_conversation_title(self, conv_id: str, title: str) -> bool:
        conv = self.get_conversation(conv_id)
        if not conv:
            return False
        conv.title = title
        self.db.commit()
        return True

    def delete_conversation(self, conv_id: str) -> bool:
        conv = self.get_conversation(conv_id)
        if not conv:
            return False
        self.db.delete(conv)
        self.db.commit()
        return True

    def get_message_count(self, conv_id: str) -> int:
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conv_id)
            .count()
        )

    # ── 消息 ──

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        question: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        sources: Optional[list] = None,
        confidence: Optional[dict] = None,
    ) -> Message:
        msg = Message(
            id=_new_id(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            question=question,
            rewritten_query=rewritten_query,
            sources=sources,
            confidence=confidence,
        )
        self.db.add(msg)

        # 更新对话的 updated_at
        conv = self.get_conversation(conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_messages(self, conversation_id: str) -> list[Message]:
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .all()
        )

    # ── 收藏 ──

    def add_favorite(
        self,
        fav_type: str,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        source_data: Optional[dict] = None,
        note: Optional[str] = None,
    ) -> Favorite:
        fav = Favorite(
            id=_new_id(),
            fav_type=fav_type,
            conversation_id=conversation_id,
            message_id=message_id,
            source_data=source_data,
            note=note,
        )
        self.db.add(fav)
        self.db.commit()
        self.db.refresh(fav)
        return fav

    def list_favorites(self, fav_type: Optional[str] = None, limit: int = 50) -> list[Favorite]:
        q = self.db.query(Favorite).order_by(desc(Favorite.created_at))
        if fav_type:
            q = q.filter(Favorite.fav_type == fav_type)
        return q.limit(limit).all()

    def delete_favorite(self, fav_id: str) -> bool:
        fav = self.db.query(Favorite).filter(Favorite.id == fav_id).first()
        if not fav:
            return False
        self.db.delete(fav)
        self.db.commit()
        return True

    def check_favorited(self, fav_type: str, conversation_id: Optional[str] = None,
                        message_id: Optional[str] = None) -> bool:
        q = self.db.query(Favorite).filter(Favorite.fav_type == fav_type)
        if conversation_id:
            q = q.filter(Favorite.conversation_id == conversation_id)
        if message_id:
            q = q.filter(Favorite.message_id == message_id)
        return q.first() is not None
