"""Pydantic models for API request/response."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Document Upload ──

class UploadResponse(BaseModel):
    filename: str
    file_path: str
    doc_type: str
    size_bytes: int


# ── Index ──

class IndexRequest(BaseModel):
    file_path: str
    doc_type: str = "law"


class IndexResponse(BaseModel):
    chunk_count: int
    doc_type: str
    file_path: str


# ── Search ──

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    doc_type_filter: Optional[str] = None
    law_name_filter: Optional[str] = None
    authority_filter: Optional[str] = None
    use_reranker: bool = True


class SearchResultItem(BaseModel):
    text: str
    source: str
    article_num: str
    score: float
    law_name: str
    doc_type: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


# ── Q&A ──

class QARequest(BaseModel):
    question: str
    doc_type_filter: Optional[str] = None
    use_reranker: bool = True
    use_query_rewrite: bool = True
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)


class SourceItem(BaseModel):
    source: str
    article_num: str
    text: str
    score: float


class ConfidenceScores(BaseModel):
    total: float
    retrieval: float
    coverage: float


class QAResponse(BaseModel):
    question: str
    answer: str
    rewritten_query: Optional[str] = None
    sources: list[SourceItem]
    confidence: ConfidenceScores


# ── Conversation / History ──

class ConversationCreate(BaseModel):
    title: str = "新对话"


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    question: Optional[str] = None
    rewritten_query: Optional[str] = None
    sources: Optional[list] = None
    confidence: Optional[dict] = None
    created_at: datetime


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[MessageOut]
    created_at: datetime
    updated_at: datetime


# ── Favorites ──

class FavoriteCreate(BaseModel):
    fav_type: str = Field(..., pattern=r"^(conversation|source)$")
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    source_data: Optional[dict] = None
    note: Optional[str] = None


class FavoriteOut(BaseModel):
    id: str
    fav_type: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    source_data: Optional[dict] = None
    note: Optional[str] = None
    created_at: datetime


class FavoritesCheckOut(BaseModel):
    favorited: bool


# ── QA 流式请求扩展 ──

class QAStreamRequest(QARequest):
    conversation_id: Optional[str] = Field(None, description="已有对话ID，为空则新建")
