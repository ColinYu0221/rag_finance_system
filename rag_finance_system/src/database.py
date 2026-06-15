"""
database.py
SQLAlchemy 引擎与会话管理 (MySQL)
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "rag_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "rag123456")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_finance")

SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """创建所有表（如不存在）"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖注入用，自动关闭 session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
