import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

# Normalize DATABASE_URL for postgres://
def normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

DEFAULT_SQLITE = "sqlite:///eco.db"
DATABASE_URL = normalize_db_url(os.environ.get("DATABASE_URL", DEFAULT_SQLITE))

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()
