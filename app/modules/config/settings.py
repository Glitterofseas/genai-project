"""Central configuration. Everything reads from .env; nothing hardcodes a secret."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CONVERSATIONS_PATH = DATA_DIR / "sms_conversations.json"
JOB_DESCRIPTION_PDF = DATA_DIR / "Python Developer Job Description.pdf"

# Split is fixed and committed so every run and every grader sees the same numbers.
TRAIN_CONVERSATION_IDS = tuple(range(1, 11))   # 1..10  -> 40 recruiter turns
TEST_CONVERSATION_IDS = tuple(range(11, 16))   # 11..15 -> 19 recruiter turns


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embed_model: str
    schedule_backend: str
    sqlite_path: Path
    mssql_conn: str
    exit_advisor_model: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        schedule_backend=os.getenv("SCHEDULE_BACKEND", "sqlite").lower(),
        sqlite_path=PROJECT_ROOT / os.getenv("SQLITE_PATH", "data/schedule.sqlite"),
        mssql_conn=os.getenv("MSSQL_CONN", ""),
        exit_advisor_model=os.getenv("EXIT_ADVISOR_MODEL", ""),
    )
