# app/celery_app.py
from celery import Celery
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

celery = Celery(
    "pfe2026",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["app.research.research_task"]
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)