from celery import Celery
from app.config import settings

celery_app = Celery(
    "llm_platform",
    broker=settings.redis_url,
    backend=f"mongodb://{settings.mongo_url.replace('mongodb://', '')}/{settings.mongo_db_name}",
    include=[
        "workers.training_tasks",
        "workers.dataset_tasks",
        "workers.evaluation_tasks",
        "workers.export_tasks",
        "workers.rag_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker (important for GPU)
    result_expires=86400,          # results kept for 24 hours
    # solo pool avoids fork()-based workers which deadlock CUDA/bitsandbytes
    worker_pool="solo",
)
