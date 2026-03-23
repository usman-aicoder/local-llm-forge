"""
Celery worker entry point.

Start with:
    celery -A celery_worker worker --loglevel=info

The worker_pool=solo setting in celery_app.py is required for GPU training —
Celery's default fork-based pool causes CUDA/bitsandbytes to deadlock on the
first training step. solo pool runs tasks in-process without forking.
"""
from workers.celery_app import celery_app  # noqa: F401 — registers all tasks via include=

if __name__ == "__main__":
    celery_app.start()
