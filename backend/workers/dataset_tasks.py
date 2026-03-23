from workers.celery_app import celery_app


@celery_app.task(bind=True, name="workers.dataset_tasks.run_eda_task")
def run_eda_task(self, dataset_id: str) -> dict:
    """EDA pipeline task. Implemented in Phase 2."""
    return {"status": "placeholder", "dataset_id": dataset_id}


@celery_app.task(bind=True, name="workers.dataset_tasks.run_clean_task")
def run_clean_task(self, dataset_id: str, config: dict) -> dict:
    """Cleaning pipeline task. Implemented in Phase 2."""
    return {"status": "placeholder", "dataset_id": dataset_id}


@celery_app.task(bind=True, name="workers.dataset_tasks.run_format_task")
def run_format_task(self, dataset_id: str, format_type: str, base_model: str) -> dict:
    """Formatting task. Implemented in Phase 2."""
    return {"status": "placeholder", "dataset_id": dataset_id}


@celery_app.task(bind=True, name="workers.dataset_tasks.run_tokenize_task")
def run_tokenize_task(self, dataset_id: str, max_seq_len: int, val_split: float) -> dict:
    """Tokenize + split task. Implemented in Phase 2."""
    return {"status": "placeholder", "dataset_id": dataset_id}
