from app.models.project import Project
from app.models.dataset import Dataset, DatasetStats
from app.models.job import TrainingJob
from app.models.checkpoint import Checkpoint
from app.models.evaluation import Evaluation
from app.models.rag import RAGCollection, RAGDocument
from app.models.task import TaskRecord

ALL_MODELS = [
    Project,
    Dataset,
    TrainingJob,
    Checkpoint,
    Evaluation,
    RAGCollection,
    RAGDocument,
    TaskRecord,
]
