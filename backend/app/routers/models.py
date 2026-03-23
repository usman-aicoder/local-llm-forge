from fastapi import APIRouter, HTTPException
from app.services.ollama_service import ollama_service
from app.services.hf_model_service import scan_local_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/ollama")
async def list_ollama_models():
    """List all models currently available in Ollama."""
    try:
        models = await ollama_service.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")


@router.get("/hf")
async def list_hf_models():
    """List HuggingFace model weights downloaded to local storage."""
    models = scan_local_models()
    return {"models": models}
