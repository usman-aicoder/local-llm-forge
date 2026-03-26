from fastapi import APIRouter, HTTPException
import httpx
from app.config import settings
from app.services.ollama_service import ollama_service
from app.services.hf_model_service import scan_local_models

router = APIRouter(prefix="/models", tags=["models"])

_SIZE_VRAM = [
    ("70b",  {"qlora_gb": 40, "full_lora_gb": 140}),
    ("34b",  {"qlora_gb": 20, "full_lora_gb": 68}),
    ("13b",  {"qlora_gb": 9,  "full_lora_gb": 26}),
    ("8b",   {"qlora_gb": 6,  "full_lora_gb": 16}),
    ("7b",   {"qlora_gb": 5,  "full_lora_gb": 14}),
    ("3b",   {"qlora_gb": 3,  "full_lora_gb": 7}),
    ("1.5b", {"qlora_gb": 2,  "full_lora_gb": 4}),
    ("1b",   {"qlora_gb": 2,  "full_lora_gb": 4}),
]


def _estimate_vram(model_info: dict) -> dict:
    """Rough VRAM estimate from model ID and tags."""
    haystack = (model_info.get("id", "") + " " + " ".join(model_info.get("tags", []))).lower()
    for size_tag, vram in _SIZE_VRAM:
        if size_tag in haystack:
            return vram
    return {"qlora_gb": "unknown", "full_lora_gb": "unknown"}


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


@router.get("/browse")
async def browse_hf_models(
    search: str = "",
    limit: int = 20,
):
    """
    Search HuggingFace Hub for instruction-tuned text-generation models.
    Returns models sorted by downloads with VRAM estimates and local download status.
    """
    params: dict = {
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "direction": -1,
        "limit": min(limit, 50),
    }
    if search:
        params["search"] = search

    headers: dict = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://huggingface.co/api/models",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
        hf_models = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {exc}")

    # Determine which models are already downloaded locally
    local_names: set[str] = set()
    hf_dir = settings.abs(settings.models_hf_dir)
    if hf_dir.exists():
        local_names = {d.name for d in hf_dir.iterdir() if d.is_dir()}

    def _is_downloaded(model_id: str) -> bool:
        # HF cache uses "--" instead of "/" in directory names
        cache_name = model_id.replace("/", "--")
        return any(cache_name in name or name in model_id for name in local_names)

    return [
        {
            "id": m.get("id", ""),
            "author": m.get("id", "").split("/")[0] if "/" in m.get("id", "") else "",
            "model_name": m.get("id", "").split("/")[-1],
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": m.get("tags", []),
            "is_downloaded": _is_downloaded(m.get("id", "")),
            "vram_estimate": _estimate_vram(m),
        }
        for m in hf_models
        if m.get("id")
    ]
