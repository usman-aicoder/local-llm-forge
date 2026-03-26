"""
Phase 4.3 — Dataset augmentation tests.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Unit: augment_dataset function ────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _mock_ollama_response(pairs: list[dict]):
    """Return a mock httpx response with a JSON array of instruction/output pairs."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"response": json.dumps(pairs)}
    return mock


def test_augment_returns_more_rows_than_input(tmp_path):
    from ml.augment import augment_dataset

    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "Capital of France?", "output": "Paris"},
    ])
    output_file = tmp_path / "out.jsonl"

    fake_pairs = [
        {"instruction": "How much is 2 plus 2?", "output": "4"},
        {"instruction": "What does 2+2 equal?", "output": "4"},
    ]

    with patch("httpx.post", return_value=_mock_ollama_response(fake_pairs)):
        result = augment_dataset(
            str(input_file), str(output_file),
            ollama_url="http://localhost:11434",
            model_name="test-model",
            paraphrases_per_row=2,
        )

    assert result["augmented_rows"] > result["original_rows"]
    assert result["new_rows_added"] > 0
    assert output_file.exists()


def test_augment_preserves_original_rows(tmp_path):
    from ml.augment import augment_dataset

    original_rows = [
        {"instruction": "What is 1+1?", "output": "2"},
        {"instruction": "What color is the sky?", "output": "Blue"},
    ]
    input_file = tmp_path / "orig.jsonl"
    _write_jsonl(input_file, original_rows)
    output_file = tmp_path / "aug.jsonl"

    fake_pairs = [{"instruction": "How much is 1+1?", "output": "2"}]
    with patch("httpx.post", return_value=_mock_ollama_response(fake_pairs)):
        augment_dataset(
            str(input_file), str(output_file),
            ollama_url="http://localhost:11434",
            model_name="test",
            paraphrases_per_row=1,
        )

    output_rows = [json.loads(l) for l in output_file.read_text().splitlines() if l.strip()]
    output_instructions = [r["instruction"] for r in output_rows]
    for orig in original_rows:
        assert orig["instruction"] in output_instructions


def test_augment_handles_ollama_failure_gracefully(tmp_path):
    """Row failures must not crash — augmentation continues with remaining rows."""
    from ml.augment import augment_dataset
    import httpx

    input_file = tmp_path / "inp.jsonl"
    _write_jsonl(input_file, [
        {"instruction": "Q1?", "output": "A1"},
        {"instruction": "Q2?", "output": "A2"},
    ])
    output_file = tmp_path / "out.jsonl"

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        result = augment_dataset(
            str(input_file), str(output_file),
            ollama_url="http://localhost:11434",
            model_name="test",
            paraphrases_per_row=2,
        )

    # Originals preserved even when Ollama fails
    assert result["original_rows"] == 2
    assert result["augmented_rows"] == 2   # originals only
    assert result["rows_failed"] == 2
    assert output_file.exists()


def test_augment_respects_max_rows_limit(tmp_path):
    from ml.augment import augment_dataset

    rows = [{"instruction": f"Q{i}?", "output": f"A{i}"} for i in range(10)]
    input_file = tmp_path / "big.jsonl"
    _write_jsonl(input_file, rows)
    output_file = tmp_path / "out.jsonl"

    call_count = 0
    fake_pairs = [{"instruction": "paraphrase", "output": "paraphrase"}]

    def counting_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_ollama_response(fake_pairs)

    with patch("httpx.post", side_effect=counting_post):
        augment_dataset(
            str(input_file), str(output_file),
            ollama_url="http://localhost:11434",
            model_name="test",
            paraphrases_per_row=1,
            max_rows_to_augment=3,   # only augment first 3
        )

    assert call_count == 3


def test_augment_logs_progress(tmp_path):
    from ml.augment import augment_dataset

    input_file = tmp_path / "log.jsonl"
    _write_jsonl(input_file, [{"instruction": "Q?", "output": "A"}])
    output_file = tmp_path / "out.jsonl"

    logs: list[str] = []
    fake_pairs = [{"instruction": "paraphrase", "output": "paraphrase"}]

    with patch("httpx.post", return_value=_mock_ollama_response(fake_pairs)):
        augment_dataset(
            str(input_file), str(output_file),
            ollama_url="http://localhost:11434",
            model_name="test",
            paraphrases_per_row=1,
            on_log=logs.append,
        )

    assert any("Augment" in l or "augment" in l for l in logs)


def test_parse_pairs_handles_json_in_freetext():
    from ml.augment import _parse_pairs

    raw = 'Sure! Here you go:\n[{"instruction": "Hi?", "output": "Hello!"}]\nHope this helps.'
    pairs = _parse_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["instruction"] == "Hi?"


def test_parse_pairs_handles_markdown_fences():
    from ml.augment import _parse_pairs

    raw = '```json\n[{"instruction": "A?", "output": "B"}]\n```'
    pairs = _parse_pairs(raw)
    assert len(pairs) == 1


def test_parse_pairs_returns_empty_on_bad_json():
    from ml.augment import _parse_pairs
    assert _parse_pairs("not json at all") == []


# ── Integration: /datasets/{id}/augment endpoint ─────────────────────────────

@pytest.mark.asyncio
async def test_augment_endpoint_requires_formatted_dataset(client):
    import io
    proj = await client.post("/projects", json={"name": "aug-test"})
    proj_id = proj.json()["id"]

    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(b'{"text":"hi"}\n'), "application/json")},
        data={"name": "raw-ds"},
    )
    ds_id = ds.json()["id"]
    # Status is 'uploaded' — augment should reject it
    resp = await client.post(f"/datasets/{ds_id}/augment")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_augment_endpoint_returns_task_id(client, tmp_path):
    import io
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    proj = await client.post("/projects", json={"name": "aug-test2"})
    proj_id = proj.json()["id"]

    ds_data = b'{"instruction":"Q?","output":"A"}\n'
    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(ds_data), "application/json")},
        data={"name": "fmt-ds"},
    )
    ds_id = ds.json()["id"]

    # Write the actual file and set status
    fmt_path = tmp_path / f"{ds_id}.jsonl"
    fmt_path.write_text('{"instruction":"Q?","output":"A"}\n')

    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["datasets"].update_one(
        {"_id": ObjectId(ds_id)},
        {"$set": {"status": "formatted", "file_path": str(fmt_path)}},
    )
    motor.close()

    resp = await client.post(
        f"/datasets/{ds_id}/augment",
        params={"ollama_model": "test-model", "paraphrases_per_row": 1},
    )
    assert resp.status_code == 200
    assert "task_id" in resp.json()
