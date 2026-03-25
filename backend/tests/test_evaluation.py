"""
Tests for evaluation utilities — especially _extract_id() which handles
multiple MongoDB reference types (DBRef, Beanie Link, plain ObjectId).

These are pure unit tests (no DB, no HTTP client needed).
"""
import pytest
from bson import ObjectId, DBRef


# ── _extract_id() ─────────────────────────────────────────────────────────────

def test_extract_id_plain_objectid():
    from workers.evaluation_tasks import _extract_id
    oid = ObjectId()
    assert _extract_id(oid) == str(oid)


def test_extract_id_from_dbref():
    from workers.evaluation_tasks import _extract_id
    oid = ObjectId()
    ref = DBRef("datasets", oid)
    assert _extract_id(ref) == str(oid)


def test_extract_id_from_string():
    from workers.evaluation_tasks import _extract_id
    oid = ObjectId()
    assert _extract_id(str(oid)) == str(oid)


def test_extract_id_from_object_with_id_attr():
    """Simulates a Beanie Link-like object that has a .id attribute."""
    from workers.evaluation_tasks import _extract_id

    class FakeLink:
        def __init__(self, oid):
            self.id = oid

    oid = ObjectId()
    link = FakeLink(oid)
    assert _extract_id(link) == str(oid)


def test_extract_id_returns_string():
    """Return type must always be str regardless of input type."""
    from workers.evaluation_tasks import _extract_id
    oid = ObjectId()
    result = _extract_id(DBRef("col", oid))
    assert isinstance(result, str)


def test_extract_id_different_objectids_produce_different_strings():
    from workers.evaluation_tasks import _extract_id
    oid1, oid2 = ObjectId(), ObjectId()
    assert _extract_id(oid1) != _extract_id(oid2)


# ── ROUGE / BLEU calculation ──────────────────────────────────────────────────

def test_rouge_score_import():
    """rouge_score library must be installed."""
    from rouge_score import rouge_scorer  # noqa: F401


def test_nltk_bleu_import():
    """nltk must be installed with bleu support."""
    from nltk.translate.bleu_score import sentence_bleu  # noqa: F401


def test_rouge_l_perfect_match():
    """Identical strings should produce ROUGE-L = 1.0."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    text = "The quick brown fox jumps over the lazy dog"
    scores = scorer.score(text, text)
    assert scores["rougeL"].fmeasure == pytest.approx(1.0)


def test_rouge_l_no_overlap():
    """Completely different strings should produce ROUGE-L close to 0."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score("aaa bbb ccc", "xxx yyy zzz")
    assert scores["rougeL"].fmeasure == pytest.approx(0.0)


def test_bleu_perfect_match():
    """BLEU score of a sentence against itself should be 1.0."""
    from nltk.translate.bleu_score import sentence_bleu
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    score = sentence_bleu([tokens], tokens)
    assert score == pytest.approx(1.0)


def test_rouge_scores_between_0_and_1(tmp_path):
    """All ROUGE variants must stay in [0, 1]."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    ref = "Paris is the capital of France and a major European city"
    hyp = "France's capital city is Paris located in western Europe"
    scores = scorer.score(ref, hyp)
    for key, score in scores.items():
        assert 0.0 <= score.fmeasure <= 1.0, f"{key} out of range: {score.fmeasure}"


# ── Evaluation API ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_evaluations_empty(client, project):
    resp = await client.get(f"/projects/{project['id']}/evaluations")
    assert resp.status_code == 200
    assert resp.json().get("evaluations", []) == []


@pytest.mark.asyncio
async def test_run_eval_on_nonexistent_job_returns_404(client):
    resp = await client.post("/jobs/000000000000000000000000/evaluate")
    assert resp.status_code == 404
