"""
Test suite for YouTube Toxicity Analyzer.
Run: pytest tests/ -v --cov=app --cov-report=term-missing
"""
import pytest
import json
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client with mocked model."""
    with patch("app.sentiment_pipeline") as mock_pipe:
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.10},
            {"label": "neutral",  "score": 0.05},
        ]]
        from app import app as flask_app
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c


MOCK_COMMENTS = [
    {"text": "Great video!", "author": "User1", "likes": 5, "published": "2024-01-01"},
    {"text": "This is terrible", "author": "User2", "likes": 0, "published": "2024-01-02"},
    {"text": "ok", "author": "User3", "likes": 1, "published": "2024-01-03"},
]


# ── URL Extraction Tests ───────────────────────────────────────

def test_extract_video_id_standard():
    from app import extract_video_id
    vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert vid == "dQw4w9WgXcQ"

def test_extract_video_id_short():
    from app import extract_video_id
    vid = extract_video_id("https://youtu.be/dQw4w9WgXcQ")
    assert vid == "dQw4w9WgXcQ"

def test_extract_video_id_embed():
    from app import extract_video_id
    vid = extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
    assert vid == "dQw4w9WgXcQ"

def test_extract_video_id_invalid():
    from app import extract_video_id
    assert extract_video_id("https://google.com") is None
    assert extract_video_id("not a url") is None
    assert extract_video_id("") is None


# ── Input Validation Tests ────────────────────────────────────

def test_analyze_missing_url(client):
    res = client.post("/analyze",
        data=json.dumps({}),
        content_type="application/json")
    assert res.status_code == 400
    assert "required" in res.get_json()["error"].lower()

def test_analyze_empty_url(client):
    res = client.post("/analyze",
        data=json.dumps({"url": "   "}),
        content_type="application/json")
    assert res.status_code == 400

def test_analyze_invalid_url(client):
    res = client.post("/analyze",
        data=json.dumps({"url": "https://google.com"}),
        content_type="application/json")
    assert res.status_code == 400
    assert "Invalid" in res.get_json()["error"]


# ── Classification Tests ──────────────────────────────────────

def test_classify_positive():
    from app import classify_comment
    with patch("app.sentiment_pipeline") as mock:
        mock.return_value = [[
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.05},
            {"label": "neutral",  "score": 0.05},
        ]]
        result = classify_comment("I love this!")
        assert result["label"] == "Love"
        assert result["confidence"] == 90.0

def test_classify_negative():
    from app import classify_comment
    with patch("app.sentiment_pipeline") as mock:
        mock.return_value = [[
            {"label": "positive", "score": 0.05},
            {"label": "negative", "score": 0.90},
            {"label": "neutral",  "score": 0.05},
        ]]
        result = classify_comment("This is awful")
        assert result["label"] == "Hate"

def test_classify_neutral():
    from app import classify_comment
    with patch("app.sentiment_pipeline") as mock:
        mock.return_value = [[
            {"label": "positive", "score": 0.30},
            {"label": "negative", "score": 0.30},
            {"label": "neutral",  "score": 0.40},
        ]]
        result = classify_comment("ok")
        assert result["label"] == "Neutral"


# ── Toxicity Score Tests ──────────────────────────────────────

def test_toxicity_score_all_hate():
    from app import compute_toxicity_score
    comments = [{"sentiment": {"label": "Hate"}} for _ in range(10)]
    assert compute_toxicity_score(comments) == 100

def test_toxicity_score_all_love():
    from app import compute_toxicity_score
    comments = [{"sentiment": {"label": "Love"}} for _ in range(10)]
    assert compute_toxicity_score(comments) == 0

def test_toxicity_score_empty():
    from app import compute_toxicity_score
    assert compute_toxicity_score([]) == 0

def test_toxicity_score_mixed():
    from app import compute_toxicity_score
    comments = (
        [{"sentiment": {"label": "Hate"}}] * 5 +
        [{"sentiment": {"label": "Love"}}] * 5
    )
    score = compute_toxicity_score(comments)
    assert 40 <= score <= 60


# ── Health Endpoint ───────────────────────────────────────────

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "cache_hit_rate_pct" in data
    assert "total_requests" in data


# ── Cache Tests ───────────────────────────────────────────────

def test_cache_hit_returns_faster(client):
    """Second request for same video should return X-Cache: HIT."""
    with patch("app.fetch_comments", return_value=MOCK_COMMENTS):
        res1 = client.post("/analyze",
            data=json.dumps({"url": "https://youtu.be/dQw4w9WgXcQ"}),
            content_type="application/json")
        assert res1.status_code == 200

        res2 = client.post("/analyze",
            data=json.dumps({"url": "https://youtu.be/dQw4w9WgXcQ"}),
            content_type="application/json")
        assert res2.status_code == 200
        assert res2.headers.get("X-Cache") == "HIT"

def test_different_videos_not_cached(client):
    """Different video IDs should get separate cache entries."""
    with patch("app.fetch_comments", return_value=MOCK_COMMENTS):
        res1 = client.post("/analyze",
            data=json.dumps({"url": "https://youtu.be/aaaaaaaaaaa"}),
            content_type="application/json")
        res2 = client.post("/analyze",
            data=json.dumps({"url": "https://youtu.be/bbbbbbbbbbb"}),
            content_type="application/json")
        assert res1.headers.get("X-Cache") == "MISS"
        assert res2.headers.get("X-Cache") == "MISS"
