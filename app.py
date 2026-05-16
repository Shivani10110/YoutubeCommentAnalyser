import os
import logging
import hashlib
import time
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
from cachetools import TTLCache
from dotenv import load_dotenv
import re

load_dotenv()

# ── Logging Setup ──────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler("logs/app.log", maxBytes=10_000_000, backupCount=5)
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
))
logger.addHandler(handler)

# ── App Init ───────────────────────────────────────────────────
app = Flask(__name__)

# ── Rate Limiting ──────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ── Cache: SHA-256 keyed, 100 entries, 1hr TTL ─────────────────
cache = TTLCache(maxsize=100, ttl=3600)
cache_hits = 0
cache_misses = 0
total_requests = 0
start_time = time.time()

# ── Multilingual Sentiment Model ───────────────────────────────
# Uses cardiffnlp/twitter-xlm-roberta-base-sentiment
# Works on English + Hinglish + Hindi-romanized text
logger.info("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None,
    truncation=True,
    max_length=512
)
logger.info("Model loaded successfully.")

# ── YouTube API ────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_comments(video_id: str, max_comments: int = 100) -> list[dict]:
    """Fetch top-level comments from YouTube Data API v3."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            ).execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "likes": snippet["likeCount"],
                    "published": snippet["publishedAt"][:10]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        raise

    return comments


def classify_comment(text: str) -> dict:
    """
    Classify a comment using multilingual XLM-RoBERTa.
    Maps POSITIVE/NEGATIVE/NEUTRAL + confidence to
    Love / Hate / Neutral / Sarcasm labels.
    """
    try:
        results = sentiment_pipeline(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}

        pos = scores.get("positive", 0)
        neg = scores.get("negative", 0)
        neu = scores.get("neutral", 0)
        confidence = max(pos, neg, neu)

        # Sarcasm heuristic: high positive score but contains
        # negative trigger words → likely sarcasm
        sarcasm_triggers = [
            "lol", "haha", "sure", "obviously", "totally",
            "great job", "wow", "amazing", "fantastic",
            "genius", "teri", "waah", "wah"
        ]
        text_lower = text.lower()
        has_sarcasm_signal = any(t in text_lower for t in sarcasm_triggers)

        if neg > 0.55:
            label = "Hate"
            color = "hate"
        elif pos > 0.65 and has_sarcasm_signal and neg > 0.15:
            label = "Sarcasm"
            color = "sarcasm"
        elif pos > 0.55:
            label = "Love"
            color = "love"
        else:
            label = "Neutral"
            color = "neutral"

        return {
            "label": label,
            "color": color,
            "confidence": round(confidence * 100, 1),
            "scores": {
                "positive": round(pos * 100, 1),
                "negative": round(neg * 100, 1),
                "neutral": round(neu * 100, 1)
            }
        }
    except Exception as e:
        logger.warning(f"Classification error: {e}")
        return {"label": "Neutral", "color": "neutral", "confidence": 50.0,
                "scores": {"positive": 0, "negative": 0, "neutral": 100}}


def compute_toxicity_score(classified: list[dict]) -> int:
    """Compute overall toxicity score 0–100."""
    if not classified:
        return 0
    hate_count = sum(1 for c in classified if c["sentiment"]["label"] == "Hate")
    sarcasm_count = sum(1 for c in classified if c["sentiment"]["label"] == "Sarcasm")
    score = (hate_count * 1.0 + sarcasm_count * 0.4) / len(classified) * 100
    return min(100, round(score))


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@limiter.limit("10 per minute; 50 per hour")
def analyze():
    global cache_hits, cache_misses, total_requests
    total_requests += 1
    start = time.time()

    data = request.get_json()
    url = (data or {}).get("url", "").strip()

    if not url:
        return jsonify({"error": "YouTube URL is required."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL. Could not extract video ID."}), 400

    # Cache lookup
    cache_key = hashlib.sha256(video_id.encode()).hexdigest()
    if cache_key in cache:
        cache_hits += 1
        logger.info(f"CACHE HIT | video={video_id} | total_requests={total_requests}")
        response = jsonify(cache[cache_key])
        response.headers["X-Cache"] = "HIT"
        return response

    cache_misses += 1

    try:
        comments = fetch_comments(video_id, max_comments=100)
    except HttpError as e:
        return jsonify({"error": f"YouTube API error: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"Unexpected fetch error: {e}")
        return jsonify({"error": "Failed to fetch comments."}), 500

    if not comments:
        return jsonify({"error": "No comments found for this video."}), 404

    # Classify each comment
    classified = []
    for c in comments:
        sentiment = classify_comment(c["text"])
        classified.append({**c, "sentiment": sentiment})

    # Aggregate stats
    label_counts = {"Love": 0, "Hate": 0, "Neutral": 0, "Sarcasm": 0}
    for c in classified:
        label = c["sentiment"]["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(classified)
    label_pct = {k: round(v / total * 100, 1) for k, v in label_counts.items()}
    toxicity_score = compute_toxicity_score(classified)

    result = {
        "video_id": video_id,
        "total_comments": total,
        "toxicity_score": toxicity_score,
        "label_counts": label_counts,
        "label_percentages": label_pct,
        "comments": classified,
        "cached": False
    }

    cache[cache_key] = {**result, "cached": True}

    elapsed = round((time.time() - start) * 1000)
    logger.info(
        f"CACHE MISS | video={video_id} | comments={total} | "
        f"toxicity={toxicity_score} | elapsed={elapsed}ms"
    )

    response = jsonify(result)
    response.headers["X-Cache"] = "MISS"
    return response


@app.route("/health")
def health():
    uptime = round(time.time() - start_time)
    hit_rate = (
        round(cache_hits / (cache_hits + cache_misses) * 100, 1)
        if (cache_hits + cache_misses) > 0 else 0
    )
    return jsonify({
        "status": "ok",
        "uptime_seconds": uptime,
        "cache_size": len(cache),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate_pct": hit_rate,
        "total_requests": total_requests
    })


@app.errorhandler(429)
def ratelimit_error(e):
    return jsonify({
        "error": "Too many requests. Please wait before trying again.",
        "retry_after": str(e.description)
    }), 429


if __name__ == "__main__":
    app.run(debug=False, port=8080)
