# YouTube Comment Toxicity Analyzer

A multilingual NLP tool that fetches YouTube comments and classifies
each one as **Love**, **Hate**, **Neutral**, or **Sarcasm** — with
confidence scores, a toxicity dashboard, and real-time filtering.

🔗 **Live Demo**: https://youtube-toxicity-454795031838.asia-south1.run.app

---

## Features

- Multilingual sentiment model (English + Hinglish + Hindi-romanized)
- Per-comment confidence scores (not just hard labels)
- Overall toxicity score 0–100 with animated bar
- Sentiment breakdown with percentage bars
- Filter comments by label + pagination (15 per page)
- SHA-256 keyed response caching (1hr TTL, 100 entries)
- Rate limiting: 10 requests/min per IP
- `/health` endpoint with cache hit rate and uptime
- Deployed on Google Cloud Run (asia-south1)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.0 |
| NLP Model | cardiffnlp/twitter-xlm-roberta-base-sentiment |
| YouTube Data | YouTube Data API v3 |
| Caching | cachetools TTLCache (SHA-256 keyed) |
| Rate Limiting | Flask-Limiter |
| Server | Gunicorn (2 workers, 4 threads) |
| Deployment | Google Cloud Run, Docker |
| Testing | pytest, unittest.mock |

---

## Local Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/yourname/youtube-toxicity
cd youtube-toxicity

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your YOUTUBE_API_KEY

# 5. Run the app
python app.py
# Visit http://localhost:8080
```

---

## Docker Setup

```bash
docker build -t youtube-toxicity .
docker run -p 8080:8080 --env-file .env youtube-toxicity
```

---

## Deploy to Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/youtube-toxicity

# Deploy to Cloud Run
gcloud run deploy youtube-toxicity \
  --image gcr.io/YOUR_PROJECT/youtube-toxicity \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars YOUTUBE_API_KEY=your_key_here
```

---

## API Reference

### POST /analyze
Analyzes comments for a YouTube video.

**Request:**
```json
{ "url": "https://www.youtube.com/watch?v=VIDEO_ID" }
```

**Response:**
```json
{
  "video_id": "VIDEO_ID",
  "total_comments": 100,
  "toxicity_score": 23,
  "label_counts": { "Love": 45, "Hate": 23, "Neutral": 28, "Sarcasm": 4 },
  "label_percentages": { "Love": 45.0, "Hate": 23.0, "Neutral": 28.0, "Sarcasm": 4.0 },
  "comments": [
    {
      "text": "Great video!",
      "author": "User1",
      "likes": 5,
      "published": "2024-01-01",
      "sentiment": {
        "label": "Love",
        "color": "love",
        "confidence": 90.0,
        "scores": { "positive": 90.0, "negative": 5.0, "neutral": 5.0 }
      }
    }
  ]
}
```

**Headers:**
- `X-Cache: HIT` — result served from cache
- `X-Cache: MISS` — result freshly computed

**Rate limit:** 10/min, 50/hr per IP. Returns 429 if exceeded.

---

### GET /health
Returns application health metrics.

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "cache_size": 12,
  "cache_hits": 45,
  "cache_misses": 23,
  "cache_hit_rate_pct": 66.2,
  "total_requests": 68
}
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `YOUTUBE_API_KEY` | Yes | YouTube Data API v3 key from Google Cloud Console |
| `FLASK_ENV` | No | `production` or `development` (default: development) |

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

Target: 80%+ coverage.

---

## How to get a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create credentials → API Key
5. Copy the key into your `.env` file
