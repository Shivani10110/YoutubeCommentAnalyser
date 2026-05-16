# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy app source
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Pre-download the model at build time so cold starts are fast
RUN python -c "from transformers import pipeline; \
    pipeline('text-classification', \
    model='cardiffnlp/twitter-xlm-roberta-base-sentiment', \
    top_k=None)"

EXPOSE 8080

CMD ["gunicorn", "app:app", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-"]
