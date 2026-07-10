# ── Stage 1: Dependency Builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install Python dependencies into a staging location
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Lean Runtime Image ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder (avoids re-downloading)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code only (no .env — secrets injected at runtime)
COPY app/ ./app/
COPY requirements.txt .

# Create the logs directory expected by the logger
RUN mkdir -p logs

# Document the exposed port (actual binding via $PORT from host)
EXPOSE 8000

# ── Startup ───────────────────────────────────────────────────────────────────
# PORT is provided automatically by Render, Railway, and Fly.io.
# --workers 2 is suitable for a 512 MB RAM instance; scale up for larger plans.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
