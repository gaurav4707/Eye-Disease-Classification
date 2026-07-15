# ─────────────────────────────────────────────────────────────────────────────
# OcuScan AI — Dockerfile
# Phase 4 | v1.0.0
# Base: python:3.10-slim (Debian Bullseye)
# Exposed: port 8501 (Streamlit default)
# Volumes: /app/models (checkpoints), /app/ocuscan.db (SQLite)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app
ENV PYTHONPATH="/app/src:/app"

# ── Copy requirements first (Docker layer cache optimisation) ─────────────────
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY src/ ./src/
COPY app/ ./app/

# labels.csv — copy only if present, using RUN + shell test
COPY dataset/ ./dataset_src/
RUN mkdir -p ./dataset && \
    if [ -f ./dataset_src/labels.csv ]; then cp ./dataset_src/labels.csv ./dataset/labels.csv; fi && \
    rm -rf ./dataset_src

# ── Create required directories ────────────────────────────────────────────────
RUN mkdir -p models results/gradcam dataset/normal dataset/ocp \
    dataset/ocp_chronic dataset/post_viral_ded dataset/sjs dataset/symblepharon

# streamlit config — always write default, then overwrite if a custom one exists in the build context
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml
COPY docker/streamlit_config.tom[l] /root/.streamlit/

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Add src to PYTHONPATH so imports work without sys.path hacks
ENV PYTHONPATH="/app/src:/app:${PYTHONPATH}"

# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# Volumes expected at runtime:
#   -v $(pwd)/models:/app/models           (model checkpoints)
#   -v $(pwd)/ocuscan.db:/app/ocuscan.db   (SQLite persistence)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
