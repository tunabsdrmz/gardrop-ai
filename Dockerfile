FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 HF_HOME=/app/.cache

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

COPY app ./app

# Download CLIP during the build, so the first request isn't slow.
RUN uv run --no-dev python -c "from app.services.embedder import get_embedder; get_embedder().embed_texts(['warm up'])"

# Hugging Face Spaces expects port 7860.
EXPOSE 7860
CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
