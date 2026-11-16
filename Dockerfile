FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY predict.py model_2025.bin ./

EXPOSE 9696

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]