FROM python:3.13.6-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system .

COPY app/ app/
COPY config.toml .
COPY prompts/ prompts/
COPY scripts/ scripts/

EXPOSE 7860

CMD ["python", "-m", "app.app"]