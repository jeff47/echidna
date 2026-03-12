FROM python:3.12-slim

ARG ECHIDNA_BUILD_DATETIME=unknown
ARG ECHIDNA_APP_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ECHIDNA_BUILD_DATETIME=${ECHIDNA_BUILD_DATETIME} \
    ECHIDNA_APP_COMMIT=${ECHIDNA_APP_COMMIT}

WORKDIR /app

RUN useradd -m -u 10001 echidna
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY app /app/app
COPY templates /app/templates
COPY echidna.jpg /app/echidna.jpg
COPY favicon.ico /app/favicon.ico

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /app/.runtime/runs && chown -R echidna:echidna /app
USER echidna

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
