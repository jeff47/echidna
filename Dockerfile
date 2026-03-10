FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -m -u 10001 echidna

COPY pyproject.toml README.md /app/
COPY app /app/app
COPY affiliation_normalizer /app/affiliation_normalizer
COPY templates /app/templates
COPY echidna.jpg /app/echidna.jpg
COPY favicon.ico /app/favicon.ico

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /app/.runtime/runs && chown -R echidna:echidna /app
USER echidna

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
