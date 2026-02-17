from python:3.9-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

from base as builder

RUN apt-get update && apt-get install y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# copying pyproject to build wheel
COPY pyproject.toml /app/
RUN pip wheel --no-cache-dir -w / wheels .

# testing stage
from base as tester

COPY --from==builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*  \
    && pip install --no-cache-dir .[dev] \
    && rm -rf /wheels
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests

# docker environment variable
ENV CONFIG_DIR=/app/configs
ENV LOG_DIR=/app/artifacts/logs

CMD ["pytest", "-q"]

# trainer stage
from base as trainer

COPY --from==builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*  \
    && rm -rf /wheels

COPY artifacts /app/artifacts
COPY src /app/src
COPY data /app/data
COPY configs /app/configs

# docker environment variable
ENV CONFIG_DIR=/app/configs
ENV LOG_DIR=/app/artifacts/logs

CMD ["python", "src/cli.py", "process"]
CMD ["python", "src/cli.py", "train"]

# production stage(build backend for model)
from base as production

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from==builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY artifacts /app/artifacts
COPY src/api /app/src/api
COPY configs /app/configs

# docker environment variable
ENV CONFIG_DIR=/app/configs
ENV LOG_DIR=/app/artifacts/logs

USER app

expose 8000

CMD ["uvicorn","src.inference.main:app", "--host","0.0.0.0", "port", "8000"]



