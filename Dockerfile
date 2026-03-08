FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

# Optional: prebuild model/index artifacts at image build time.
# Enable with: --build-arg PREBUILD_INDEX=true
ARG PREBUILD_INDEX=false
RUN if [ "$PREBUILD_INDEX" = "true" ]; then python scripts/build_index.py; fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
