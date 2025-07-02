#!/bin/bash

# Start the FastAPI application with Gunicorn for better production performance
exec gunicorn src.api.BestModelApi:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --log-level info \
    --access-logfile - \
    --error-logfile -
