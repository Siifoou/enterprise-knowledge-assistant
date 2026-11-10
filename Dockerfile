FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

# CPU-only PyTorch. On Linux a plain "pip install torch" pulls the GPU (CUDA) build,
# several GB of libraries that are useless here.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else, at the versions tested on the development machine
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# The application code (see .dockerignore for what is left out)
COPY . .

EXPOSE 8000 8501

# Default command = the API. The compose file overrides it for the Streamlit service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]