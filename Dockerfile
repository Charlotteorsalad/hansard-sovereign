# FastAPI backend for Hansard Sovereign.
#
# Runs CPU-only on purpose: the query embedder loads on CPU and text generation
# happens in Ollama (a separate process/host), so the API never needs a GPU. We
# install the CPU build of torch first so the image doesn't pull multi-GB CUDA
# wheels it would never use.
FROM python:3.12-slim

# build-essential covers the few deps without prebuilt wheels (e.g. parts of the
# chromadb / sentence-transformers stack).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch up front; later installs then see torch as already satisfied
# and won't drag in the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the myhansard package (library/) + its pinned deps.
COPY pyproject.toml README.md ./
COPY library ./library
RUN pip install --no-cache-dir .

# The ASGI app lives in scripts/api.py and is imported as the namespace package
# `scripts.api`; data/ (sqlite + chroma) is mounted at runtime, not baked in.
COPY scripts ./scripts

# No GPU in the container — keep torch from probing CUDA at all.
ENV CUDA_VISIBLE_DEVICES=""
# Reach Ollama on the Docker host by default (overridable via compose).
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"

EXPOSE 8000
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
