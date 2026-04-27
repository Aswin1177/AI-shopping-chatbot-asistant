FROM python:3.10-slim

WORKDIR /app

# Avoid pip warnings and caching issues
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY . .

# Upgrade pip first
RUN pip install --upgrade pip

# Install heavy dependencies separately (reduces timeout failures)
RUN pip install --no-cache-dir --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

RUN pip install --no-cache-dir --default-timeout=1000 \
    sentence-transformers==2.6.1 faiss-cpu

# Install remaining lighter dependencies
RUN pip install --no-cache-dir --default-timeout=1000 \
    fastapi uvicorn python-dotenv pandas numpy google-generativeai

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]