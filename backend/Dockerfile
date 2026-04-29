FROM python:3.10-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY . .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

RUN pip install --no-cache-dir --default-timeout=1000 \
    sentence-transformers==2.6.1 faiss-cpu \
    fastapi uvicorn python-dotenv pandas google-generativeai
    
RUN python -c "import numpy; print('FINAL NUMPY VERSION:', __import__('numpy').__version__)"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]