FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev git git-lfs && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-deps voyageai && pip install -r requirements.txt

COPY backend/ ./backend/
COPY training/scripts/ ./training/scripts/
COPY training/tokenizer/ ./training/tokenizer/

# Download model using Git LFS
RUN mkdir -p /app/training/model && \
    git lfs install && \
    git clone --depth 1 https://github.com/perez-eduardo/Quantum-Computing-LLM.git /tmp/repo && \
    cd /tmp/repo && \
    git lfs pull --include="training/model/final_model.pt" && \
    cp /tmp/repo/training/model/final_model.pt /app/training/model/ && \
    rm -rf /tmp/repo

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "-c", "import os; import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]