FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev curl && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-deps voyageai && pip install -r requirements.txt

COPY backend/ ./backend/
COPY training/scripts/ ./training/scripts/
COPY training/tokenizer/ ./training/tokenizer/

RUN mkdir -p /app/training/model && \
    curl -L -o /app/training/model/final_model.pt \
    "https://github.com/perez-eduardo/Quantum-Computing-LLM/raw/main/training/model/final_model.pt"

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "-c", "import os; import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]