FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY requirements.txt .
RUN pip install --no-deps voyageai && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Update `backend/requirements.txt` (remove torch):
```
fastapi==0.109.0
uvicorn==0.27.0
psycopg2-binary==2.9.9
voyageai==0.2.1
tokenizers==0.15.0
python-dotenv==1.0.0
pydantic==2.5.3