# Quick Reference: Project Setup

## Local Machine (Windows PowerShell)

```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
```

.\venv\Scripts\activate

---


## backend
cd E:\Personal_projects\Quantum-Computing-LLM
docker build -f backend/Dockerfile -t quantum-backend .
docker run -p 8000:8000 --env-file .env -e PORT=8000 quantum-backend


## frontend
cd E:\Personal_projects\Quantum-Computing-LLM
pip install flask requests
cd frontend
set BACKEND_URL=http://localhost:8000
python app.py


## HPC (SSH)

### Connect
```powershell
ssh pereze4@submit-b.hpc.engr.oregonstate.edu
```

### Navigate and Activate
```bash
cd ~/hpc-share/quantum-llm
source venv/bin/activate
```

### One-liner
```bash
cd ~/hpc-share/quantum-llm && source venv/bin/activate
```



### Modal


python -m venv venv
modal deploy inference.py
modal app logs quantum-llm
curl https://perez-eduardo--quantum-llm-health.modal.run

Invoke-WebRequest -Uri "https://perez-eduardo--quantum-llm-query.modal.run" -Method POST -ContentType "application/json" -Body '{"context": "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously.", "question": "What is a qubit?"}'

### Voyage api
pa-kj1_wQwgbrPmPTLd8YFJ9x8XUB8CUv0NNNCDLfiawKl