# Quick Reference: Project Setup

## Local Machine (Windows PowerShell)

```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
```

.\venv\Scripts\activate

---


## backend
cd E:\Personal_projects\Quantum-Computing-LLM
.\venv\Scripts\Activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## frontend
cd E:\Personal_projects\Quantum-Computing-LLM
.\venv\Scripts\Activate
cd frontend
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




### Voyage api
pa-kj1_wQwgbrPmPTLd8YFJ9x8XUB8CUv0NNNCDLfiawKl