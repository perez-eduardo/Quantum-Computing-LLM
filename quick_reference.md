# Quick Reference: Project Setup

## Local Machine (Windows PowerShell)

```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
```

No virtual environment needed for local (just running scp commands).

---

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
