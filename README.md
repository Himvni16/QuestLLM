# QuestLLM

QuestLLM is a local-first quiz generation application built with Python and Streamlit.

Phase 1 provides the project scaffold and a minimal landing page. Text processing, PDF
extraction, model loading, and quiz generation will be added incrementally in later phases.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

## Quality checks

```powershell
python -m ruff check .
python -m pytest
```
