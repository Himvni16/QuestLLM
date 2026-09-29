# QuestLLM

QuestLLM is a local-first quiz generation application built with Python and Streamlit.

Phase 3 supports pasted text and text-based PDF ingestion, sentence preprocessing, and
sentence-aware chunking. OCR, candidate extraction, model loading, and quiz generation will be
added incrementally in later phases.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Before processing text for the first time, download NLTK's sentence-tokenizer data explicitly:

```powershell
python -m questllm.nltk_resources --download
```

## Quality checks

```powershell
python -m ruff check .
python -m pytest
```
