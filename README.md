# QuestLLM

QuestLLM is a local-first quiz generation application built with Python and Streamlit.

Phase 5 supports pasted text and text-based PDF ingestion, sentence preprocessing,
sentence-aware chunking, candidate-answer ranking, and local T5 question-stem previews. OCR,
multiple-choice construction, quiz-taking, and scoring will be added incrementally in later phases.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Before processing text for the first time, download NLTK's sentence-tokenizer and POS-tagger data
explicitly:

```powershell
python -m questllm.nltk_resources --download
```

The first **Generate Question Preview** request downloads the configured Hugging Face model
(`valhalla/t5-base-qa-qg-hl`) if it is not already cached. It runs locally on CUDA when available,
otherwise on CPU. To explicitly test that model outside the UI (this may download model files):

```powershell
python scripts/smoke_test_model.py
```

## Quality checks

```powershell
python -m ruff check .
python -m pytest
```
