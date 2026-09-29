# QuestLLM

QuestLLM is a local-first quiz generation application built with Python and Streamlit.

Phase 7 supports pasted text and text-based PDF ingestion, sentence preprocessing,
candidate-answer ranking, local T5 question stems, quality validation, duplicate removal, and
assembled quiz previews. Quiz-taking and scoring will be added incrementally in later phases.

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

The quiz preview intentionally shows correct answers and source excerpts for development
verification. QuestLLM returns a smaller quiz with a warning when it cannot safely satisfy the
requested count; it never duplicates questions or forces weak MCQ distractors.

## Quality checks

```powershell
python -m ruff check .
python -m pytest
```
