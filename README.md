# QuestLLM

QuestLLM is a local-first study tool that turns pasted text, text-based PDFs, or a selected
Wikipedia article into grounded practice quizzes. It generates several question formats, lets the
learner attempt a quiz without seeing answers, and then provides scoring, review, and source
provenance.

## Features

- Text-to-quiz, PDF-to-quiz, and Wikipedia-backed topic-to-quiz
- Multiple Choice, True/False, Fill-in-the-Blank, and Short Answer questions
- Question-type, count, and heuristic difficulty selection
- Local Hugging Face T5 question generation for stem-based question types
- NLTK and TF-IDF candidate extraction, validation, and duplicate filtering
- Stable quiz attempts, conservative scoring, answer review, and source references
- Local PDF/text processing and explicit Wikipedia article selection for topic mode

## Architecture

```text
Input
  → Ingestion
  → Preprocessing
  → Chunking
  → Candidate Extraction
  → T5 Question Generation
  → Question Builders
  → Validation & Deduplication
  → Quiz
  → Scoring & Review
```

## Tech stack

- Python, Streamlit, pytest, and Ruff
- Hugging Face Transformers, T5, PyTorch, and SentencePiece
- NLTK and scikit-learn
- pypdf for text-based PDFs
- Requests and the MediaWiki API for topic sources

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m questllm.nltk_resources --download
python -m questllm.nltk_resources --status
python -m streamlit run app.py
```

The NLTK setup command stores `punkt`, `punkt_tab`, and
`averaged_perceptron_tagger_eng` in the active virtual environment. This explicit setup remains the
recommended local workflow. If any resource is missing when Streamlit starts, the app downloads it
once and caches the successful initialization for the lifetime of that app process.

## First model download

On the first quiz generation that needs T5, QuestLLM downloads
[`valhalla/t5-base-qa-qg-hl`](https://huggingface.co/valhalla/t5-base-qa-qg-hl) into the local
Hugging Face cache. The model is large enough that the first run may take time depending on network
speed; later runs reuse the cache. CUDA is used when available, otherwise generation runs on CPU.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Create a new app.
5. Select:
   - repository: `Himvni16/QuestLLM`
   - branch: `main`
   - main file: `app.py`
6. In Advanced settings, select Python `3.11` to match the project's tested local environment.
   No secrets are required.
7. Deploy.

The first T5 model load downloads an approximately 900 MB checkpoint and may take several minutes.
Later requests in the same running app process reuse Streamlit's cached model and Hugging Face's
on-disk cache. Topic mode needs outbound internet access for Wikipedia. Free-tier memory and CPU
can limit very large documents or quizzes, so QuestLLM caps pasted/article text at 50,000
characters, PDFs at 10 MB and 50 pages, and quizzes at 12 questions.

## Testing and smoke checks

The offline suite is deterministic and does not download a model or contact Wikipedia.

```powershell
python -m ruff check .
python -m pytest
python -m pip check
python -m compileall -q app.py src tests
```

The offline pytest suite covers the application and deployment helpers. Optional live checks are
separate:

```powershell
python scripts/smoke_test_model.py
python scripts/smoke_test_topic.py Photosynthesis
```

## Privacy and networking

- Pasted text and uploaded PDFs are processed locally after dependencies are installed.
- Topic mode sends the entered query to Wikipedia/MediaWiki and retrieves the article the learner
  selects. The resulting quiz keeps the article title and URL as provenance.
- Hugging Face internet access is needed only when the T5 model is not already cached.
- QuestLLM does not require API keys or other secrets. Wikipedia/MediaWiki access is
  unauthenticated.

## Limitations

- This is an English-focused MVP.
- Scanned/image-only PDFs require OCR and are intentionally unsupported.
- CPU generation can be slow.
- Difficulty is heuristic, and short-answer grading is deliberately conservative.
- Wikipedia topic mode requires internet access.
- Quiz quality depends on the clarity and factual quality of the source material.

## Project structure

```text
app.py                         Streamlit workflow and UI
src/questllm/
  ingestion/                   Text, PDF, and MediaWiki source adapters
  preprocessing.py             Sentence preparation and provenance
  chunking.py                  Context-window chunking
  candidates.py                NLTK/TF-IDF candidate ranking
  generation.py                T5 stem generation
  question_builders.py         Question-format construction
  validation.py                Quality and duplicate checks
  quiz_assembly.py             Bounded quiz assembly
  workflow.py / scoring.py     Attempt state and grading
  nltk_resources.py            Explicit environment-local NLTK setup
tests/                         Offline, deterministic test suite
scripts/                       Optional model and topic smoke checks
```

## Future improvements

- OCR for scanned PDFs
- Multilingual ingestion and generation
- Stronger semantic distractors
- Quiz export formats
- Persistent quiz history
