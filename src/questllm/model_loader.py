"""Lazy, local loading for QuestLLM's Hugging Face question-generation model."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from questllm.config import QUESTION_GENERATION_MODEL_ID
from questllm.exceptions import ModelLoadError


@dataclass(frozen=True, slots=True)
class LoadedQuestionGenerationModel:
    """The local model resources required for deterministic question generation."""

    tokenizer: object
    model: object
    device: str
    model_id: str


def select_device() -> str:
    """Prefer CUDA when it is available, while keeping CPU inference fully supported."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_question_generation_model(
    model_id: str = QUESTION_GENERATION_MODEL_ID,
) -> LoadedQuestionGenerationModel:
    """Load the configured model only when a caller explicitly requests generation."""

    device = select_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError) as error:
        message = str(error).lower()
        if "out of memory" in message:
            detail = "QuestLLM ran out of memory while loading the question-generation model."
        elif "connection" in message or "network" in message or "download" in message:
            detail = (
                "QuestLLM could not download the question-generation model. Check your connection."
            )
        else:
            detail = "QuestLLM could not load the local question-generation model or tokenizer."
        raise ModelLoadError(f"{detail} Details: {error}") from error

    return LoadedQuestionGenerationModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_id=model_id,
    )
