"""Explicit online smoke test for QuestLLM's configured local Hugging Face model."""

from questllm.candidates import CandidateAnswer, CandidateType
from questllm.exceptions import QuestLLMError
from questllm.generation import FocusedContext, QuestionGenerationSettings, T5QuestionGenerator
from questllm.model_loader import load_question_generation_model


def main() -> int:
    """Load the model on demand and generate one known answer-aware question stem."""

    candidate = CandidateAnswer(
        text="42",
        normalized_text="42",
        source_sentence="42 is the answer to life, the universe, and everything.",
        sentence_index=0,
        paragraph_index=0,
        page_number=None,
        importance_score=1.0,
        candidate_type=CandidateType.GENERAL_CONCEPT,
    )
    context = FocusedContext(
        text=candidate.source_sentence,
        source_sentence=candidate.source_sentence,
        sentence_index=0,
        page_number=None,
    )
    try:
        generator = T5QuestionGenerator(load_question_generation_model())
        result = generator.generate(candidate, context, QuestionGenerationSettings(preview_limit=1))
    except QuestLLMError as error:
        print(f"Model smoke test failed: {error}")
        return 1

    print(f"Device: {generator.device}")
    print(f"Question: {result.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
