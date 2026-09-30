"""Offline unit tests for candidate extraction, ranking, and source coverage."""

import pytest

from questllm import candidates
from questllm.candidates import (
    CandidateAnswer,
    CandidateType,
    deduplicate_candidates,
    extract_and_rank_candidates,
    extract_candidates,
    rank_candidates,
    select_diverse_candidates,
)
from questllm.exceptions import NltkResourceError
from questllm.preprocessing import ProcessedSentence


def _sentence(
    index: int,
    text: str,
    *,
    paragraph_index: int = 0,
    page_number: int | None = None,
) -> ProcessedSentence:
    return ProcessedSentence(
        index=index,
        text=text,
        paragraph_index=paragraph_index,
        page_number=page_number,
    )


@pytest.fixture
def patch_pos_tagger(monkeypatch: pytest.MonkeyPatch) -> None:
    tags = {
        "Albert": "NNP",
        "Einstein": "NNP",
        "artificial": "JJ",
        "intelligence": "NN",
        "photosynthesis": "NN",
        "chlorophyll": "NN",
        "DNA": "NNP",
        "AI": "NNP",
        "important": "JJ",
        "research": "NN",
        "system": "NN",
        "concept": "NN",
        "weak": "JJ",
        "term": "NN",
    }

    def fake_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
        return [(token, tags.get(token, "VBZ")) for token in tokens]

    monkeypatch.setattr(candidates, "ensure_pos_tagger_resources", lambda: None)
    monkeypatch.setattr(candidates, "pos_tag", fake_pos_tag)


def test_extracts_proper_nouns_noun_phrases_patterns_and_provenance(
    patch_pos_tagger: None,
) -> None:
    sentences = (
        _sentence(
            0,
            "Albert Einstein developed artificial intelligence research in 1947.",
            page_number=2,
        ),
        _sentence(
            1,
            "Photosynthesis is defined as a process that uses chlorophyll and 75 percent light.",
            page_number=3,
        ),
        _sentence(2, "The sample contained 5% oxygen.", page_number=3),
        _sentence(3, "The experiment began on March 15, 1947.", page_number=3),
    )

    extracted = extract_candidates(sentences)
    by_text = {(candidate.text, candidate.candidate_type): candidate for candidate in extracted}

    assert ("Albert Einstein", CandidateType.PROPER_NOUN) in by_text
    assert ("artificial intelligence research", CandidateType.NOUN_PHRASE) in by_text
    assert ("1947", CandidateType.DATE_YEAR) in by_text
    assert ("Photosynthesis", CandidateType.DEFINITION_TERM) in by_text
    assert ("75 percent", CandidateType.QUANTITY_PERCENTAGE) in by_text
    assert ("5%", CandidateType.QUANTITY_PERCENTAGE) in by_text
    assert ("March 15, 1947", CandidateType.DATE_YEAR) in by_text
    assert by_text[("Albert Einstein", CandidateType.PROPER_NOUN)].page_number == 2
    assert (
        by_text[("Photosynthesis", CandidateType.DEFINITION_TERM)].source_sentence
        == sentences[1].text
    )
    assert by_text[("Photosynthesis", CandidateType.DEFINITION_TERM)].sentence_index == 1
    assert by_text[("Photosynthesis", CandidateType.DEFINITION_TERM)].paragraph_index == 0


def test_keeps_short_technical_terms_and_rejects_pronouns_stopwords_and_long_phrases(
    patch_pos_tagger: None,
) -> None:
    sentences = (
        _sentence(0, "DNA and AI are important concepts."),
        _sentence(1, "It is the system that they use."),
        _sentence(2, "Artificial intelligence research system concept weak term is useful."),
    )

    extracted = extract_candidates(sentences)
    normalized = {candidate.normalized_text for candidate in extracted}

    assert "dna" in normalized
    assert "ai" in normalized
    assert "it" not in normalized
    assert "they" not in normalized
    assert "the" not in normalized
    assert "artificial intelligence research system concept weak term" not in normalized


def test_deduplication_normalizes_casing_whitespace_and_retains_frequency() -> None:
    candidates_to_deduplicate = (
        CandidateAnswer(
            text="Artificial Intelligence",
            normalized_text="artificial intelligence",
            source_sentence="Artificial Intelligence supports research.",
            sentence_index=0,
            paragraph_index=0,
            page_number=1,
            importance_score=0.2,
            candidate_type=CandidateType.NOUN_PHRASE,
        ),
        CandidateAnswer(
            text="artificial   intelligence!",
            normalized_text="artificial intelligence",
            source_sentence="artificial intelligence is widely studied.",
            sentence_index=1,
            paragraph_index=1,
            page_number=2,
            importance_score=0.4,
            candidate_type=CandidateType.DEFINITION_TERM,
        ),
    )

    deduplicated = deduplicate_candidates(candidates_to_deduplicate)

    assert len(deduplicated) == 1
    assert deduplicated[0].text == "artificial   intelligence!"
    assert deduplicated[0].frequency == 2
    assert deduplicated[0].page_number == 2


def test_ranking_rewards_definition_terms_and_repeated_meaningful_candidates(
    patch_pos_tagger: None,
) -> None:
    sentences = (
        _sentence(0, "Photosynthesis is defined as a process using chlorophyll."),
        _sentence(1, "Chlorophyll helps photosynthesis capture light."),
        _sentence(2, "A weak term appears once."),
    )

    ranked = rank_candidates(extract_candidates(sentences))
    ranking = {candidate.normalized_text: candidate for candidate in ranked}

    assert ranking["photosynthesis"].candidate_type == CandidateType.DEFINITION_TERM
    assert ranking["photosynthesis"].frequency >= 2
    assert ranking["photosynthesis"].importance_score > ranking["weak term"].importance_score
    assert ranked == rank_candidates(extract_candidates(sentences))


def test_diverse_selection_favors_another_paragraph_when_scores_are_close() -> None:
    ranked = (
        CandidateAnswer(
            "alpha", "alpha", "Alpha is useful.", 0, 0, 1, 0.90, CandidateType.NOUN_PHRASE
        ),
        CandidateAnswer(
            "beta", "beta", "Beta is useful.", 1, 0, 1, 0.88, CandidateType.NOUN_PHRASE
        ),
        CandidateAnswer(
            "gamma", "gamma", "Gamma is useful.", 2, 1, 2, 0.84, CandidateType.NOUN_PHRASE
        ),
    )

    selected = select_diverse_candidates(ranked, limit=2)

    assert [candidate.text for candidate in selected] == ["alpha", "gamma"]


def test_empty_pool_and_missing_nltk_resource_are_handled_offline(
    patch_pos_tagger: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert extract_candidates(()) == ()
    assert extract_and_rank_candidates(()) == ()

    def raise_missing_resource() -> None:
        raise NltkResourceError("NLTK POS tagger data is missing")

    monkeypatch.setattr(candidates, "ensure_pos_tagger_resources", raise_missing_resource)
    with pytest.raises(NltkResourceError, match="POS tagger"):
        extract_candidates((_sentence(0, "DNA is useful."),))


def test_relation_candidates_prefer_complete_facts_across_domains(
    patch_pos_tagger: None,
) -> None:
    sentences = (
        _sentence(0, "Antibiotics are used to treat bacterial infections."),
        _sentence(1, "Photosynthesis converts light energy into chemical energy."),
        _sentence(2, "DNA stores genetic information."),
    )

    extracted = extract_candidates(sentences)
    relation_candidates = {
        candidate.text: candidate
        for candidate in extracted
        if candidate.candidate_type is CandidateType.RELATION_PHRASE
    }

    assert relation_candidates["bacterial infections"].subject == "Antibiotics"
    assert relation_candidates["bacterial infections"].relation == "are"
    assert relation_candidates["chemical energy"].subject == "Photosynthesis"
    assert relation_candidates["chemical energy"].relation == "converts"
    assert relation_candidates["genetic information"].subject == "DNA"
    assert relation_candidates["genetic information"].relation == "stores"


def test_supervised_learning_regression_keeps_task_phrase_not_embedded_spam(
    patch_pos_tagger: None,
) -> None:
    sentences = (
        _sentence(
            0,
            "Supervised Learning: The computer trains on labeled data that includes "
            "correct answers.",
        ),
        _sentence(1, "It learns to predict outcomes for new data."),
        _sentence(2, "A common task is classifying emails as spam or not spam."),
    )

    extracted = extract_candidates(sentences)
    normalized = {candidate.normalized_text for candidate in extracted}
    task = next(
        candidate
        for candidate in extracted
        if candidate.text == "classifying emails as spam or not spam"
    )

    assert task.candidate_type is CandidateType.RELATION_PHRASE
    assert task.subject == "Supervised Learning"
    assert "spam" not in normalized


@pytest.mark.parametrize(
    ("framing", "answer"),
    (
        ("The goal is producing clean electricity", "producing clean electricity"),
        ("An example is a solar panel", "a solar panel"),
    ),
)
def test_generic_relation_framings_inherit_only_the_local_topic_subject(
    patch_pos_tagger: None,
    framing: str,
    answer: str,
) -> None:
    sentences = (
        _sentence(0, "Renewable Energy: It reduces dependence on fossil fuels."),
        _sentence(1, f"{framing}."),
        _sentence(2, "Unrelated Topic: It has a separate purpose.", paragraph_index=1),
    )

    candidate = next(
        candidate for candidate in extract_candidates(sentences) if candidate.text == answer
    )

    assert candidate.subject == "Renewable Energy"
