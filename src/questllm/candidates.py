"""Explainable extraction and ranking of answer candidates from processed sentences."""

import re
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum

from nltk import pos_tag
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from questllm.config import (
    CANDIDATE_COVERAGE_PENALTY,
    CANDIDATE_FREQUENCY_BONUS,
    CANDIDATE_LENGTH_PENALTY,
    CANDIDATE_SENTENCE_QUALITY_BONUS,
    CANDIDATE_STOPWORDS,
    DEFAULT_CANDIDATE_LIMIT,
    MAXIMUM_CANDIDATE_CHARACTERS,
    MAXIMUM_CANDIDATE_TOKENS,
    MINIMUM_CANDIDATE_CHARACTERS,
)
from questllm.nltk_resources import ensure_pos_tagger_resources
from questllm.preprocessing import ProcessedSentence


class CandidateType(StrEnum):
    """Small, practical categories that explain why a candidate was selected."""

    PROPER_NOUN = "proper noun"
    NOUN_PHRASE = "noun phrase"
    DATE_YEAR = "date/year"
    QUANTITY_PERCENTAGE = "quantity/percentage"
    DEFINITION_TERM = "definition term"
    GENERAL_CONCEPT = "general concept"


@dataclass(frozen=True, slots=True)
class CandidateAnswer:
    """A possible answer with source provenance for later question generation."""

    text: str
    normalized_text: str
    source_sentence: str
    sentence_index: int
    paragraph_index: int
    page_number: int | None
    importance_score: float
    candidate_type: CandidateType
    frequency: int = 1


_DEFINITION_PATTERN = re.compile(
    r"^\s*(?P<term>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,4}?)\s+"
    r"(?:is\s+defined\s+as|refers\s+to|means|is)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_PATTERN = re.compile(
    rf"\b(?:(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:,\s*|\s+)(?:1[5-9]\d{{2}}|20\d{{2}}|21\d{{2}})|"
    rf"\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+(?:1[5-9]\d{{2}}|20\d{{2}}|21\d{{2}}))\b",
    re.IGNORECASE,
)
_QUANTITY_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:percent(?:age)?|thousand|million|"
    r"billion|kilometers?|kilometres?|meters?|metres?|kilograms?|kg|km)\b",
    re.IGNORECASE,
)
_NORMALIZATION_PATTERN = re.compile(r"[^\w\s%]+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|%")
_NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_NOUN_PHRASE_TAGS = _NOUN_TAGS | frozenset({"JJ", "JJR", "JJS"})
_TYPE_BONUSES = {
    CandidateType.DEFINITION_TERM: 0.25,
    CandidateType.PROPER_NOUN: 0.16,
    CandidateType.DATE_YEAR: 0.12,
    CandidateType.QUANTITY_PERCENTAGE: 0.12,
    CandidateType.NOUN_PHRASE: 0.06,
    CandidateType.GENERAL_CONCEPT: 0.0,
}
_TYPE_PRIORITY = {
    CandidateType.DEFINITION_TERM: 6,
    CandidateType.PROPER_NOUN: 5,
    CandidateType.DATE_YEAR: 4,
    CandidateType.QUANTITY_PERCENTAGE: 4,
    CandidateType.NOUN_PHRASE: 3,
    CandidateType.GENERAL_CONCEPT: 2,
}


def normalize_candidate(text: str) -> str:
    """Create a stable, punctuation-tolerant key for candidate comparison."""

    simplified = _NORMALIZATION_PATTERN.sub(" ", text.lower())
    return " ".join(simplified.split())


def _candidate_tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(text))


def _is_usable_candidate(text: str, sentence_text: str) -> bool:
    """Reject weak fragments while preserving concise factual answers such as DNA and 5%."""

    normalized = normalize_candidate(text)
    tokens = _candidate_tokens(normalized)
    word_tokens = tuple(token for token in tokens if token != "%")
    if not normalized or not word_tokens:
        return False
    if len(normalized) < MINIMUM_CANDIDATE_CHARACTERS:
        return False
    if len(normalized) > MAXIMUM_CANDIDATE_CHARACTERS or len(tokens) > MAXIMUM_CANDIDATE_TOKENS:
        return False
    if all(token.lower() in CANDIDATE_STOPWORDS for token in word_tokens):
        return False
    sentence_normalized = normalize_candidate(sentence_text)
    return not (normalized == sentence_normalized and len(word_tokens) > 3)


def _make_candidate(
    text: str,
    sentence: ProcessedSentence,
    candidate_type: CandidateType,
) -> CandidateAnswer | None:
    cleaned_text = " ".join(text.split()).strip(" ,;:-")
    if not _is_usable_candidate(cleaned_text, sentence.text):
        return None
    return CandidateAnswer(
        text=cleaned_text,
        normalized_text=normalize_candidate(cleaned_text),
        source_sentence=sentence.text,
        sentence_index=sentence.index,
        paragraph_index=sentence.paragraph_index,
        page_number=sentence.page_number,
        importance_score=0.0,
        candidate_type=candidate_type,
    )


def _definition_candidates(sentence: ProcessedSentence) -> list[CandidateAnswer]:
    match = _DEFINITION_PATTERN.match(sentence.text)
    if not match:
        return []
    term = re.sub(r"^(?:the\s+)?term\s+", "", match.group("term"), flags=re.IGNORECASE)
    candidate = _make_candidate(term, sentence, CandidateType.DEFINITION_TERM)
    return [candidate] if candidate else []


def _pattern_candidates(sentence: ProcessedSentence) -> list[CandidateAnswer]:
    candidates = []
    for match in _DATE_PATTERN.finditer(sentence.text):
        candidate = _make_candidate(match.group(), sentence, CandidateType.DATE_YEAR)
        if candidate:
            candidates.append(candidate)
    for match in _YEAR_PATTERN.finditer(sentence.text):
        candidate = _make_candidate(match.group(), sentence, CandidateType.DATE_YEAR)
        if candidate:
            candidates.append(candidate)
    for match in _QUANTITY_PATTERN.finditer(sentence.text):
        candidate = _make_candidate(match.group(), sentence, CandidateType.QUANTITY_PERCENTAGE)
        if candidate:
            candidates.append(candidate)
    return candidates


def _pos_candidates(sentence: ProcessedSentence) -> list[CandidateAnswer]:
    """Extract proper-name sequences and compact noun phrases from one sentence."""

    tagged_tokens = pos_tag(wordpunct_tokenize(sentence.text))
    candidates = []
    proper_sequence: list[str] = []
    phrase_sequence: list[str] = []

    def add_sequence(sequence: list[str], candidate_type: CandidateType) -> None:
        if sequence:
            candidate = _make_candidate(" ".join(sequence), sentence, candidate_type)
            if candidate:
                candidates.append(candidate)

    for token, tag in (*tagged_tokens, ("", "END")):
        if tag in {"NNP", "NNPS"}:
            proper_sequence.append(token)
        else:
            add_sequence(proper_sequence, CandidateType.PROPER_NOUN)
            proper_sequence = []

        if tag in _NOUN_PHRASE_TAGS:
            phrase_sequence.append(token)
        else:
            add_sequence(phrase_sequence, CandidateType.NOUN_PHRASE)
            phrase_sequence = []

        if tag in _NOUN_TAGS:
            candidate = _make_candidate(token, sentence, CandidateType.GENERAL_CONCEPT)
            if candidate:
                candidates.append(candidate)

    return candidates


def extract_candidates(sentences: tuple[ProcessedSentence, ...]) -> tuple[CandidateAnswer, ...]:
    """Extract unranked candidate phrases using transparent patterns and POS tags."""

    if not sentences:
        return ()

    ensure_pos_tagger_resources()
    candidates = []
    for sentence in sentences:
        candidates.extend(_definition_candidates(sentence))
        candidates.extend(_pattern_candidates(sentence))
        candidates.extend(_pos_candidates(sentence))
    return tuple(candidates)


def _sentence_documents(candidates: tuple[CandidateAnswer, ...]) -> tuple[str, ...]:
    """Return distinct source sentences in source order for document-level TF-IDF."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.sentence_index, candidate.source_sentence),
    )
    return tuple(dict.fromkeys(candidate.source_sentence for candidate in ordered))


def _tfidf_scores(candidates: tuple[CandidateAnswer, ...]) -> dict[tuple[int, str], float]:
    """Map each candidate occurrence to its phrase or strongest term TF-IDF score."""

    documents = _sentence_documents(candidates)
    if not documents:
        return {}
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, MAXIMUM_CANDIDATE_TOKENS))
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {}

    features = {feature: index for index, feature in enumerate(vectorizer.get_feature_names_out())}
    rows = {document: index for index, document in enumerate(documents)}
    scores = {}
    for candidate in candidates:
        row = matrix.getrow(rows[candidate.source_sentence])
        phrase = " ".join(
            token for token in _candidate_tokens(candidate.normalized_text) if token != "%"
        )
        feature_terms = [phrase, *phrase.split()]
        scores[(candidate.sentence_index, candidate.normalized_text)] = max(
            (float(row[0, features[term]]) for term in feature_terms if term in features),
            default=0.0,
        )
    return scores


def _score_candidate(candidate: CandidateAnswer, tfidf_score: float, frequency: int) -> float:
    """Combine TF-IDF with small, documented signals that keep ranking explainable."""

    token_count = len(_candidate_tokens(candidate.normalized_text))
    sentence_quality = min(len(candidate.source_sentence) / 180, 1.0)
    score = (
        tfidf_score
        + _TYPE_BONUSES[candidate.candidate_type]
        + CANDIDATE_FREQUENCY_BONUS * min(max(frequency - 1, 0), 3)
        + CANDIDATE_SENTENCE_QUALITY_BONUS * sentence_quality
        - CANDIDATE_LENGTH_PENALTY * max(token_count - 3, 0)
    )
    return round(score, 6)


def deduplicate_candidates(candidates: tuple[CandidateAnswer, ...]) -> tuple[CandidateAnswer, ...]:
    """Keep one strongest occurrence per normalized phrase and retain its frequency signal."""

    grouped: dict[str, list[CandidateAnswer]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.normalized_text, []).append(candidate)

    unique = []
    for occurrences in grouped.values():
        frequency = len(occurrences)
        best = min(
            occurrences,
            key=lambda candidate: (
                -candidate.importance_score,
                -_TYPE_PRIORITY[candidate.candidate_type],
                candidate.sentence_index,
                candidate.text.lower(),
            ),
        )
        unique.append(replace(best, frequency=frequency))
    return tuple(unique)


def rank_candidates(candidates: tuple[CandidateAnswer, ...]) -> tuple[CandidateAnswer, ...]:
    """Apply deterministic TF-IDF and heuristic scoring, then remove duplicate occurrences."""

    if not candidates:
        return ()

    frequencies = Counter(candidate.normalized_text for candidate in candidates)
    tfidf_scores = _tfidf_scores(candidates)
    scored = tuple(
        replace(
            candidate,
            importance_score=_score_candidate(
                candidate,
                tfidf_scores.get((candidate.sentence_index, candidate.normalized_text), 0.0),
                frequencies[candidate.normalized_text],
            ),
        )
        for candidate in candidates
    )
    deduplicated = deduplicate_candidates(scored)
    return tuple(
        sorted(
            deduplicated,
            key=lambda candidate: (
                -candidate.importance_score,
                -_TYPE_PRIORITY[candidate.candidate_type],
                candidate.normalized_text,
                candidate.sentence_index,
            ),
        )
    )


def select_diverse_candidates(
    candidates: tuple[CandidateAnswer, ...],
    *,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> tuple[CandidateAnswer, ...]:
    """Return top candidates while mildly favoring paragraphs not yet represented."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    remaining = list(candidates)
    selected = []
    coverage: Counter[tuple[int | None, int]] = Counter()
    while remaining and len(selected) < limit:
        def coverage_key(candidate: CandidateAnswer) -> tuple[float, float, str, int]:
            source_key = (candidate.page_number, candidate.paragraph_index)
            adjusted_score = candidate.importance_score - (
                CANDIDATE_COVERAGE_PENALTY * coverage[source_key]
            )
            return (
                -adjusted_score,
                -candidate.importance_score,
                candidate.normalized_text,
                candidate.sentence_index,
            )

        next_candidate = min(remaining, key=coverage_key)
        remaining.remove(next_candidate)
        selected.append(next_candidate)
        coverage[(next_candidate.page_number, next_candidate.paragraph_index)] += 1
    return tuple(selected)


def extract_and_rank_candidates(
    sentences: tuple[ProcessedSentence, ...],
    *,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> tuple[CandidateAnswer, ...]:
    """Extract, rank, deduplicate, and select a diverse candidate set for the UI or later models."""

    return select_diverse_candidates(rank_candidates(extract_candidates(sentences)), limit=limit)
