"""Optional live MediaWiki ingestion smoke test; never run by the offline test suite."""

import argparse

from questllm.candidates import extract_and_rank_candidates
from questllm.chunking import chunk_sentences
from questllm.exceptions import NltkResourceError
from questllm.ingestion.topic import WikipediaClient
from questllm.preprocessing import preprocess_document


def main() -> None:
    """Retrieve a real topic and verify it reaches QuestLLM's local preparation pipeline."""

    parser = argparse.ArgumentParser(description="Run an optional live QuestLLM topic smoke test.")
    parser.add_argument("topic", nargs="?", default="Photosynthesis")
    args = parser.parse_args()

    client = WikipediaClient()
    results = client.search(args.topic)
    selected = next(
        (result for result in results if result.title.casefold() == args.topic.casefold()),
        results[0],
    )
    document = client.retrieve_article(selected, requested_topic=args.topic)
    print(f"Article: {document.metadata['article_title']}")
    print(f"Selected from {len(results)} search results: {selected.title}")
    print(f"URL: {document.metadata['article_url']}")
    print(f"Characters: {document.character_count}")
    try:
        sentences = preprocess_document(document)
        chunks = chunk_sentences(sentences)
        candidates = extract_and_rank_candidates(sentences)
    except NltkResourceError as error:
        raise SystemExit(
            f"Article retrieval succeeded, but local NLTK data is missing. {error}"
        ) from error
    print(f"Sentences: {len(sentences)} · Chunks: {len(chunks)} · Candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
