"""Benchmark ground truth dataset."""

BENCHMARK: list[dict] = [
    {
        "query": "Які підстави для відрахування студента?",
        "expected_section_id": 4,
        "relevant_chunk_ids": [],
        "relevant_page_ids": [],
    },
    {
        "query": "Як отримати академічну відпустку?",
        "expected_section_id": 4,
        "relevant_chunk_ids": [],
        "relevant_page_ids": [],
    },
    {
        "query": "Що таке академічна доброчесність?",
        "expected_section_id": 1,
        "relevant_chunk_ids": [],
        "relevant_page_ids": [],
    },
    {
        "query": "Як відбувається переведення на іншу спеціальність?",
        "expected_section_id": 4,
        "relevant_chunk_ids": [],
        "relevant_page_ids": [],
    },
    {
        "query": "Хто має право на соціальну стипендію?",
        "expected_section_id": 5,
        "relevant_chunk_ids": [],
        "relevant_page_ids": [],
    }
]
