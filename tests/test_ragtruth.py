from __future__ import annotations

import json

from rag_support_assistant import ragtruth
from rag_support_assistant.validation import load_ragtruth_subset


def test_committed_subset_matches_manifest_and_is_balanced():
    items = load_ragtruth_subset()  # raises if the hash does not match
    assert len(items) == 200
    assert sum(not i.grounded for i in items) == 100
    assert all(bool(i.spans) != i.grounded for i in items)


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_selection_filters_and_is_seeded(tmp_path):
    sources = [
        {"source_id": "1", "task_type": "QA", "source_info": {"question": "q?", "passages": "p"}},
        {"source_id": "2", "task_type": "Summary", "source_info": "text"},
    ]
    responses = []
    for i in range(12):
        responses.append(
            {"id": str(i), "source_id": "1" if i < 10 else "2", "model": "m", "split": "test",
             "quality": "good" if i != 3 else "truncated", "response": f"r{i}",
             "labels": [{"text": "x", "label_type": "Evident Conflict"}] if i % 2 else []}
        )  # fmt: skip
    _write(tmp_path / "source_info.jsonl", sources)
    _write(tmp_path / "response.jsonl", responses)
    a = ragtruth.select(tmp_path, seed=1, n_per_class=2)
    b = ragtruth.select(tmp_path, seed=1, n_per_class=2)
    assert a == b
    assert {i.response_id for i in a} <= {str(i) for i in range(10) if i != 3}
    assert sum(i.grounded for i in a) == 2
