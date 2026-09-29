from __future__ import annotations

import json

from rag_support_assistant.grid import Grid, _best, load_split_records


def test_best_prefers_the_earlier_candidate_on_ties():
    assert _best({"a": 0.5, "b": 0.5, "c": 0.4}) == "a"
    assert _best({"a": 0.5, "b": 0.6}) == "b"


def test_grid_runs_end_to_end_with_fakes(fake_ctx, questions, tmp_path):
    dev = [q for q in questions if q.split == "dev"][:12]
    test = [q for q in questions if q.split == "test"][:12]
    grid = Grid(fake_ctx, tmp_path, {"dev": dev, "test": test})
    run = grid.run_all(strong=True)
    assert [s["step"] for s in run.steps] == ["chunking", "embedding", "first stage", "reranker"]
    assert set(run.alpha_sweep) == {f"{a / 10:.1f}" for a in range(11)}
    assert len(run.generation_configs) == 3
    assert not set(run.generation_configs) & {
        s for s, c in run.configs.items() if c["role"] == "strong_arm"
    }
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["steps"] == run.steps
    records = load_split_records(tmp_path, "test")
    assert set(records) == set(run.configs)
    for recs in records.values():
        assert {r.item_id for r in recs} == {q.question_id for q in test if q.gold_article_ids}
        assert all(r.cluster for r in recs)
