from __future__ import annotations

import shutil

import pytest

from rag_support_assistant.data import REPO_ROOT
from rag_support_assistant.readme import PENDING, render, render_cost

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "results" / "retrieval" / "selection.json").exists(),
    reason="committed retrieval results are missing",
)


def test_render_uses_committed_results():
    text = render()
    assert "**Retrieval, test split**" in text
    assert "nDCG@10" in text


def test_render_marks_live_rows_pending_without_live_results(tmp_path):
    # Only the offline retrieval results: every live row reads pending.
    shutil.copytree(REPO_ROOT / "results" / "retrieval", tmp_path / "retrieval")
    text = render(tmp_path)
    assert "nDCG@10" in text
    assert PENDING in text
    assert "have not been made yet" in text


def test_readme_section_is_current():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    body = readme.split(start)[1].split(end)[0].strip()
    assert body == render().strip(), "README results are stale: run `make demo`"


def test_cost_section_is_current():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    body = readme.split("<!-- cost:start -->")[1].split("<!-- cost:end -->")[0].strip()
    assert body == render_cost().strip(), "README cost section is stale: run `make demo`"
