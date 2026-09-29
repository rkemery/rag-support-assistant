# Tallowbrook data (vendored)

A pinned copy of the files this repo needs from the synthetic Tallowbrook Neobank Support dataset: the help-center corpus, the policy facts file and the RAG questions (50 dev, 150 test, 40 of the 200 unanswerable). `MANIFEST.json` names the source commit and the sha256 of every file, and `uv run python scripts/sync_data.py` checks them.

Tallowbrook is a fictional bank. The dataset was written by Claude from one structured facts file, and it has not been audited by a person. Its canonical home will be a Hugging Face dataset, pinned by revision.

License: CC-BY-4.0 (Tallowbrook Neobank Support dataset, Richard K.).
