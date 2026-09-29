.PHONY: install lint format test test-download demo data retrieval contexts validation hhem snapshot estimate judge-dev judge-freeze eval-live eval-replay

# Dollar cap for every live model call (DollarCap refuses anything that could pass it).
CAP ?= 8.00

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

# Tests that download the real models at their pinned revisions.
test-download:
	uv run pytest -q -m download

# Offline, no keys: checks data hashes and rewrites the README results section
# from the committed results. Live rows print "pending live run" until they exist.
demo:
	uv run eval verify-data
	uv run eval demo

# Verify the vendored Tallowbrook files against data/tallowbrook/MANIFEST.json.
data:
	uv run python scripts/sync_data.py

# ---- offline, CPU (downloads the open models on first use)

# Retrieval grid: dev selection, then every config on test once. Includes the Qwen3 strong arm.
retrieval:
	uv run eval retrieval

# Freeze the top chunks of the three generation configs (so replays are byte-identical).
contexts:
	uv run eval contexts

# Rebuild the perturbation set and the seeded RAGTruth subset (downloads RAGTruth at a pinned commit).
validation:
	uv run eval build-validation

hhem:
	uv run eval hhem

snapshot export-snapshot:
	uv run eval export-snapshot

estimate:
	uv run eval estimate

# ---- model calls (need AZURE_OPENAI_BASE_URL plus AZURE_OPENAI_API_KEY or Entra ID)

# Judge the dev perturbations. Edit src/rag_support_assistant/prompts/ and repeat, dev only.
judge-dev:
	uv run eval judge-dev --live --cap $(CAP)

# Freeze the judges. Test-set judging refuses to run on a judge that changed after this.
judge-freeze:
	uv run eval judge-freeze

# The full live run on test: judges on the perturbation and RAGTruth sets, answers for
# three retrieval configs plus full context, both judges on every answer, the contextual
# retrieval cell and Ragas. Freezes the judges after the dev run if they are not frozen yet.
eval-live:
	uv run eval run --live --cap $(CAP)
	uv run eval hhem --answers-only
	uv run eval demo

# Replay the committed cache. Sends nothing, fails on any cache miss.
eval-replay:
	uv run eval run --replay
