.PHONY: build validate test all

build:
	uv run python scripts/build_corpus.py
	uv run python scripts/build_questions.py
	uv run python scripts/build_agents.py

validate:
	uv run python scripts/validate.py

test:
	uv run pytest -q

all: build validate test
