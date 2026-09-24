# make setup   -> virtualenv, dependencies, warehouse (first time)
# make ask Q="how many accounts have more than 50000?"
# make help    -> everything else
# Works on macOS, Linux and Windows (Git Bash / WSL).

.DEFAULT_GOAL := help
.PHONY: help setup install data verify ask api ui lint typecheck check eval eval-ablations docker clean

VENV := .venv
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
    PY       := python
else
    VENV_BIN := $(VENV)/bin
    PY       := python3
endif
PYTHON := $(VENV_BIN)/python
NL2SQL := $(VENV_BIN)/nl2sql
Q ?= How many accounts have more than 50000 in them?

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)  ## Install the package with all extras
	$(PYTHON) -m pip install -e ".[embeddings,ui,dev]"

setup: install data  ## One-time setup: dependencies + warehouse

data:  ## Download the dataset and build the warehouse (~1 min)
	$(NL2SQL) data build

verify:  ## Check the warehouse's integrity
	$(NL2SQL) data verify

ask:  ## Ask a question: make ask Q="..."
	@$(NL2SQL) ask "$(Q)"

api:  ## HTTP API on :8000 (docs at /docs)
	$(NL2SQL) serve

ui:  ## Streamlit demo on :8501
	$(NL2SQL) ui

lint:  ## Style check
	$(PYTHON) -m ruff check src app

typecheck:  ## Static type check
	$(PYTHON) -m mypy src/nl2sql

check: lint typecheck verify  ## Everything CI runs

eval:  ## Run the 70-question benchmark
	$(NL2SQL) eval

eval-ablations:  ## Measure what each component contributes
	$(NL2SQL) eval --quiet
	$(NL2SQL) eval --quiet --no-retrieval
	$(NL2SQL) eval --quiet --no-few-shot
	$(NL2SQL) eval --quiet --no-repair

docker:  ## Build and start the API with docker compose
	docker compose up --build

clean:  ## Remove caches, the warehouse and downloaded data
	rm -rf .mypy_cache .ruff_cache build dist data/raw data/db data/cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
