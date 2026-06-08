# Variáveis
N_WORKERS ?= 4
N_SAMPLES ?= 50
SEED ?= 8082
PREFIX := \033[0;36m[Makefile][aibox.nlp]\033[0m

all: format

configure-pre-commit:
	@pre-commit install

ensure-compatibility:
	@echo -e "$(PREFIX) Running compatibility tests..."
	@python tools/ensure_compatibility.py

format:
	@echo -e "$(PREFIX) Formatting source code..."
	@isort src/ tests/
	@black src/ tests/

test-extractor-samples:
	@echo -e "$(PREFIX) Running feature extraction for 'essayBR' samples..."
	@python tools/test_features_dataset_samples.py -n $(N_SAMPLES) -s $(SEED) -w $(N_WORKERS) essayBR
	@echo -e "$(PREFIX) Running feature extraction for 'narrativeEssaysBR' samples..."
	@python tools/test_features_dataset_samples.py -n $(N_SAMPLES) -s $(SEED) -w $(N_WORKERS) narrativeEssaysBR

run-pytest:
	@echo -e "$(PREFIX) Running tests with 'pytest'..."
	@TEST_EXPENSIVE_AIBOX_NLP=1 pytest -x

prune-merged:
	@git checkout main
	@git fetch --prune
	@git branch --merged | sed '/\*.*/d' | sed 's/  //g' | xargs git branch -D

run-tests: run-pytest test-extractor-samples ensure-compatibility
