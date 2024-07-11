# Makefile

.PHONY: all make_dataset train_model predict_model visualize test pull_data lint coverage build_docker run_docker run_pipeline

# Define your Python executable and script names
PYTHON := python
SCRIPTS_DIR := mushroom_classification
TEST_DIR := tests
COV_REPORT_DIR := reports/coverage

RAW_DIR := data/raw
PROCESSED_DIR := data/processed
RANDOM_STATE := 42
SAVE_MODEL := models/resnet50.ckpt
MODEL_NAME := resnet50.a1_in1k
NUM_CLASSES := 9
BATCH_SIZE := 32
OUTPUT_DIR := outputs
VAL_SIZE := 0.15
TEST_SIZE := 0.15
RANDOM_STATE := 42
PREDICTION_PATH := outputs/predictions.npy
REPORT_DIR := reports/figures
NUM_IMAGES := 16
FIGURE_ARRANGE := "(4,4)"
METRICS_PATH := outputs/metrics.csv

MAKE_DATASET_SCRIPT := $(SCRIPTS_DIR)/data/make_dataset.py
TRAIN_MODEL_SCRIPT := $(SCRIPTS_DIR)/models/train_model.py
PREDICT_MODEL_SCRIPT := $(SCRIPTS_DIR)/models/predict_model.py
VISUALIZE_SCRIPT := $(SCRIPTS_DIR)/visulaization/visualize.py

# Default target
all: make_dataset train_model predict_model visualize

# Target to create the dataset
make_dataset: pull_data
	$(PYTHON) $(MAKE_DATASET_SCRIPT) -d $(RAW_DIR) -p $(PROCESSED_DIR) -v $(VAL_SIZE) -t $(TEST_SIZE) -r $(RANDOM_STATE)

# Target to train the model
train_model:
	$(PYTHON) $(TRAIN_MODEL_SCRIPT)

# Target to predict using the model
predict_model:
	$(PYTHON) $(PREDICT_MODEL_SCRIPT) -p $(PROCESSED_DIR) -s $(SAVE_MODEL) -n $(MODEL_NAME) -c $(NUM_CLASSES) -b $(BATCH_SIZE) -o $(OUTPUT_DIR)

# Target to visualize the results
visualize:
	$(PYTHON) $(VISUALIZE_SCRIPT) -p $(PROCESSED_DIR) -e $(PREDICTION_PATH) -o $(REPORT_DIR) -i $(NUM_IMAGES) -f $(FIGURE_ARRANGE) -r $(RANDOM_STATE) -m $(METRICS_PATH)

# Target to pull data with DVC
pull_data:
	dvc pull

# Target to run linter
lint:
	ruff check $(SCRIPTS_DIR)
	ruff check $(TEST_DIR)

# Target to run tests
test:
	pytest $(TEST_DIR)

# Target to generate coverage report
coverage:
	coverage run -m pytest $(TEST_DIR)
	coverage report
	coverage html -d $(COV_REPORT_DIR)

# Target to build the Docker image
build_docker:
	docker build -f mushroom.dockerfile . -t mushroom:latest

# Target to run the Docker container
run_docker:
	docker run --name mc1 mushroom:latest

# Target to run the entire pipeline
run_pipeline: pull_data lint test coverage make_dataset train_model predict_model visualize

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')