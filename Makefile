.PHONY: all make_dataset train_model predict_model visualize test lint coverage build_docker run_docker run_pipeline pull_data

# Define your Python executable and script names
PYTHON := python
SCRIPTS_DIR := mushroom_classification
TEST_DIR := tests

# Define paths based on the environment
ifeq ($(VERTEX_AI), "true")
    RAW_DIR := /gcs/mushroom_test_bucket/data/raw
    PROCESSED_DIR := /gcs/mushroom_test_bucket/data/processed
    MODEL_DIR := /gcs/mushroom_test_bucket/models
    SAVE_MODEL := $(MODEL_DIR)/resnet50.pt
    OUTPUT_DIR := /gcs/mushroom_test_bucket/outputs
    REPORT_DIR := /gcs/mushroom_test_bucket/reports/figures
    COV_REPORT_DIR := /gcs/mushroom_test_bucket/reports/coverage
    METRICS_PATH := $(OUTPUT_DIR)/metrics.csv
    PREDICTION_PATH := $(OUTPUT_DIR)/predictions.npy
    VERTEX_FLAG := true
else
    RAW_DIR := data/raw
    PROCESSED_DIR := data/processed
    MODEL_DIR := models
    SAVE_MODEL := $(MODEL_DIR)/resnet50.pt
    OUTPUT_DIR := outputs
    REPORT_DIR := reports/figures
    COV_REPORT_DIR := reports/coverage
    METRICS_PATH := $(OUTPUT_DIR)/metrics.csv
    PREDICTION_PATH := $(OUTPUT_DIR)/predictions.npy
    VERTEX_FLAG := false
endif

# Training related variables
VAL_SIZE := 0.15
TEST_SIZE := 0.15
RANDOM_STATE := 42
NUM_IMAGES := 16
FIGURE_ARRANGE := "(4,4)"

MAKE_DATASET_SCRIPT := $(SCRIPTS_DIR)/data/make_dataset.py
TRAIN_MODEL_SCRIPT := $(SCRIPTS_DIR)/models/train_model.py
PREDICT_MODEL_SCRIPT := $(SCRIPTS_DIR)/models/predict_model.py
VISUALIZE_SCRIPT := $(SCRIPTS_DIR)/visualization/visualize.py

# Default target
all: make_dataset train_model predict_model visualize

# Target to pull data with DVC
# pull_data:
#	dvc pull

# Target to create the dataset
make_dataset: # pull_data
	$(PYTHON) $(MAKE_DATASET_SCRIPT) -d $(RAW_DIR) -p $(PROCESSED_DIR) -v $(VAL_SIZE) -t $(TEST_SIZE) -r $(RANDOM_STATE)

# Target to train the model
train_model:
	$(PYTHON) $(TRAIN_MODEL_SCRIPT) -v $(VERTEX_FLAG) -p $(PROCESSED_DIR) -s $(SAVE_MODEL)

# Target to predict using the model
predict_model:
	$(PYTHON) $(PREDICT_MODEL_SCRIPT) -v $(VERTEX_FLAG) -p $(PROCESSED_DIR) -s $(SAVE_MODEL) -o $(OUTPUT_DIR)

# Target to visualize the results
visualize:
	$(PYTHON) $(VISUALIZE_SCRIPT) -p $(PROCESSED_DIR) -e $(PREDICTION_PATH) -o $(REPORT_DIR) -i $(NUM_IMAGES) -f $(FIGURE_ARRANGE) -r $(RANDOM_STATE) -m $(METRICS_PATH)

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
	docker run --name mc1 -e VERTEX_AI=$(VERTEX_AI) mushroom:latest

# Target to run the entire pipeline
run_pipeline: make_dataset train_model predict_model visualize

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
