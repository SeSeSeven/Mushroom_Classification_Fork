# Makefile

.PHONY: all make_dataset train_model predict_model visualize test pull_data lint coverage build_docker run_docker run_pipeline

# Define your Python executable and script names
PYTHON := python
SCRIPTS_DIR := mushroom_classification
TEST_DIR := tests
COV_REPORT_DIR := reports/coverage

RAW_DIR := data/Mushroom_Image_Dataset
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
VISUALIZE_SCRIPT := $(SCRIPTS_DIR)/visualization/visualize.py

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