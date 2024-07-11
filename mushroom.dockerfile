# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc wget python3.12 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile
COPY data.dvc data.dvc
COPY mushroom_classification/ mushroom_classification/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Install make
RUN apt-get update && apt-get install -y make

# Always run make
ENTRYPOINT ["make"]

# Default target to run
CMD ["run_pipeline"]
