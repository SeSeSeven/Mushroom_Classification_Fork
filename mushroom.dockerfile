# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile

COPY mushroom_classification/ mushroom_classification/

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Install make
RUN apt-get update && apt-get install -y make

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true
RUN dvc pull

# Always run make
ENTRYPOINT ["make"]

# Default target to run
CMD ["run_pipeline"]
