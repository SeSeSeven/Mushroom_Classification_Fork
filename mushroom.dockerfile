# Base image
FROM python:3.10-slim

# Environment variable to control training environment
ENV VERTEX_AI=true

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc wget python3.10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
# COPY data.dvc data.dvc
COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile
COPY mushroom_classification/ mushroom_classification/
COPY tests/ tests/

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Always run make
ENTRYPOINT ["make"]

# Default target to run
CMD ["run_pipeline"]
