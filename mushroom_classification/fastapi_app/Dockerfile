FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY ../mushroom_classification/ /app/mushroom_classification/

RUN pip install --no-cache-dir -r requirements.txt

ENV VERTEX_AI="false"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
