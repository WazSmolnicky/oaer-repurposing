FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY protocols ./protocols
COPY catalogue ./catalogue
COPY pipelines ./pipelines
COPY tests ./tests
COPY pyproject.toml README.md LICENSE NOTICE ./

ENV PYTHONPATH=/workspace/src

ENTRYPOINT ["python", "-m", "oaer.console.rank", "--protocol", "protocols/main.yaml"]
