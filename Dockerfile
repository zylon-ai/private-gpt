FROM python:3.10-slim

RUN apt update
RUN apt install -y build-essential

WORKDIR /code
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m pip install sentence_transformers