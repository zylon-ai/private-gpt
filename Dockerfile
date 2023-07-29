FROM python:3.10-slim

WORKDIR /code
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt