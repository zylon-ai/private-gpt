FROM python:3.10-slim

RUN apt update
RUN apt install -y build-essential

WORKDIR /code
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt