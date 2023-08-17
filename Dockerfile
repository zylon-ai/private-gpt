#FROM alpine:3.17.5
FROM ubuntu:22.04

# Copy all sources.
COPY . .
COPY sh/entrypoint.sh /
COPY sh/ask_gpt /user/bin
COPY sh/update_gpt /user/bin

# Installing base dependecies
RUN apt update && apt install -y wget python3 pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Installing python dependencies
RUN pip3 install -r requirements.txt

# Download ggmml
RUN mkdir "models"
RUN wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin -P /models

# Install poetry
RUN pip install sentence_transformers poetry \
    poetry install \
    poetry shell

# ingest all the data
RUN python3 /ingest.py

# Copy envirnoment
COPY example.env .env

#CMD ["/bin/sh"]
ENTRYPOINT ["/entrypoint.sh"]