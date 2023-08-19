FROM python:3.11-slim

COPY *.py /
COPY example.env .env
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# Installing base dependecies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install build-essential gcc wget curl -y \
    && apt-get clean

# Update pip
RUN pip install --root-user-action=ignore --upgrade pip  \
    && pip install --root-user-action=ignore -r requirements.txt

# Download ggmml
RUN mkdir "models"
RUN wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin -P /models

# Install poetry
RUN pip install sentence_transformers poetry
RUN poetry env use python3.11
RUN poetry config installer.max-workers 10
RUN poetry --no-interaction --no-ansi -vvv --no-root install

# TODO: Fix next lines.
#RUN useradd -ms /bin/bash python
#USER python
#WORKDIR /
#RUN poetry shell

RUN echo "alias update_gpt='python3 /ingest.py'" >> ~/.bashrc  \
    && echo "alias ask_gpt='python3 /privateGPT.py'" >> ~/.bashrc

# Ingest all the data
#RUN python ingest.py

CMD ["/bin/bash"]
