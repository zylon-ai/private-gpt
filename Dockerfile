FROM ubuntu:latest

RUN apt-get update \ 
    && apt-get upgrade -y \
    && apt-get install -y \
    git \ 
    python3 \ 
    python-is-python3 \
    pip \
    wget

WORKDIR /home 
RUN git clone https://github.com/imartinez/privateGPT.git 
WORKDIR /home/privateGPT
RUN pip install -r requirements.txt

RUN mkdir -p /home/privateGPT/models

WORKDIR /home/privateGPT/models
RUN wget --progress=bar:force https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
WORKDIR /home/privateGPT
RUN echo " MODEL_TYPE=GPT4All\nPERSIST_DIRECTORY=db\nMODEL_PATH=models/ggml-gpt4all-j-v1.3-groovy.bin\nEMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2\nMODEL_N_CTX=1000" > .env \
    && chmod a+x .env
