FROM ubuntu:latest

RUN apt-get update -qq && apt-get install -y \
    git \
    python3 \
    python-is-python3 \
    pip \
    wget

RUN cd home \
    && git clone https://github.com/imartinez/privateGPT.git \
    && cd privateGPT \
    && pip install -r requirements.txt

RUN echo -e \ "PERSIST_DIRECTORY=db\nLLAMA_EMBEDDINGS_MODEL=models/ggml-model-q4_0.bib\nMODEL_TYPE=GPT4Alb\nMODEL_PATH=models/ggml-gpt4all-j-v1.3-groovy.bib\nMODEL_N_CTX=1000" > .env \
    && chmod a+x .env

RUN mkdir models \
    && cd models \
    && wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin \
    && wget https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin \
    && cd ..
