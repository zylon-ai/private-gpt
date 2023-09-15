FROM python:3-slim-bullseye
EXPOSE 8080
ENV PORT=8080

# Install everything to build llama.cpp and install it
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python --verbose

WORKDIR /app

# Download a sample model, in this case Llama-7b
ADD "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf" models/llama-2-7b-chat.Q4_0.gguf

# Copy dependencies first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy everything else
COPY . .
ENTRYPOINT python api/main.py