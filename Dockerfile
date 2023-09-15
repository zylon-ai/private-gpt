FROM python:3-slim-bullseye
EXPOSE 8080
ENV PORT=8080

# Install everything to build llama.cpp and install it
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config wget
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python --verbose

WORKDIR /app

# Copy dependencies first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy everything else
COPY . .
ENTRYPOINT python api/main.py