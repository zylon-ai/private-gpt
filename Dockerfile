#FROM python:3.10.11
#FROM wallies/python-cuda:3.10-cuda11.6-runtime

# Using argument for base image to avoid multiplying Dockerfiles
ARG BASEIMAGE
FROM $BASEIMAGE

RUN groupadd -g 10009 -o privategpt && useradd -m -u 10009 -g 10009 -o -s /bin/bash privategpt
USER privategpt
WORKDIR /home/privategpt

COPY ./src/requirements.txt src/requirements.txt
ARG LLAMA_CMAKE
#RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install $(grep llama-cpp-python src/requirements.txt)
RUN ( /bin/bash -c "${LLAMA_CMAKE} pip install \$(grep llama-cpp-python src/requirements.txt)" 2>&1 | tee llama-build.log ) && sleep 10
RUN pip install --no-cache-dir -r src/requirements.txt 2>&1 | tee pip-install.log

COPY ./src src

# ENTRYPOINT ["python", "src/privateGPT.py"]
