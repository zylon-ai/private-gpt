FROM python:3.10.11

RUN groupadd -g 10009 -o privategpt && useradd -m -u 10009 -g 10009 -o -s /bin/bash privategpt
USER privategpt
WORKDIR /home/privategpt

COPY ./src src

RUN cd src && pip install -r requirements.txt

# ENTRYPOINT ["python", "src/privateGPT.py"]