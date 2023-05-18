FROM python:3.10.9
WORKDIR /root/
RUN apt-get update && apt-get upgrade -y
ENV PATH="/root/.local/bin:${PATH}"
COPY requirements.txt .
RUN pip3 install --user --upgrade pip
RUN pip3 install --user -r requirements.txt
