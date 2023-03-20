FROM python:3.10-slim

RUN apt-get update
RUN apt-get -y install gcc

WORKDIR /app

COPY . .

ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev


WORKDIR /
ENTRYPOINT ["/bin/bash"]
