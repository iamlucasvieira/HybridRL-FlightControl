FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update
RUN apt-get -y install gcc

WORKDIR /app

COPY . .

RUN uv install --no-dev


WORKDIR /
ENTRYPOINT ["/bin/bash"]
