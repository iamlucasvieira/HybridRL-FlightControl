FROM continuumio/miniconda3

WORKDIR /app

COPY . .

RUN conda env create -n env --file environment-docker.yml python=3.10.8

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

#RUN pip install --upgrade pip && pip --version && python3 --version
# Needed because of the issue of gym with setuptools
RUN pip install setuptools==65.5.0 --no-cache-dir &&\
    pip install stable-baselines3 && \
    echo "Make sure Gym is installed:" && \
    python -c "import gym" && \
    rm -rf $HOME/.cache/pip && \
    conda remove nox scikit-learn jupyterlab


WORKDIR /
ENTRYPOINT ["/bin/bash"]
