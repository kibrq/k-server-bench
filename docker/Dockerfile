## syntax=docker/dockerfile:1.4
FROM mambaorg/micromamba:1.5.8-jammy

ENV K_SERVER_PYTHON_VERSION=3.11
ENV K_SERVER_ENV_NAME=k-server

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    coreutils \
    git \
    openssh-client \
    util-linux \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash kserver

RUN mkdir -p /home/kserver/workspace \
    && chown -R kserver:kserver /home/kserver

ENV HOME=/home/kserver
ENV USER=kserver
ENV LOGNAME=kserver
ENV XDG_CACHE_HOME=/home/kserver/.cache
ENV MAMBA_ROOT_PREFIX=/home/kserver/micromamba
ENV PATH=/home/kserver/micromamba/envs/k-server/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

USER kserver

RUN micromamba create -y -n ${K_SERVER_ENV_NAME} -c conda-forge \
    python=${K_SERVER_PYTHON_VERSION} \
    pip \
    && micromamba clean -a -y

COPY --chown=kserver:kserver k-servers /home/kserver/k-server-bench/k-servers

RUN cd /home/kserver/k-server-bench/k-servers \
    && micromamba run -n ${K_SERVER_ENV_NAME} python -m pip install --no-cache-dir -r requirements.txt \
    && micromamba run -n ${K_SERVER_ENV_NAME} python -m pip install --no-cache-dir -e .

COPY --chown=kserver:kserver metrics /home/kserver/k-server-bench/metrics
COPY --chown=kserver:kserver tasks /home/kserver/k-server-bench/tasks
COPY --chown=kserver:kserver tools /home/kserver/k-server-bench/tools
COPY --chown=kserver:kserver examples /home/kserver/k-server-bench/examples

WORKDIR /home/kserver/workspace

USER root

COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["bash"]
