## syntax=docker/dockerfile:1.4
FROM k-server-env

ARG K_SERVER_BENCH_USE_LEGACY_EVALUATOR=false

ENV K_SERVER_ENV_NAME=k-server
ENV K_SERVER_BENCH_USE_LEGACY_EVALUATOR=${K_SERVER_BENCH_USE_LEGACY_EVALUATOR}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=kserver:kserver agents /home/kserver/k-server-bench/agents
RUN chmod +x /home/kserver/k-server-bench/agents/entrypoint.sh

WORKDIR /home/kserver/workspace

ENTRYPOINT ["/opt/entrypoint.sh", "/home/kserver/k-server-bench/agents/entrypoint.sh"]
CMD ["bash"]
