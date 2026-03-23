## syntax=docker/dockerfile:1.4
FROM k-server-env

ARG K_SERVER_BENCH_USE_LEGACY_EVALUATOR=false

ENV K_SERVER_ENV_NAME=k-server
ENV K_SERVER_BENCH_USE_LEGACY_EVALUATOR=${K_SERVER_BENCH_USE_LEGACY_EVALUATOR}
ENV CODEX_AUTH_MOUNT_PATH=/run/secrets/codex-auth.json

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER kserver

RUN micromamba install -y -n ${K_SERVER_ENV_NAME} -c conda-forge \
    nodejs \
    && micromamba clean -a -y

RUN micromamba run -n ${K_SERVER_ENV_NAME} npm install -g @openai/codex

USER root

COPY --chown=kserver:kserver agents /home/kserver/k-server-bench/agents
RUN chmod +x /home/kserver/k-server-bench/agents/codex/entrypoint.sh

WORKDIR /home/kserver/workspace

CMD ["/home/kserver/k-server-bench/agents/codex/entrypoint.sh", "codex", "--dangerously-bypass-approvals-and-sandbox"]
