# syntax=docker/dockerfile:1.7

FROM --platform=linux/arm64 rust:1.94-bookworm AS builder
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2
RUN cargo build --locked --release -p cascadia-cli-v2
RUN cargo build --locked --release -p cascadia-differential \
    --features legacy-teacher \
    --bin r2-map-cross-arch-focal \
    --bin r2-map-focal-control

FROM builder AS test
RUN cargo test --locked --release -p cascadia-eval focal --lib \
    && cargo test --locked --release -p cascadia-eval longitudinal --lib \
    && cargo test --locked --release -p cascadia-differential \
        --features legacy-teacher r2_map_cross_arch_focal

FROM scratch AS provenance-source
COPY CASCADIA_V2_GOAL.txt orchestrator-rewrite.md Cargo.toml Cargo.lock Makefile pyproject.toml uv.lock /repo/
COPY apps/web/src /repo/apps/web/src
COPY crates /repo/crates
COPY legacy/crates /repo/legacy/crates
COPY python/cascadia_mlx /repo/python/cascadia_mlx
COPY python/cascadia_cluster /repo/python/cascadia_cluster
COPY docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md /repo/docs/v2/
COPY docs/v2/reports/r2-map-bootstrap-cross-architecture-250-preregistration-v1.md /repo/docs/v2/reports/
COPY docs/v2/reports/legacy-nnue-v4opp-mlx-exact-rollout-wave-v1.json /repo/docs/v2/reports/

FROM --platform=linux/arm64 debian:bookworm-slim AS runtime
ARG SOURCE_REVISION=unknown
ARG SOURCE_BLAKE3=unknown
ARG IMAGE_TAG=local
LABEL org.opencontainers.image.title="Cascadia R2-MAP worker" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.source-blake3="${SOURCE_BLAKE3}"
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libopenblas0-pthread python3 python3-numpy python3-pip \
    && python3 -m pip install --break-system-packages --no-cache-dir blake3==1.0.8 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 cascadia \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin cascadia \
    && install -d -o cascadia -g cascadia -m 0750 /input /output
COPY --from=test /src/target/release/cascadia-v2 /usr/local/bin/cascadia-v2
COPY --from=builder /src/target/release/r2-map-cross-arch-focal /usr/local/bin/r2-map-cross-arch-focal
COPY --from=builder /src/target/release/r2-map-focal-control /usr/local/bin/r2-map-focal-control
COPY infra/bacalhau/cluster-job-entrypoint.sh /usr/local/bin/cascadia-cluster-job
COPY --from=provenance-source /repo /opt/cascadia/repo
RUN chmod 0755 /usr/local/bin/cascadia-cluster-job \
    && chmod -R a=rX /opt/cascadia/repo
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/cascadia/repo/python \
    OPENBLAS_NUM_THREADS=8 \
    OMP_NUM_THREADS=8
USER 10001:10001
WORKDIR /opt/cascadia/repo
ENTRYPOINT ["/usr/local/bin/cascadia-v2"]
CMD ["--help"]
