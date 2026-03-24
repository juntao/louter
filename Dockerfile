# Multi-stage build for Louter
# Stage 1: Build frontend
FROM node:22-slim AS frontend
WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Build Rust binary
FROM rust:1.83-bookworm AS backend
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY migrations/ migrations/
COPY --from=frontend /build/web/dist web/dist
RUN cargo build --release

# Stage 3: Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=backend /build/target/release/louter /usr/local/bin/louter
COPY distill/ /opt/louter/distill/

WORKDIR /app/data
EXPOSE 6188

CMD ["louter"]
