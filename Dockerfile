FROM rust:1.83-bookworm AS builder
WORKDIR /app
COPY Cargo.toml ./
COPY migrations ./migrations
COPY src ./src
RUN cargo build --release --bin rag-server --bin rag-mcp --bin rag-eval

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/rag-server /usr/local/bin/rag-server
COPY --from=builder /app/target/release/rag-mcp /usr/local/bin/rag-mcp
COPY --from=builder /app/target/release/rag-eval /usr/local/bin/rag-eval
EXPOSE 8010
CMD ["rag-server"]
