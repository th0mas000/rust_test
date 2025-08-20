
FROM rust:1.75

WORKDIR /app

COPY Cargo.toml Cargo.lock ./


COPY src/ ./src/
COPY static/ ./static/


RUN mkdir -p data

RUN cargo build --release --bin server

EXPOSE 8000

ENV RUST_LOG=info


CMD ["./target/release/server"]
