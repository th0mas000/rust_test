# Use the official Rust image as the base
FROM rust:1.75

# Set the working directory
WORKDIR /app

# Copy Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ ./src/
COPY static/ ./static/

# Create data directory
RUN mkdir -p data

# Build the application
RUN cargo build --release --bin server

# Expose the port
EXPOSE 8000

# Set environment variables
ENV RUST_LOG=info

# Run the server
CMD ["./target/release/server"]
