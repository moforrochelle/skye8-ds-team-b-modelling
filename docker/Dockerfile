# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install compilation tools temporarily
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition and source code
COPY pyproject.toml .
COPY src/ src/

# Install the package and dependencies locally into /root/.local
RUN pip install --no-cache-dir --user .

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed python packages and source from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/src /app/src

# Make sure scripts in .local are usable in PATH
ENV PATH=/root/.local/bin:$PATH

# Create a non-root user for security and assign ownership
RUN useradd -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set the default command to the registered CLI script
ENTRYPOINT ["score-claims"]