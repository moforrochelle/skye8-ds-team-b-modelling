## Use a lightweight, secure python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration file
COPY pyproject.toml ./

# Copy source code into the container
COPY src/ src/

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Setting up default command to invoke the scoring CLI module
ENTRYPOINT ["score-claims"]