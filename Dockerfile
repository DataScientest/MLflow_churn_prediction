FROM python:3.10-slim

# Install uv and Docker CLI (client only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# uv sync creates the environment in .venv by default
RUN uv sync --frozen

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY . .

# Default entrypoint
CMD ["python", "src/train.py"]
