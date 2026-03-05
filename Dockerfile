# Pull the uv image to avoid an additional download step
FROM ghcr.io/astral-sh/uv:0.9.30-python3.13-trixie-slim

# Create a non-root user to run the application
RUN useradd --create-home appuser

WORKDIR /app

# Make the venv available without uv run
ENV PATH="/app/.venv/bin:$PATH"

# Copy the lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install required dependencies.
# The `--no-install-project` flag is used to avoid installing the project itself,
# which is handled in a later step to avoid re-downloading dependencies
# when the source code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src src
COPY README.md .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Switch to non-root after install
USER appuser

ENTRYPOINT ["python", "src/journal_2026/trial.py"]