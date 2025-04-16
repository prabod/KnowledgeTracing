#!/bin/bash

set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to path
    source $HOME/.local/bin/env
fi

# Create a new virtual environment
uv venv --python 3.11.6

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using uv pip
uv pip install -r requirements.txt

# Install the knowledge tracing package
uv pip install -e knowledge_tracing

echo "UV environment setup complete!"