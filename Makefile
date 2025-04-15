# Env vars
include .env
export

# Version for production deployment
VERSION := 0.0.1


# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
NC := \033[0m  # No Color

# Explicitly use bash
SHELL := /bin/bash


lint:
	ruff check

format:
	ruff format

ruff: lint format

# UV environment setup target
uv:
	@echo "$(GREEN)Setting up UV environment...$(NC)"
	@bash scripts/setup.sh


.PHONY: lint format ruff uv