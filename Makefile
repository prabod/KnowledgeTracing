# Env vars
include .env
export

# Version for production deployment
VERSION := 0.0.1
REPEAT := 10

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

train:
	python train.py --use-previous-responses true --lr 0.01 --repeat $(REPEAT) --max-epochs 20

train-with-interaction:
	python train.py --use-previous-responses true --lr 0.01 --repeat $(REPEAT) --max-epochs 20 --with-interaction true

clean:
	rm -rf lightning_logs/*
	rm -rf logs/train/*

.PHONY: lint format ruff uv clean train train-with-interaction