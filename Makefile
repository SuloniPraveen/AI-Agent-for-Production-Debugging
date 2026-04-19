install:
	pip install uv
	uv sync

DOCKER_COMPOSE ?= docker-compose

set-env:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make set-env ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ] && [ "$(ENV)" != "test" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production, test"; \
		exit 1; \
	fi
	@echo "Setting environment to $(ENV)"
	@bash -c "source scripts/set_env.sh $(ENV)"

prod:
	@echo "Starting server in production environment"
	@bash -c "source scripts/set_env.sh production && ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop uvloop"

staging:
	@echo "Starting server in staging environment"
	@bash -c "source scripts/set_env.sh staging && ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop uvloop"

# Uvicorn only (no free-port). Use after `make docker-db` / `make start` so we never kill PIDs
# between "host can reach 5433" and the first DB connection (Docker Desktop can flake otherwise).
dev-server:
	@bash -c "source scripts/set_env.sh development && exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --loop uvloop"

# Single process (no --reload). Frees 8000 first for manual runs when something already listens.
dev:
	@echo "Starting server in development environment"
	@bash scripts/free-port.sh 8000
	@$(MAKE) dev-server

# Hot reload (may fail DB connect on first boot if the reloader imports before port 5433 is ready).
dev-reload:
	@echo "Starting server in development environment (uvicorn --reload)"
	@bash scripts/free-port.sh 8000
	@bash -c "source scripts/set_env.sh development && exec uv run uvicorn app.main:app --host 127.0.0.1 --reload --port 8000 --loop uvloop"

# Postgres only (loads POSTGRES_* from .env.development for compose interpolation)
docker-db:
	@bash scripts/start-db-docker.sh

# Postgres in Docker, then API (open Docker Desktop first). Free 8000 *before* DB so we never
# kill processes after the host port 5433 check (that pattern broke Docker port forwards on macOS).
start:
	@docker info >/dev/null 2>&1 || { echo ""; echo "Docker is not running. Open Docker Desktop, wait until it says it is running, then:"; echo "  make start"; echo ""; exit 1; }
	@bash scripts/free-port.sh 8000
	@$(MAKE) docker-db
	@sleep 2
	@$(MAKE) dev-server

# Evaluation commands
eval:
	@echo "Running evaluation with interactive mode"
	@bash -c "source scripts/set_env.sh ${ENV:-development} && python -m evals.main --interactive"

eval-quick:
	@echo "Running evaluation with default settings"
	@bash -c "source scripts/set_env.sh ${ENV:-development} && python -m evals.main --quick"

# Phase 3: root-cause RAG eval (synthetic dataset, three baselines, JSON under evals/reports/)
eval-rag:
	@echo "Running rag_root_cause eval (see evals/rag_root_cause/METHODOLOGY.md)"
	@bash -c "source scripts/set_env.sh $${ENV:-development} && PYTHONPATH=. uv run python -m evals.rag_root_cause.run"

# Phase 4: regenerate synthetic logs (default 50k lines → demo_data/sample_service.log)
demo-data:
	PYTHONPATH=. uv run python scripts/generate_demo_logs.py

# Streamlit UI against API on API_BASE_URL (default http://localhost:8000). Needs DEMO_API_KEY in env.
# Free 8501 first — a previous Streamlit run often still holds the port.
demo-ui:
	@bash scripts/free-port.sh 8501
	@bash -c 'source scripts/set_env.sh $${ENV:-development} && exec uv run --extra demo streamlit run demo/streamlit_app.py --server.port 8501'

# Reviewer happy path: Postgres + Redis + API + demo — open http://localhost:8501
demo-up:
	@echo "Starting db, redis, app, demo — open http://localhost:8501"
	@ENV_FILE=.env.$${ENV:-development}; \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Missing $$ENV_FILE — copy .env.example and save as $$ENV_FILE"; \
		exit 1; \
	fi; \
	APP_ENV=$${ENV:-development} $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d --build db redis app demo

# Pick up app/ or demo/ code changes without rebuilding the whole stack
demo-restart:
	@ENV_FILE=.env.$${ENV:-development}; \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Missing $$ENV_FILE — copy .env.example and save as $$ENV_FILE"; \
		exit 1; \
	fi; \
	APP_ENV=$${ENV:-development} $(DOCKER_COMPOSE) --env-file $$ENV_FILE restart app demo

# Live logs (Ctrl+C stops following). API is usually what you want when debugging ingest/chat.
demo-logs:
	@$(DOCKER_COMPOSE) logs -f app demo

demo-logs-all:
	@$(DOCKER_COMPOSE) logs -f

eval-no-report:
	@echo "Running evaluation without generating report"
	@bash -c "source scripts/set_env.sh ${ENV:-development} && python -m evals.main --no-report"

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache

docker-build:
	docker build -t fastapi-langgraph-template .

docker-build-env:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-build-env ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production"; \
		exit 1; \
	fi
	@./scripts/build-docker.sh $(ENV)

docker-run:
	@ENV_FILE=.env.development; \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=development $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d --build db app

docker-run-env:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-run-env ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d --build db app
	# @./scripts/ensure-db-user.sh $(ENV)

docker-logs:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-logs ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE logs -f app db

docker-stop:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-stop ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE down

# Docker Compose commands for the entire stack
docker-compose-up:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-compose-up ENV=development|staging|production"; \
		exit 1; \
	fi
	@if [ "$(ENV)" != "development" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "production" ]; then \
		echo "ENV is not valid. Must be one of: development, staging, production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d

docker-compose-down:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-compose-down ENV=development|staging|production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE down

docker-compose-logs:
	@if [ -z "$(ENV)" ]; then \
		echo "ENV is not set. Usage: make docker-compose-logs ENV=development|staging|production"; \
		exit 1; \
	fi
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Environment file $$ENV_FILE not found. Please create it."; \
		exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE logs -f

# Prometheus + Grafana only (no API). Use after `make docker-compose-up` or `make demo-up` if you want metrics; or run this alone to open the UIs (scrape targets stay down until app runs on Compose).
monitoring-up:
	@echo "Starting Prometheus + Grafana (Docker Desktop must be running)"
	@ENV_FILE=.env.$${ENV:-development}; \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Missing $$ENV_FILE — copy .env.example to .env.development"; \
		exit 1; \
	fi; \
	APP_ENV=$${ENV:-development} $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d prometheus grafana
	@echo ""
	@echo "  Prometheus → http://localhost:9090"
	@echo "  Grafana    → http://localhost:3000  (login: admin / admin)"
	@echo ""

# Help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  install: Install dependencies"
	@echo "  set-env ENV=<environment>: Set environment variables (development, staging, production, test)"
	@echo "  run ENV=<environment>: Set environment and run server"
	@echo "  prod: Run server in production environment"
	@echo "  staging: Run server in staging environment"
	@echo "  dev: Run server in development environment"
	@echo "  start: Docker Desktop must be running; then docker-db + dev (full local stack)"
	@echo "  docker-db: Start Postgres (pgvector) via Docker Compose using .env.development"
	@echo "  eval: Run evaluation with interactive mode"
	@echo "  eval-quick: Run evaluation with default settings"
	@echo "  eval-no-report: Run evaluation without generating report"
	@echo "  test: Run tests"
	@echo "  clean: Clean up"
	@echo "  docker-build: Build default Docker image"
	@echo "  docker-build-env ENV=<environment>: Build Docker image for specific environment"
	@echo "  docker-run: Run default Docker container"
	@echo "  docker-run-env ENV=<environment>: Run Docker container for specific environment"
	@echo "  docker-logs ENV=<environment>: View logs from running container"
	@echo "  docker-stop ENV=<environment>: Stop and remove container"
	@echo "  monitoring-up: Start only Prometheus + Grafana (ENV defaults to development)"
	@echo "  docker-compose-up: Start the entire stack (API, Prometheus, Grafana)"
	@echo "  docker-compose-down: Stop the entire stack"
	@echo "  docker-compose-logs: View logs from all services"
	@echo "  demo-up: Start db, redis, app, demo (Phase 4)"
	@echo "  demo-restart: Restart app + demo containers (after code edits)"
	@echo "  demo-logs: Follow app + demo container logs live"
	@echo "  demo-logs-all: Follow every Compose service"
	@echo "  demo-ui / demo-data: Local Streamlit / regenerate demo log file"
	@echo "  eval-rag: Phase 3 root-cause RAG eval"