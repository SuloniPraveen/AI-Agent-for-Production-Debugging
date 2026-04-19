"""Prometheus metrics configuration for the application.

This module sets up and configures Prometheus metrics for monitoring the application.
"""

from prometheus_client import Counter, Histogram, Gauge
from starlette_prometheus import metrics, PrometheusMiddleware

# Request metrics
http_requests_total = Counter("http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"])

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)

# Database metrics
db_connections = Gauge("db_connections", "Number of active database connections")

# Custom business metrics
orders_processed = Counter("orders_processed_total", "Total number of orders processed")

# LLM latency (chat completions) — use this name for dashboards / SLOs
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "Wall time for a single LLM chat completion in the agent",
    ["model"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

llm_stream_duration_seconds = Histogram(
    "llm_stream_duration_seconds",
    "Time spent processing LLM stream inference",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Log search cache + retrieval (Phase 2)
cache_hits_total = Counter("cache_hits_total", "Cache hits by cache name", ["cache"])

cache_misses_total = Counter("cache_misses_total", "Cache misses by cache name", ["cache"])

# phase: embed, vector_search, total_miss (embed+DB+cache set), total_hit (redis only)
retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Log search retrieval latency by stage",
    ["phase"],
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)


def setup_metrics(app):
    """Set up Prometheus metrics middleware and endpoints.

    Args:
        app: FastAPI application instance
    """
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Add metrics endpoint
    app.add_route("/metrics", metrics)
