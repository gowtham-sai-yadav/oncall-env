#!/usr/bin/env python3
"""Generate 40+ diverse, realistic incident scenarios for OnCallEnv.

Usage:
    python generate_scenarios.py              # writes 48 scenarios to scenarios/
    python generate_scenarios.py --count 60   # override count per difficulty
    python generate_scenarios.py --validate   # validate generated scenarios
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

TASK_DIRS = {
    "easy": "task1_easy",
    "medium": "task2_medium",
    "hard": "task3_hard",
    "expert": "task4_expert",
}

DIFFICULTIES = ["easy", "medium", "hard", "expert"]

SEVERITY_MAP = {
    "easy": "SEV3",
    "medium": "SEV2",
    "hard": "SEV1",
    "expert": "SEV1",
}

# ---------------------------------------------------------------------------
# Service Pool
# ---------------------------------------------------------------------------

SERVICE_POOL = [
    {"name": "payment-api", "type": "api", "tier": "critical"},
    {"name": "order-service", "type": "api", "tier": "critical"},
    {"name": "user-service", "type": "api", "tier": "high"},
    {"name": "inventory-service", "type": "api", "tier": "high"},
    {"name": "notification-service", "type": "worker", "tier": "medium"},
    {"name": "search-service", "type": "api", "tier": "medium"},
    {"name": "auth-service", "type": "api", "tier": "critical"},
    {"name": "session-store", "type": "datastore", "tier": "critical"},
    {"name": "cache-layer", "type": "datastore", "tier": "high"},
    {"name": "message-queue", "type": "infra", "tier": "critical"},
    {"name": "api-gateway", "type": "infra", "tier": "critical"},
    {"name": "cdn-proxy", "type": "infra", "tier": "medium"},
    {"name": "logging-pipeline", "type": "infra", "tier": "low"},
    {"name": "metrics-collector", "type": "infra", "tier": "low"},
    {"name": "email-service", "type": "worker", "tier": "low"},
    {"name": "billing-service", "type": "api", "tier": "critical"},
    {"name": "analytics-service", "type": "worker", "tier": "low"},
    {"name": "config-server", "type": "infra", "tier": "high"},
    {"name": "internal-dns", "type": "infra", "tier": "critical"},
    {"name": "service-mesh", "type": "infra", "tier": "critical"},
]

SERVICE_VERSIONS = [
    "1.0.0", "1.1.0", "1.2.3", "1.5.0", "2.0.0", "2.1.1", "2.3.0",
    "2.4.1", "3.0.0", "3.1.2", "3.2.0", "4.0.1", "4.1.0", "4.2.0", "5.0.3",
]

DEPLOYERS = ["ci-bot", "ops-team", "infra-team", "platform-team", "dev-team",
             "ml-team", "search-team", "sre-bot", "release-manager"]

# ---------------------------------------------------------------------------
# Difficulty Profiles
# ---------------------------------------------------------------------------

DIFFICULTY_PROFILES = {
    "easy":   {"num_services": (4, 5), "num_alerts": (4, 5), "critical_alerts": (1, 2),
               "noise_alerts": (1, 2), "logs_per_service": (3, 5), "metrics_per_service": (1, 2),
               "red_herrings": (0, 0), "cascade_depth": (1, 1), "expected_diags": (1, 2)},
    "medium": {"num_services": (5, 6), "num_alerts": (5, 7), "critical_alerts": (2, 3),
               "noise_alerts": (2, 3), "logs_per_service": (4, 6), "metrics_per_service": (2, 3),
               "red_herrings": (0, 1), "cascade_depth": (1, 2), "expected_diags": (2, 3)},
    "hard":   {"num_services": (6, 7), "num_alerts": (7, 9), "critical_alerts": (2, 4),
               "noise_alerts": (2, 4), "logs_per_service": (5, 8), "metrics_per_service": (2, 3),
               "red_herrings": (1, 2), "cascade_depth": (2, 3), "expected_diags": (3, 4)},
    "expert": {"num_services": (6, 8), "num_alerts": (8, 12), "critical_alerts": (3, 5),
               "noise_alerts": (3, 5), "logs_per_service": (5, 8), "metrics_per_service": (2, 4),
               "red_herrings": (2, 4), "cascade_depth": (2, 4), "expected_diags": (3, 5)},
}

# ---------------------------------------------------------------------------
# Grading Weight Profiles
# ---------------------------------------------------------------------------

WEIGHT_PROFILES = {
    "easy":   {"triage": 0.30, "diagnostic": 0.20, "root_cause": 0.15,
               "remediation": 0.15, "efficiency": 0.10, "documentation": 0.10},
    "medium": {"triage": 0.20, "diagnostic": 0.25, "root_cause": 0.20,
               "remediation": 0.20, "efficiency": 0.10, "documentation": 0.05},
    "hard":   {"triage": 0.10, "diagnostic": 0.25, "root_cause": 0.30,
               "remediation": 0.20, "efficiency": 0.10, "documentation": 0.05},
    "expert": {"triage": 0.05, "diagnostic": 0.20, "root_cause": 0.35,
               "remediation": 0.20, "efficiency": 0.10, "documentation": 0.10},
}

# ---------------------------------------------------------------------------
# Incident Templates (18 templates across 7 categories)
# ---------------------------------------------------------------------------

INCIDENT_TEMPLATES = [
    # ---- DATABASE (3) ----
    {
        "id": "db_conn_pool",
        "category": "database",
        "name": "Connection Pool Exhaustion",
        "root_cause_description": "DB connection pool exhaustion on {service} caused by connection leak in {version} deployment",
        "root_cause_keywords": ["connection pool", "database", "db", "exhausted", "leak", "connection"],
        "preferred_service_types": ["api", "datastore"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Error rate exceeded 5% threshold - currently at {error_rate}%"},
            {"severity": "critical", "target": "root", "msg": "P99 latency > 2000ms (currently {latency}ms)"},
            {"severity": "warning", "target": "affected", "msg": "Increased 4xx responses from downstream dependency"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Connection pool exhausted - cannot acquire connection to {service}-db"},
            {"level": "ERROR", "msg": "Timeout waiting for DB connection after 5000ms"},
            {"level": "WARN", "msg": "Circuit breaker OPEN for {service}-db connections"},
            {"level": "ERROR", "msg": "Failed to process request txn-{txn_id}: DB connection unavailable"},
            {"level": "INFO", "msg": "Retrying DB connection pool initialization"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "{root_service} returned 503 for request req-{req_id}"},
            {"level": "ERROR", "msg": "Failed to complete operation: upstream {root_service} unavailable"},
            {"level": "WARN", "msg": "Fallback: queuing request for retry"},
        ],
        "metric_patterns": {
            "root": [("error_rate", "exponential_rise"), ("latency_p99", "exponential_rise"), ("db_connections", "saturating")],
            "affected": [("error_rate", "exponential_rise")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "db_replication_lag",
        "category": "database",
        "name": "Replication Lag",
        "root_cause_description": "Database replication lag on {service} caused read replicas to serve stale data after {version} schema migration",
        "root_cause_keywords": ["replication", "lag", "stale", "replica", "schema", "migration", "database"],
        "preferred_service_types": ["api", "datastore"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Data inconsistency detected - read replicas {repl_lag}s behind primary"},
            {"severity": "warning", "target": "root", "msg": "Replication lag exceeding 30s threshold"},
            {"severity": "warning", "target": "affected", "msg": "Stale data returned for user queries"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Replication lag at {repl_lag}s - exceeds SLO of 5s"},
            {"level": "WARN", "msg": "Read replica {service}-replica-2 falling behind primary"},
            {"level": "ERROR", "msg": "Schema migration v{version} causing replication bottleneck"},
            {"level": "INFO", "msg": "Attempting to pause non-critical replication streams"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "Inconsistent read from {root_service}: got stale version"},
            {"level": "ERROR", "msg": "Cache invalidation failed: stale data propagated from {root_service}"},
        ],
        "metric_patterns": {
            "root": [("replication_lag_s", "exponential_rise"), ("error_rate", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "restart_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "db_deadlock",
        "category": "database",
        "name": "Deadlock Storm",
        "root_cause_description": "Deadlock storm in {service} database caused by concurrent transaction ordering bug in {version}",
        "root_cause_keywords": ["deadlock", "transaction", "lock", "contention", "database", "concurrent"],
        "preferred_service_types": ["api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Transaction deadlocks spiking - {deadlocks}/min detected"},
            {"severity": "critical", "target": "root", "msg": "Request queue depth exceeding 500 - requests backing up"},
            {"severity": "warning", "target": "affected", "msg": "Upstream {root_service} response time degraded"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Deadlock detected: txn-{txn_id} waiting for lock held by txn-{txn_id2}"},
            {"level": "ERROR", "msg": "Transaction aborted after deadlock timeout: 30s"},
            {"level": "WARN", "msg": "Request queue depth: {queue_depth} - approaching limit"},
            {"level": "ERROR", "msg": "Lock wait timeout exceeded for table {service}_records"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Timeout calling {root_service}: exceeded 10s deadline"},
            {"level": "WARN", "msg": "Circuit breaker half-open for {root_service}"},
        ],
        "metric_patterns": {
            "root": [("deadlocks_per_min", "exponential_rise"), ("error_rate", "exponential_rise"), ("latency_p99", "exponential_rise")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    # ---- MEMORY/CPU (3) ----
    {
        "id": "memory_leak",
        "category": "memory_cpu",
        "name": "Memory Leak (OOM)",
        "root_cause_description": "Memory leak in {service} {version} causing OOM kills - unbounded cache growth in request handler",
        "root_cause_keywords": ["memory", "leak", "OOM", "out of memory", "heap", "cache", "growth"],
        "preferred_service_types": ["api", "worker"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Memory usage at {mem_pct}% - OOM kill imminent"},
            {"severity": "critical", "target": "root", "msg": "Service restarts detected - {restarts} OOM kills in last 10 minutes"},
            {"severity": "warning", "target": "affected", "msg": "Elevated error rate due to {root_service} instability"},
        ],
        "log_patterns_root": [
            {"level": "WARN", "msg": "Heap usage at {mem_pct}% - garbage collection ineffective"},
            {"level": "ERROR", "msg": "Out of memory: process killed by OOM killer (RSS: {rss_mb}MB)"},
            {"level": "INFO", "msg": "Service restarting after OOM kill - attempt {restart_count}"},
            {"level": "WARN", "msg": "Request cache size growing unbounded: {cache_size} entries"},
            {"level": "ERROR", "msg": "Failed to allocate memory for request processing"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Connection refused from {root_service}: service restarting"},
            {"level": "WARN", "msg": "Retrying request to {root_service} after connection failure"},
        ],
        "metric_patterns": {
            "root": [("memory_percent", "saturating"), ("error_rate", "step_function"), ("restarts", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "cpu_spin",
        "category": "memory_cpu",
        "name": "CPU Spin Loop",
        "root_cause_description": "CPU spin loop in {service} {version} caused by infinite retry loop in error handler",
        "root_cause_keywords": ["cpu", "spin", "loop", "100%", "retry", "infinite", "hang"],
        "preferred_service_types": ["api", "worker"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "CPU usage at {cpu_pct}% - all cores saturated"},
            {"severity": "critical", "target": "root", "msg": "Service unresponsive - health check failures"},
            {"severity": "warning", "target": "affected", "msg": "Timeout errors from {root_service} increasing"},
        ],
        "log_patterns_root": [
            {"level": "WARN", "msg": "CPU usage at {cpu_pct}% - throttling requests"},
            {"level": "ERROR", "msg": "Worker thread pool exhausted - all threads blocked"},
            {"level": "ERROR", "msg": "Health check timeout after 5s - service appears hung"},
            {"level": "WARN", "msg": "Error handler retry loop detected: {retry_count} retries in 10s"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "{root_service} health check failed: timeout after 10s"},
            {"level": "WARN", "msg": "Load balancer removed {root_service} from pool"},
        ],
        "metric_patterns": {
            "root": [("cpu_percent", "saturating"), ("latency_p99", "exponential_rise")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "gc_pause",
        "category": "memory_cpu",
        "name": "GC Pause Storm",
        "root_cause_description": "GC pause storm in {service} {version} due to excessive object allocation in batch processor",
        "root_cause_keywords": ["gc", "garbage collection", "pause", "stop-the-world", "allocation", "heap"],
        "preferred_service_types": ["api", "worker"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "P99 latency spiked to {latency}ms - GC pauses detected"},
            {"severity": "warning", "target": "root", "msg": "GC pause time exceeding 2s per cycle"},
            {"severity": "warning", "target": "affected", "msg": "Intermittent timeouts from {root_service}"},
        ],
        "log_patterns_root": [
            {"level": "WARN", "msg": "GC pause duration: {gc_pause}ms (threshold: 500ms)"},
            {"level": "ERROR", "msg": "Stop-the-world GC: {gc_pause}ms - request processing halted"},
            {"level": "WARN", "msg": "Heap occupancy at {mem_pct}% - frequent full GC cycles"},
            {"level": "INFO", "msg": "GC tuning: concurrent mark phase taking {gc_pause}ms"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "Intermittent timeout from {root_service}: latency spike detected"},
            {"level": "ERROR", "msg": "Request to {root_service} timed out during GC pause"},
        ],
        "metric_patterns": {
            "root": [("gc_pause_ms", "exponential_rise"), ("latency_p99", "exponential_rise"), ("memory_percent", "saturating")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy", "scale_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    # ---- NETWORKING (3) ----
    {
        "id": "dns_failure",
        "category": "networking",
        "name": "DNS Resolution Failure",
        "root_cause_description": "DNS resolution failure on {service} after {version} deployment corrupted zone file records",
        "root_cause_keywords": ["dns", "resolution", "zone", "corrupted", "SERVFAIL", "NXDOMAIN", "zone file"],
        "preferred_service_types": ["infra"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "DNS query failure rate at {error_rate}% - resolution failing"},
            {"severity": "critical", "target": "affected", "msg": "Service connectivity lost - DNS resolution failed"},
            {"severity": "warning", "target": "root", "msg": "DNS resolution latency increased to {latency}ms"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "SERVFAIL for {affected_svc}.internal: zone file contains invalid A record"},
            {"level": "ERROR", "msg": "NXDOMAIN for {affected_db}.internal: record missing from {version} zone file"},
            {"level": "ERROR", "msg": "SERVFAIL rate increasing: {error_rate}% of queries failing"},
            {"level": "FATAL", "msg": "Multiple critical DNS records missing/corrupted after {version} zone migration"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Cannot resolve {dep}.internal - DNS lookup failed"},
            {"level": "FATAL", "msg": "Database connection pool empty: all connections failed DNS resolution"},
            {"level": "FATAL", "msg": "Service degraded: critical dependency unreachable via DNS"},
        ],
        "metric_patterns": {
            "root": [("query_failure_rate", "exponential_rise"), ("resolution_latency_ms", "exponential_rise")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "restart_service"],
        "expected_diag_types": ["query_logs", "check_metrics", "view_dependencies"],
        "cascade_pattern": "all_depend_on_root",
    },
    {
        "id": "tls_expiry",
        "category": "networking",
        "name": "TLS Certificate Expiry",
        "root_cause_description": "TLS certificate expired on {service} causing all HTTPS connections to fail",
        "root_cause_keywords": ["tls", "certificate", "expired", "ssl", "https", "cert", "x509"],
        "preferred_service_types": ["api", "infra"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "TLS certificate expired - all HTTPS connections failing"},
            {"severity": "critical", "target": "affected", "msg": "Connection refused: TLS handshake failure with {root_service}"},
            {"severity": "warning", "target": "root", "msg": "Certificate validity: EXPIRED since {expiry_time}"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "TLS handshake failed: certificate expired at {expiry_time}"},
            {"level": "ERROR", "msg": "x509: certificate has expired or is not yet valid"},
            {"level": "WARN", "msg": "Auto-renewal of TLS cert failed: ACME challenge timeout"},
            {"level": "ERROR", "msg": "All incoming HTTPS connections rejected: invalid certificate"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "TLS handshake failed connecting to {root_service}: certificate expired"},
            {"level": "WARN", "msg": "Falling back to cached data - {root_service} unreachable"},
        ],
        "metric_patterns": {
            "root": [("tls_errors", "step_function"), ("error_rate", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "update_config"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "lb_misconfig",
        "category": "networking",
        "name": "Load Balancer Misconfiguration",
        "root_cause_description": "Load balancer misconfiguration on {service} routing all traffic to single unhealthy backend after {version} config change",
        "root_cause_keywords": ["load balancer", "routing", "misconfiguration", "config", "backend", "traffic"],
        "preferred_service_types": ["infra"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Uneven traffic distribution - single backend receiving 100% traffic"},
            {"severity": "critical", "target": "affected", "msg": "Service overloaded - receiving all routed traffic"},
            {"severity": "warning", "target": "root", "msg": "Health check configuration mismatch detected"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Backend pool misconfigured: only 1 of {backend_count} backends active"},
            {"level": "WARN", "msg": "Health check endpoint changed in {version} but LB config not updated"},
            {"level": "ERROR", "msg": "All traffic routed to {affected_svc}: other backends marked unhealthy"},
            {"level": "WARN", "msg": "Backend {affected_svc} reporting 503: overloaded from traffic spike"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Request rate 10x normal - possible routing issue"},
            {"level": "WARN", "msg": "Thread pool exhausted from unexpected traffic volume"},
        ],
        "metric_patterns": {
            "root": [("backend_health_pct", "step_function"), ("error_rate", "step_function")],
            "affected": [("cpu_percent", "saturating"), ("error_rate", "exponential_rise")],
        },
        "valid_remediations": ["update_config", "rollback_deploy", "restart_service"],
        "expected_diag_types": ["query_logs", "check_metrics", "view_dependencies"],
        "cascade_pattern": "downstream",
    },
    # ---- DEPLOYMENT (2) ----
    {
        "id": "bad_config",
        "category": "deployment",
        "name": "Bad Config Rollout",
        "root_cause_description": "Bad configuration rollout to {service} in {version} - invalid database connection string causing all queries to fail",
        "root_cause_keywords": ["config", "configuration", "rollout", "invalid", "connection string", "misconfigured"],
        "preferred_service_types": ["api", "worker"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Error rate at {error_rate}% after config deployment"},
            {"severity": "warning", "target": "root", "msg": "Config refresh failed - serving with invalid config"},
            {"severity": "warning", "target": "affected", "msg": "Upstream dependency {root_service} returning errors"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Database connection failed: invalid connection string in config v{version}"},
            {"level": "ERROR", "msg": "Config validation skipped in deployment pipeline"},
            {"level": "FATAL", "msg": "Cannot initialize connection pool: malformed DSN"},
            {"level": "WARN", "msg": "Rolling back to cached config from previous version"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "{root_service} returning 500 for all requests"},
            {"level": "WARN", "msg": "Dependency {root_service} health check failing"},
        ],
        "metric_patterns": {
            "root": [("error_rate", "step_function"), ("db_connections", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "update_config"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "dep_version_mismatch",
        "category": "deployment",
        "name": "Dependency Version Mismatch",
        "root_cause_description": "Dependency version mismatch: {service} {version} expects API v3 but upstream provides v2, causing deserialization failures",
        "root_cause_keywords": ["version", "mismatch", "api", "incompatible", "deserialization", "contract", "breaking change"],
        "preferred_service_types": ["api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Deserialization errors spiking - {error_rate}% request failure"},
            {"severity": "warning", "target": "root", "msg": "API contract violation detected in responses from upstream"},
            {"severity": "warning", "target": "affected", "msg": "Increased error responses from {root_service}"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Failed to deserialize response from upstream: unexpected field format"},
            {"level": "ERROR", "msg": "API version mismatch: expected v3 response, got v2 schema"},
            {"level": "WARN", "msg": "Contract test failures detected post-deployment of {version}"},
            {"level": "ERROR", "msg": "Null pointer: response missing required field 'metadata.trace_id'"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "{root_service} returning malformed responses"},
            {"level": "ERROR", "msg": "Failed to process response from {root_service}: schema mismatch"},
        ],
        "metric_patterns": {
            "root": [("error_rate", "step_function"), ("deserialization_errors", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "restart_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    # ---- EXTERNAL DEPS (3) ----
    {
        "id": "api_rate_limit",
        "category": "external",
        "name": "Third-Party API Rate Limit",
        "root_cause_description": "Third-party API rate limit hit by {service} due to retry storm from {version} bug - exponential backoff disabled",
        "root_cause_keywords": ["rate limit", "throttle", "429", "third-party", "api", "retry", "backoff"],
        "preferred_service_types": ["api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "External API returning 429 Too Many Requests - rate limited"},
            {"severity": "warning", "target": "root", "msg": "API call volume 5x baseline - possible retry storm"},
            {"severity": "warning", "target": "affected", "msg": "Feature degraded: {root_service} dependency rate-limited"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "External API returned 429: rate limit exceeded (limit: {rate_limit}/min)"},
            {"level": "WARN", "msg": "Retry storm detected: {retry_count} retries in last minute"},
            {"level": "ERROR", "msg": "Exponential backoff disabled in {version}: retrying immediately on 429"},
            {"level": "WARN", "msg": "Circuit breaker opening for external API after {failure_count} failures"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "Feature unavailable: {root_service} cannot reach external API"},
            {"level": "ERROR", "msg": "Fallback response returned: external dependency {root_service} degraded"},
        ],
        "metric_patterns": {
            "root": [("external_api_429s", "exponential_rise"), ("error_rate", "exponential_rise")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "restart_service", "update_config"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "cdn_origin_fail",
        "category": "external",
        "name": "CDN Origin Failure",
        "root_cause_description": "CDN origin failure on {service} - origin server returning 502 after {version} deploy broke health check endpoint",
        "root_cause_keywords": ["cdn", "origin", "502", "cache", "miss", "stale", "health check"],
        "preferred_service_types": ["infra"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "CDN cache miss rate at {cache_miss}% - origin failing"},
            {"severity": "warning", "target": "root", "msg": "CDN serving stale content - origin health check failing"},
            {"severity": "warning", "target": "affected", "msg": "Static asset loading failures reported"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Origin server returning 502 for all cache miss requests"},
            {"level": "WARN", "msg": "CDN health check failing: origin /healthz returning 500"},
            {"level": "ERROR", "msg": "Cache TTL expired - serving stale content as fallback"},
            {"level": "WARN", "msg": "Origin {version} health endpoint path changed from /healthz to /health"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "Static assets failing to load from CDN"},
            {"level": "ERROR", "msg": "API responses missing: CDN origin {root_service} degraded"},
        ],
        "metric_patterns": {
            "root": [("cache_miss_rate", "exponential_rise"), ("origin_error_rate", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["rollback_deploy", "update_config", "restart_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "payment_gw_degradation",
        "category": "external",
        "name": "Payment Gateway Degradation",
        "root_cause_description": "Payment gateway degradation affecting {service} - provider experiencing partial outage causing timeout and retry amplification",
        "root_cause_keywords": ["payment", "gateway", "timeout", "provider", "degradation", "partial outage"],
        "preferred_service_types": ["api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Payment processing failure rate at {error_rate}%"},
            {"severity": "critical", "target": "root", "msg": "Payment gateway timeout rate at {timeout_pct}%"},
            {"severity": "warning", "target": "affected", "msg": "Order completion rate dropping - payment failures"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Payment gateway timeout after 30s - transaction {txn_id} failed"},
            {"level": "WARN", "msg": "Payment gateway health check degraded: 60% success rate"},
            {"level": "ERROR", "msg": "Retry amplification: {retry_count} retries for single payment"},
            {"level": "WARN", "msg": "Thread pool saturated from pending payment gateway calls"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Cannot complete order: payment via {root_service} timed out"},
            {"level": "WARN", "msg": "Cart abandonment rate increasing - payment flow unavailable"},
        ],
        "metric_patterns": {
            "root": [("error_rate", "exponential_rise"), ("latency_p99", "exponential_rise"), ("gateway_timeout_pct", "exponential_rise")],
            "affected": [("error_rate", "exponential_rise")],
        },
        "valid_remediations": ["update_config", "restart_service", "scale_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    # ---- DATA (2) ----
    {
        "id": "cache_poisoning",
        "category": "data",
        "name": "Cache Poisoning",
        "root_cause_description": "Cache poisoning in {service} after {version} deployed incorrect serialization - stale/wrong data served from cache",
        "root_cause_keywords": ["cache", "poisoning", "stale", "serialization", "corrupted", "invalid data"],
        "preferred_service_types": ["datastore", "api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Data integrity alert - inconsistent responses detected"},
            {"severity": "warning", "target": "root", "msg": "Cache hit ratio anomaly - serving corrupted entries"},
            {"severity": "warning", "target": "affected", "msg": "User-facing data inconsistencies reported"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Cache entry validation failed: serialization mismatch in {version}"},
            {"level": "ERROR", "msg": "Corrupted cache entry for key user:{user_id}: invalid JSON"},
            {"level": "WARN", "msg": "Cache hit serving stale data: TTL not respected after {version} update"},
            {"level": "WARN", "msg": "Cache warm-up populating entries with incorrect format"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Received malformed data from {root_service} cache"},
            {"level": "WARN", "msg": "Data validation failed: response from {root_service} does not match schema"},
        ],
        "metric_patterns": {
            "root": [("cache_error_rate", "step_function"), ("data_integrity_errors", "exponential_rise")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "rollback_deploy", "update_config"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "queue_backlog",
        "category": "data",
        "name": "Queue Backlog Overflow",
        "root_cause_description": "Message queue backlog overflow on {service} after consumer {version} deploy introduced processing bug causing consumer lag",
        "root_cause_keywords": ["queue", "backlog", "overflow", "consumer", "lag", "message", "processing"],
        "preferred_service_types": ["infra", "worker"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Queue depth at {queue_depth} messages - backlog growing"},
            {"severity": "critical", "target": "root", "msg": "Consumer lag exceeding {lag}s - messages not being processed"},
            {"severity": "warning", "target": "affected", "msg": "Async operations delayed - queue backpressure detected"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Queue depth exceeding limit: {queue_depth} pending messages"},
            {"level": "WARN", "msg": "Consumer group {service}-consumers: all partitions lagging"},
            {"level": "ERROR", "msg": "Message processing failure rate at {error_rate}%: {version} deserialization bug"},
            {"level": "WARN", "msg": "Dead letter queue filling up: {dlq_count} messages in last 5 minutes"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "Async notification delayed: queue backpressure from {root_service}"},
            {"level": "ERROR", "msg": "Event processing timeout: {root_service} consumer lag at {lag}s"},
        ],
        "metric_patterns": {
            "root": [("queue_depth", "exponential_rise"), ("consumer_lag_s", "exponential_rise"), ("error_rate", "exponential_rise")],
            "affected": [("processing_delay_s", "exponential_rise")],
        },
        "valid_remediations": ["rollback_deploy", "restart_service", "scale_service"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    # ---- SECURITY (2) ----
    {
        "id": "cred_rotation_fail",
        "category": "security",
        "name": "Credential Rotation Failure",
        "root_cause_description": "Credential rotation failure on {service} - automated rotation completed but {version} still using old credentials",
        "root_cause_keywords": ["credential", "rotation", "auth", "password", "expired", "secret", "token"],
        "preferred_service_types": ["api", "infra"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "Authentication failures at {error_rate}% - credential mismatch"},
            {"severity": "warning", "target": "root", "msg": "Database auth rejected: invalid credentials"},
            {"severity": "warning", "target": "affected", "msg": "Upstream {root_service} authentication failing"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "Authentication failed: credentials expired after rotation at {rotation_time}"},
            {"level": "ERROR", "msg": "Database connection rejected: password authentication failed"},
            {"level": "WARN", "msg": "Secret manager returned new credentials but {service} cache not refreshed"},
            {"level": "ERROR", "msg": "All connection attempts failing: 401 Unauthorized"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "{root_service} returning 401 for all requests"},
            {"level": "WARN", "msg": "Service degraded: authentication dependency {root_service} failing"},
        ],
        "metric_patterns": {
            "root": [("auth_failures", "step_function"), ("error_rate", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["restart_service", "update_config"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
    {
        "id": "waf_false_positive",
        "category": "security",
        "name": "WAF False Positives",
        "root_cause_description": "WAF false positives on {service} blocking legitimate traffic after {version} rule update matched common request patterns",
        "root_cause_keywords": ["waf", "firewall", "blocked", "false positive", "rule", "403", "security"],
        "preferred_service_types": ["infra", "api"],
        "alert_patterns": [
            {"severity": "critical", "target": "root", "msg": "403 error rate at {error_rate}% - WAF blocking legitimate traffic"},
            {"severity": "warning", "target": "root", "msg": "WAF rule {rule_id} triggering on {block_rate}% of requests"},
            {"severity": "warning", "target": "affected", "msg": "Users reporting access denied errors"},
        ],
        "log_patterns_root": [
            {"level": "ERROR", "msg": "WAF rule {rule_id} blocking request: false positive on content-type header"},
            {"level": "WARN", "msg": "Rule {version} matching legitimate API requests: {block_rate}% block rate"},
            {"level": "ERROR", "msg": "User-agent pattern match too broad: blocking Chrome/Safari browsers"},
            {"level": "WARN", "msg": "WAF audit log shows {block_count} blocks in last 5 minutes"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "Requests to {root_service} being blocked by WAF"},
            {"level": "WARN", "msg": "API calls failing with 403: WAF rule match on {root_service}"},
        ],
        "metric_patterns": {
            "root": [("waf_blocks", "step_function"), ("error_rate_403", "step_function")],
            "affected": [("error_rate", "step_function")],
        },
        "valid_remediations": ["update_config", "rollback_deploy"],
        "expected_diag_types": ["query_logs", "check_metrics"],
        "cascade_pattern": "downstream",
    },
]

# ---------------------------------------------------------------------------
# Red Herring Templates
# ---------------------------------------------------------------------------

RED_HERRING_ALERTS = [
    {"severity": "info", "msg": "Scheduled maintenance window starting in 2 hours"},
    {"severity": "info", "msg": "New dashboard version deployed - no action required"},
    {"severity": "info", "msg": "Elasticsearch reindex started - expected temporary latency increase"},
    {"severity": "info", "msg": "Model v4.2 deployment started - canary rollout in progress"},
    {"severity": "warning", "msg": "CPU usage at {cpu_pct}% - approaching threshold"},
    {"severity": "warning", "msg": "Disk usage at 72% on data volume"},
    {"severity": "info", "msg": "SSL certificate renewal scheduled for next week"},
    {"severity": "warning", "msg": "Log volume increased 20% - review retention policy"},
    {"severity": "info", "msg": "Automated backup completed successfully"},
    {"severity": "info", "msg": "Canary deployment proceeding as expected"},
    {"severity": "warning", "msg": "Memory usage at 68% - within normal range but trending up"},
    {"severity": "info", "msg": "Feature flag 'new-checkout-flow' enabled for 5% of users"},
]

RED_HERRING_LOGS = [
    {"level": "INFO", "msg": "Healthy - serving requests normally"},
    {"level": "INFO", "msg": "Cache hit ratio: {hit_ratio}%"},
    {"level": "INFO", "msg": "Batch job completed: processed {batch_count} records"},
    {"level": "INFO", "msg": "Scheduled reindex 40% complete - elevated latency expected"},
    {"level": "INFO", "msg": "Model serving normally - no dependency issues"},
    {"level": "WARN", "msg": "Slow query detected (non-critical): SELECT took 2.1s"},
]

RED_HERRING_DEPLOYMENTS = [
    {"deployer": "ml-team", "note": "model update"},
    {"deployer": "search-team", "note": "index rebuild"},
    {"deployer": "ops-team", "note": "monitoring update"},
    {"deployer": "dev-team", "note": "feature flag change"},
]

# ---------------------------------------------------------------------------
# Curve Generators
# ---------------------------------------------------------------------------

def exponential_rise(start: float, end: float, steps: int) -> list[float]:
    """Generate exponentially rising values."""
    if steps <= 1:
        return [end]
    ratio = (end / max(start, 0.01)) ** (1.0 / (steps - 1))
    return [round(start * (ratio ** i), 1) for i in range(steps)]


def saturating_curve(max_val: float, steps: int) -> list[float]:
    """Generate values approaching a maximum (1 - e^-x curve)."""
    return [round(max_val * (1 - math.exp(-3 * i / max(steps - 1, 1))), 1) for i in range(steps)]


def step_function(before: float, after: float, steps: int) -> list[float]:
    """Generate values with a sudden jump at the midpoint."""
    mid = steps // 2
    return [round(before, 1)] * mid + [round(after, 1)] * (steps - mid)


def noise_series(baseline: float, std: float, steps: int, rng: random.Random) -> list[float]:
    """Generate noisy values around a baseline."""
    return [round(max(0, baseline + rng.gauss(0, std)), 1) for _ in range(steps)]


def generate_curve(curve_type: str, difficulty: str, rng: random.Random) -> list[float]:
    """Generate a metric time series based on curve type and difficulty."""
    steps = rng.randint(5, 8)
    if curve_type == "exponential_rise":
        start = rng.uniform(0.1, 2.0)
        end = rng.uniform(30, 100)
        return exponential_rise(start, end, steps)
    elif curve_type == "saturating":
        max_val = rng.uniform(80, 100)
        return saturating_curve(max_val, steps)
    elif curve_type == "step_function":
        before = rng.uniform(0, 2)
        after = rng.uniform(40, 100)
        return step_function(before, after, steps)
    else:
        return noise_series(rng.uniform(20, 60), 5, steps, rng)


# ---------------------------------------------------------------------------
# Timestamp Generator
# ---------------------------------------------------------------------------

def make_timestamps(base_hour: int, base_min: int, count: int, spread_min: int = 10) -> list[str]:
    """Generate ISO timestamps spread over a period."""
    ts = []
    for i in range(count):
        offset = i * (spread_min * 60 // max(count, 1))
        m = base_min + offset // 60
        s = offset % 60
        h = base_hour + m // 60
        m = m % 60
        ts.append(f"2026-03-26T{h:02d}:{m:02d}:{s:02d}Z")
    return ts


# ---------------------------------------------------------------------------
# Scenario Builder
# ---------------------------------------------------------------------------

def pick_services(template: dict, difficulty: str, rng: random.Random) -> tuple[dict, list[dict]]:
    """Select root cause service and supporting services from the pool."""
    profile = DIFFICULTY_PROFILES[difficulty]
    num = rng.randint(*profile["num_services"])

    # Pick root cause service matching template preference
    preferred = template.get("preferred_service_types", ["api"])
    candidates = [s for s in SERVICE_POOL if s["type"] in preferred]
    if not candidates:
        candidates = SERVICE_POOL[:]
    root_svc = rng.choice(candidates)

    # Pick other services (diverse types)
    others = [s for s in SERVICE_POOL if s["name"] != root_svc["name"]]
    rng.shuffle(others)
    selected = others[:num - 1]

    return root_svc, selected


def build_dependencies(root_svc: dict, other_svcs: list[dict], template: dict, rng: random.Random) -> dict:
    """Build dependency graph for the scenario."""
    deps = {}
    root_name = root_svc["name"]

    if template.get("cascade_pattern") == "all_depend_on_root":
        # Root is infrastructure that all others depend on
        deps[root_name] = []
        for svc in other_svcs:
            deps[svc["name"]] = [f"{svc['name']}-db.internal", root_name]
    else:
        # Downstream cascade: root has its own deps, others depend on root
        deps[root_name] = [f"{root_name}-db", f"{root_name}-store"]
        for i, svc in enumerate(other_svcs):
            svc_deps = []
            if i < len(other_svcs) // 2:
                svc_deps.append(root_name)
            svc_deps.append(f"{svc['name']}-db")
            deps[svc["name"]] = svc_deps

    return deps


def compute_grading_weights(difficulty: str, rng: random.Random) -> dict:
    """Compute grading weights with per-scenario jitter."""
    base = WEIGHT_PROFILES[difficulty].copy()
    keys = ["triage", "diagnostic", "root_cause", "remediation", "efficiency", "documentation"]

    # Add jitter
    jittered = {}
    for k in keys:
        val = base[k] + rng.uniform(-0.03, 0.03)
        jittered[k] = max(0.02, min(0.40, val))

    # Normalize to sum to 1.0, adjusting last weight to absorb rounding error
    total = sum(jittered.values())
    result = {}
    running = 0.0
    for i, k in enumerate(keys):
        if i < len(keys) - 1:
            val = round(jittered[k] / total, 2)
            result[f"{k}_weight"] = val
            running += val
        else:
            result[f"{k}_weight"] = round(1.0 - running, 2)
    return result


def build_scenario(template: dict, difficulty: str, seed: int, scenario_num: int) -> dict:
    """Build a complete scenario from a template."""
    rng = random.Random(seed)
    profile = DIFFICULTY_PROFILES[difficulty]

    # Select services
    root_svc, other_svcs = pick_services(template, difficulty, rng)
    all_svcs = [root_svc] + other_svcs
    root_name = root_svc["name"]

    # Assign versions
    root_version = rng.choice(SERVICE_VERSIONS)
    svc_versions = {root_name: root_version}
    for s in other_svcs:
        svc_versions[s["name"]] = rng.choice(SERVICE_VERSIONS)

    # Dependencies
    dependencies = build_dependencies(root_svc, other_svcs, template, rng)

    # Determine affected services (those that depend on root)
    affected = []
    for svc in other_svcs:
        svc_deps = dependencies.get(svc["name"], [])
        if root_name in svc_deps or template.get("cascade_pattern") == "all_depend_on_root":
            affected.append(svc)

    cascade_depth = rng.randint(*profile["cascade_depth"])
    affected = affected[:cascade_depth] if affected else other_svcs[:1]
    healthy = [s for s in other_svcs if s not in affected]

    # Build incident ID
    incident_id = f"INC-20260326-{scenario_num:03d}"

    # Base timestamps
    base_hour = rng.randint(8, 22)
    base_min = rng.randint(0, 50)

    # --- Initial Alerts ---
    alerts = []
    alert_counter = scenario_num * 100
    num_alerts = rng.randint(*profile["num_alerts"])
    num_critical = rng.randint(*profile["critical_alerts"])
    num_noise = rng.randint(*profile["noise_alerts"])

    # Template-based alerts
    for ap in template["alert_patterns"]:
        alert_counter += 1
        target = root_name if ap["target"] == "root" else (affected[0]["name"] if affected else root_name)
        fmt_vars = dict(
            error_rate=round(rng.uniform(10, 80), 1),
            latency=rng.randint(1000, 8000),
            repl_lag=rng.randint(30, 120),
            deadlocks=rng.randint(50, 500),
            mem_pct=rng.randint(85, 98),
            cpu_pct=rng.randint(90, 100),
            cache_miss=rng.randint(60, 95),
            timeout_pct=rng.randint(40, 80),
            queue_depth=rng.randint(10000, 100000),
            lag=rng.randint(60, 600),
            block_rate=rng.randint(30, 80),
            rule_id=f"WAF-{rng.randint(1000,9999)}",
            root_service=root_name,
            restarts=rng.randint(3, 15),
            expiry_time=f"2026-03-26T{max(base_hour-2,0):02d}:00:00Z",
        )
        msg = ap["msg"].format(**fmt_vars)
        sev = ap["severity"]
        ts_offset = alert_counter % 10
        alerts.append({
            "alert_id": f"alert-{alert_counter}",
            "severity": sev,
            "service": target,
            "message": msg,
            "timestamp": f"2026-03-26T{base_hour:02d}:{base_min + ts_offset % 5:02d}:{ts_offset * 5 % 60:02d}Z",
            "acknowledged": False,
            "silenced": False,
        })

    # Extra critical alerts on affected services
    for i in range(min(num_critical, len(affected))):
        alert_counter += 1
        svc = affected[i]
        alerts.append({
            "alert_id": f"alert-{alert_counter}",
            "severity": "critical",
            "service": svc["name"],
            "message": f"{svc['name']} error rate spiking - requests failing",
            "timestamp": f"2026-03-26T{base_hour:02d}:{(base_min + 1) % 60:02d}:{rng.randint(0,59):02d}Z",
            "acknowledged": False,
            "silenced": False,
        })

    # Noise/info alerts
    noise_svcs = healthy if healthy else other_svcs
    for i in range(num_noise):
        alert_counter += 1
        svc = rng.choice(noise_svcs) if noise_svcs else rng.choice(all_svcs)
        rh = rng.choice(RED_HERRING_ALERTS)
        msg = rh["msg"].format(cpu_pct=rng.randint(60, 80))
        alerts.append({
            "alert_id": f"alert-{alert_counter}",
            "severity": rh["severity"],
            "service": svc["name"],
            "message": msg,
            "timestamp": f"2026-03-26T{base_hour:02d}:{(base_min - rng.randint(1,10)) % 60:02d}:{rng.randint(0,59):02d}Z",
            "acknowledged": False,
            "silenced": False,
        })

    # Trim/pad to target alert count
    alerts = alerts[:max(num_alerts, len(alerts))]

    # --- Services list ---
    services = []
    # Root cause service: degraded or down
    root_error_rate = round(rng.uniform(15, 100), 1)
    root_status = "down" if root_error_rate > 70 else "degraded"
    root_latency = 0 if root_status == "down" else rng.randint(1000, 8000)
    services.append({
        "name": root_name,
        "status": root_status,
        "latency_ms": root_latency,
        "error_rate": root_error_rate,
        "cpu_percent": rng.randint(40, 98),
        "memory_percent": rng.randint(40, 95),
        "version": root_version,
    })

    # Affected services: degraded
    for svc in affected:
        err = round(rng.uniform(10, 80), 1)
        services.append({
            "name": svc["name"],
            "status": "down" if err > 70 else "degraded",
            "latency_ms": 0 if err > 70 else rng.randint(500, 5000),
            "error_rate": err,
            "cpu_percent": rng.randint(20, 80),
            "memory_percent": rng.randint(30, 70),
            "version": svc_versions[svc["name"]],
        })

    # Healthy services
    for svc in healthy:
        services.append({
            "name": svc["name"],
            "status": "healthy",
            "latency_ms": rng.randint(20, 200),
            "error_rate": round(rng.uniform(0, 1.0), 1),
            "cpu_percent": rng.randint(5, 50),
            "memory_percent": rng.randint(15, 55),
            "version": svc_versions[svc["name"]],
        })

    # --- Recent Deployments ---
    recent_deployments = [
        {
            "service": root_name,
            "version": root_version,
            "timestamp": f"2026-03-26T{max(base_hour - 1, 0):02d}:{rng.randint(0,59):02d}:00Z",
            "deployer": rng.choice(DEPLOYERS),
        }
    ]
    # Red herring deployments for harder difficulties
    num_rh = rng.randint(*profile["red_herrings"])
    rh_svcs = rng.sample(healthy, min(num_rh, len(healthy))) if healthy else []
    for svc in rh_svcs:
        rhd = rng.choice(RED_HERRING_DEPLOYMENTS)
        recent_deployments.append({
            "service": svc["name"],
            "version": svc_versions[svc["name"]],
            "timestamp": f"2026-03-26T{max(base_hour - rng.randint(1,3), 0):02d}:{rng.randint(0,59):02d}:00Z",
            "deployer": rhd["deployer"],
        })

    # --- Logs ---
    logs = {}
    num_logs = rng.randint(*profile["logs_per_service"])

    # Root cause logs
    root_logs = []
    root_ts = make_timestamps(base_hour, base_min, num_logs)
    for i, lp in enumerate(template["log_patterns_root"][:num_logs]):
        msg = lp["msg"].format(
            service=root_name, version=root_version,
            txn_id=rng.randint(10000, 99999), txn_id2=rng.randint(10000, 99999),
            req_id=rng.randint(10000, 99999), mem_pct=rng.randint(85, 98),
            rss_mb=rng.randint(2000, 8000), restart_count=rng.randint(1, 5),
            cache_size=rng.randint(100000, 999999), cpu_pct=rng.randint(90, 100),
            retry_count=rng.randint(100, 5000), gc_pause=rng.randint(500, 5000),
            error_rate=round(rng.uniform(10, 80), 1), repl_lag=rng.randint(30, 120),
            queue_depth=rng.randint(10000, 100000), lag=rng.randint(60, 600),
            dlq_count=rng.randint(100, 5000), user_id=rng.randint(1000, 99999),
            rotation_time=f"2026-03-26T{max(base_hour-1,0):02d}:00:00Z",
            expiry_time=f"2026-03-26T{max(base_hour-2,0):02d}:00:00Z",
            rule_id=f"WAF-{rng.randint(1000,9999)}", block_rate=rng.randint(30, 80),
            block_count=rng.randint(500, 5000), rate_limit=rng.randint(100, 1000),
            failure_count=rng.randint(50, 500), backend_count=rng.randint(3, 8),
            affected_svc=affected[0]["name"] if affected else root_name,
            affected_db=f"{affected[0]['name']}-db" if affected else f"{root_name}-db",
            dep=f"{root_name}-db", root_service=root_name, batch_count=rng.randint(1000, 50000),
            hit_ratio=rng.randint(30, 60),
        )
        ts = root_ts[i] if i < len(root_ts) else root_ts[-1]
        root_logs.append({
            "timestamp": ts,
            "service": root_name,
            "level": lp["level"],
            "message": msg,
        })
    logs[root_name] = root_logs

    # Affected service logs
    for svc in affected:
        svc_logs = []
        num_aff_logs = min(rng.randint(2, 4), len(template["log_patterns_affected"]))
        aff_ts = make_timestamps(base_hour, base_min + 1, num_aff_logs)
        for i in range(num_aff_logs):
            lp = template["log_patterns_affected"][i % len(template["log_patterns_affected"])]
            msg = lp["msg"].format(
                root_service=root_name, req_id=rng.randint(10000, 99999),
                dep=f"{svc['name']}-db.internal",
            )
            svc_logs.append({
                "timestamp": aff_ts[i] if i < len(aff_ts) else aff_ts[-1],
                "service": svc["name"],
                "level": lp["level"],
                "message": msg,
            })
        logs[svc["name"]] = svc_logs

    # Healthy service logs (red herrings for harder difficulties)
    if num_rh > 0 and healthy:
        for svc in rng.sample(healthy, min(num_rh, len(healthy))):
            rh_log = rng.choice(RED_HERRING_LOGS)
            logs[svc["name"]] = [{
                "timestamp": f"2026-03-26T{base_hour:02d}:{base_min:02d}:00Z",
                "service": svc["name"],
                "level": rh_log["level"],
                "message": rh_log["msg"].format(
                    hit_ratio=rng.randint(85, 98),
                    batch_count=rng.randint(1000, 50000),
                ),
            }]

    # --- Metrics ---
    metrics = {}
    for metric_name, curve_type in template["metric_patterns"].get("root", []):
        key = f"{root_name}:{metric_name}"
        metrics[key] = generate_curve(curve_type, difficulty, rng)

    for svc in affected:
        for metric_name, curve_type in template["metric_patterns"].get("affected", []):
            key = f"{svc['name']}:{metric_name}"
            metrics[key] = generate_curve(curve_type, difficulty, rng)

    # Healthy service metrics (noise)
    for svc in healthy[:2]:
        metrics[f"{svc['name']}:cpu"] = noise_series(rng.uniform(20, 50), 5, 6, rng)

    # --- Root Cause ---
    root_cause = {
        "service": root_name,
        "description": template["root_cause_description"].format(
            service=root_name, version=root_version,
        ),
        "keywords": template["root_cause_keywords"][:],
    }

    # --- Valid Remediations ---
    valid_remediations = []
    for action in template["valid_remediations"]:
        valid_remediations.append({"action": action, "service": root_name})

    # --- Expected Diagnostics ---
    expected_diagnostics = []
    diag_types = template["expected_diag_types"]
    for dt in diag_types:
        if dt in ("query_logs", "check_metrics"):
            expected_diagnostics.append({
                "action_type": dt,
                "params": {"service": root_name},
            })
        elif dt == "view_dependencies":
            if affected:
                expected_diagnostics.append({
                    "action_type": "view_dependencies",
                    "params": {"service_name": affected[0]["name"]},
                })

    # Add diagnostics for affected services in harder difficulties
    num_diags = rng.randint(*profile["expected_diags"])
    for svc in affected[:num_diags - len(expected_diagnostics)]:
        expected_diagnostics.append({
            "action_type": "query_logs",
            "params": {"service": svc["name"]},
        })

    # --- Grading Rubric ---
    grading_rubric = compute_grading_weights(difficulty, rng)

    # --- Description ---
    sev = SEVERITY_MAP[difficulty]
    desc_parts = [
        f"{sev} incident: {template['name']} affecting {root_name}.",
    ]
    if len(affected) > 0:
        aff_names = ", ".join(s["name"] for s in affected)
        desc_parts.append(f"Cascading impact on {aff_names}.")
    if difficulty in ("hard", "expert"):
        desc_parts.append("Multiple concurrent events make diagnosis challenging.")
    if difficulty == "expert":
        desc_parts.append("Red herrings present - focus on evidence-based investigation.")
    description = " ".join(desc_parts)

    return {
        "incident_id": incident_id,
        "difficulty": difficulty,
        "description": description,
        "expected_severity": sev,
        "initial_alerts": alerts,
        "services": services,
        "recent_deployments": recent_deployments,
        "logs": logs,
        "metrics": metrics,
        "dependencies": dependencies,
        "root_cause": root_cause,
        "valid_remediations": valid_remediations,
        "expected_diagnostics": expected_diagnostics,
        "grading_rubric": grading_rubric,
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "incident_id", "difficulty", "description", "expected_severity",
    "initial_alerts", "services", "recent_deployments", "logs", "metrics",
    "dependencies", "root_cause", "valid_remediations", "expected_diagnostics",
    "grading_rubric",
}

SERVICE_REQUIRED_FIELDS = {"name", "status", "latency_ms", "error_rate",
                           "cpu_percent", "memory_percent", "version"}


def validate_scenario(scenario: dict) -> list[str]:
    """Validate a scenario dict, returning list of errors (empty = valid)."""
    errors = []

    # 1. Required top-level keys
    missing = REQUIRED_KEYS - set(scenario.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")

    # 2. Grading rubric weights sum to ~1.0
    rubric = scenario.get("grading_rubric", {})
    weight_sum = sum(rubric.values())
    if abs(weight_sum - 1.0) > 0.02:
        errors.append(f"Grading weights sum to {weight_sum}, expected ~1.0")

    # 3. Service names set
    svc_names = {s["name"] for s in scenario.get("services", [])}

    # 4. Root cause service exists
    rc_svc = scenario.get("root_cause", {}).get("service")
    if rc_svc and rc_svc not in svc_names:
        errors.append(f"Root cause service '{rc_svc}' not in services")

    # 5. Valid remediations reference existing services
    for rem in scenario.get("valid_remediations", []):
        if rem.get("service") not in svc_names:
            errors.append(f"Remediation service '{rem.get('service')}' not in services")

    # 6. Expected diagnostics reference existing services
    for diag in scenario.get("expected_diagnostics", []):
        svc = diag.get("params", {}).get("service") or diag.get("params", {}).get("service_name")
        if svc and svc not in svc_names:
            errors.append(f"Diagnostic service '{svc}' not in services")

    # 7. Alerts reference existing services
    for alert in scenario.get("initial_alerts", []):
        if alert.get("service") not in svc_names:
            errors.append(f"Alert service '{alert.get('service')}' not in services")

    # 8. Dependencies keys reference existing services
    for dep_svc in scenario.get("dependencies", {}):
        if dep_svc not in svc_names:
            errors.append(f"Dependency key '{dep_svc}' not in services")

    # 9. At least one critical alert
    has_critical = any(a.get("severity") == "critical" for a in scenario.get("initial_alerts", []))
    if not has_critical:
        errors.append("No critical alerts found")

    # 10. Metrics keys format
    for key in scenario.get("metrics", {}):
        parts = key.split(":")
        if len(parts) != 2:
            errors.append(f"Metric key '{key}' not in service:metric format")
        elif parts[0] not in svc_names:
            errors.append(f"Metric service '{parts[0]}' not in services")

    # 11. Log keys are valid service names
    for log_svc in scenario.get("logs", {}):
        if log_svc not in svc_names:
            errors.append(f"Log key '{log_svc}' not in services")

    # 12. Difficulty is valid
    if scenario.get("difficulty") not in DIFFICULTIES:
        errors.append(f"Invalid difficulty: {scenario.get('difficulty')}")

    # 13. Each service has required fields
    for svc in scenario.get("services", []):
        missing_fields = SERVICE_REQUIRED_FIELDS - set(svc.keys())
        if missing_fields:
            errors.append(f"Service '{svc.get('name')}' missing fields: {missing_fields}")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate OnCallEnv incident scenarios")
    parser.add_argument("--count", type=int, default=11,
                        help="Number of NEW scenarios per difficulty level (default: 11, +1 original = 12)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all generated scenarios")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    args = parser.parse_args()

    num_templates = len(INCIDENT_TEMPLATES)
    total_generated = 0
    total_errors = 0

    for difficulty in DIFFICULTIES:
        task_dir = TASK_DIRS[difficulty]
        out_dir = SCENARIOS_DIR / task_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove old generated scenario files (preserve hand-crafted scenario_001 and scenario_002)
        for old in out_dir.glob("scenario_*.json"):
            # Keep original scenario_001.json (tests depend on its hardcoded alert IDs/services)
            if old.name == "scenario_001.json":
                continue
            old.unlink()

        print(f"\n--- Generating {args.count} {difficulty} scenarios (002-{args.count+1:03d}) ---")

        # Keep scenario_001 as-is, generate scenario_002 through scenario_{count+1}
        for i in range(args.count):
            file_num = i + 2  # start at 002
            # Round-robin template assignment ensures diversity
            template = INCIDENT_TEMPLATES[i % num_templates]
            seed = args.seed * 1000 + DIFFICULTIES.index(difficulty) * 100 + i
            scenario_num = DIFFICULTIES.index(difficulty) * 100 + file_num

            scenario = build_scenario(template, difficulty, seed, scenario_num)

            # Validate
            errs = validate_scenario(scenario)
            if errs:
                print(f"  ERRORS in scenario_{file_num:03d}.json: {errs}")
                total_errors += len(errs)
            else:
                status = "OK"
                print(f"  scenario_{file_num:03d}.json [{template['id']}] - {status}")

            # Write
            path = out_dir / f"scenario_{file_num:03d}.json"
            with open(path, "w") as f:
                json.dump(scenario, f, indent=2)
                f.write("\n")

            total_generated += 1

    print(f"\n{'='*50}")
    print(f"Generated {total_generated} scenarios across {len(DIFFICULTIES)} difficulty levels")
    print(f"Validation errors: {total_errors}")

    if args.validate:
        print(f"\n--- Running full validation pass ---")
        val_errors = 0
        for difficulty in DIFFICULTIES:
            task_dir = TASK_DIRS[difficulty]
            out_dir = SCENARIOS_DIR / task_dir
            for path in sorted(out_dir.glob("scenario_*.json")):
                with open(path) as f:
                    scenario = json.load(f)
                errs = validate_scenario(scenario)
                if errs:
                    print(f"  FAIL {path.name}: {errs}")
                    val_errors += len(errs)
                else:
                    print(f"  PASS {path.name}")
        if val_errors == 0:
            print("\nAll scenarios passed validation!")
        else:
            print(f"\n{val_errors} validation errors found!")
            sys.exit(1)

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
