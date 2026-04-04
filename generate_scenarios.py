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


def rand_ip(rng: random.Random) -> str:
    """Generate a random internal 10.x.x.x IP address."""
    return f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"


def rand_port(rng: random.Random, base: int = 5432) -> int:
    """Generate a random port near a base port."""
    return base + rng.randint(0, 20)

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
            {"level": "ERROR", "msg": "c.a.s.db.HikariPool [pool-3-thread-{thread_n}] - Connection is not available, request timed out after {timeout}ms. (total={pool_max}, active={pool_max}, idle=0, waiting={waiting})"},
            {"level": "ERROR", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n}] - java.net.SocketTimeoutException: connect timed out to {db_ip}:{db_port}"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-monitor-1] - Circuit breaker '{dep}' state: CLOSED -> OPEN (failures={failure_count}/{window})"},
            {"level": "ERROR", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n2}] - java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available, request timed out after {timeout}ms"},
            {"level": "INFO", "msg": "c.a.s.db.HikariPool [pool-3-housekeeper] - HikariPool-1 - Pool stats (total={pool_max}, active={pool_max}, idle=0, waiting={waiting})"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "upstream connect error: connection refused to {root_ip}:{root_port}, reset reason: remote reset"},
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 503 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-{root_service}] - state: CLOSED -> OPEN after {failure_count} consecutive failures"},
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
            {"level": "ERROR", "msg": "c.a.s.db.ReplicaMonitor [replica-health-1] - replica {replica_ip}:{db_port} lag: {repl_lag}s (threshold: 5s)"},
            {"level": "WARN", "msg": "c.a.s.db.ReadRouter [read-pool-{thread_n}] - stale read detected: row version {row_ver_old} != primary version {row_ver_new}"},
            {"level": "ERROR", "msg": "c.a.s.db.ReplicaMonitor [replica-health-1] - replica {replica_ip2}:{db_port} lag: {repl_lag2}s (threshold: 5s)"},
            {"level": "INFO", "msg": "c.a.s.db.ReplicationManager [repl-mgr-1] - pausing non-critical replication streams to {replica_ip}:{db_port}"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} but response body contains stale entity version {row_ver_old}"},
            {"level": "ERROR", "msg": "c.a.s.cache.Invalidator [cache-inv-1] - cache entry stale: key={cache_key} local_ver={row_ver_old} remote_ver={row_ver_new}"},
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
            {"level": "ERROR", "msg": "c.a.s.db.DeadlockDetector [monitor-1] - deadlock detected: tx-{txn_id} holds lock on {service}_records.row_id={row_id1}, waiting for {service}_index.row_id={row_id2}; tx-{txn_id2} holds {service}_index.row_id={row_id2}, waiting for {service}_records.row_id={row_id1}"},
            {"level": "ERROR", "msg": "c.a.s.db.TransactionManager [tx-pool-{thread_n}] - Lock wait timeout after {timeout}ms on table '{service}_records' row_id={row_id1}"},
            {"level": "WARN", "msg": "c.a.s.db.ConnectionPool [pool-monitor-1] - {waiting} threads waiting for connection, active={pool_max}/{pool_max}"},
            {"level": "ERROR", "msg": "c.a.s.db.TransactionManager [tx-pool-{thread_n2}] - Lock wait timeout after {timeout}ms on table '{service}_index' row_id={row_id2}"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-{root_service}] - state: HALF_OPEN -> OPEN after probe failure (HTTP 504)"},
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
            {"level": "WARN", "msg": "GC overhead: pause={gc_pause}ms heap={heap_used}M/{heap_max}M eden=0B survivor={survivor}M old={old_gen}M"},
            {"level": "ERROR", "msg": "kernel: [{kern_ts}] Out of memory: Killed process {pid} ({service}) total-vm:{vm_kb}kB anon-rss:{rss_kb}kB oom_score_adj:999"},
            {"level": "WARN", "msg": "c.a.s.cache.Manager [cache-evictor-1] - heap pressure: {mem_pct}% used, eviction rate: {evict_rate}/s"},
            {"level": "WARN", "msg": "c.a.s.gc.Monitor [gc-monitor-1] - Full GC #{gc_count}: {gc_pause}ms pause, freed {freed}M, heap {heap_used}M/{heap_max}M"},
            {"level": "ERROR", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n}] - java.lang.OutOfMemoryError: Java heap space"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "upstream connect error: connection refused to {root_ip}:{root_port}, reset reason: connection failure"},
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 503 from {root_ip}:{root_port} (latency={aff_latency}ms), retrying in {backoff}ms"},
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
            {"level": "WARN", "msg": "c.a.s.retry.ExponentialBackoff [retry-pool-{thread_n}] - attempt {retry_attempt}/{retry_max} for operation '{op_name}' (elapsed: {elapsed}ms)"},
            {"level": "ERROR", "msg": "c.a.s.monitor.ThreadDump [watchdog-1] - thread 'http-nio-8080-exec-{thread_n2}' in RUNNABLE state for {stuck_secs}s, stack: c.a.s.handler.RequestHandler.processRetry(line:{line_num})"},
            {"level": "ERROR", "msg": "c.a.s.monitor.ThreadDump [watchdog-1] - {stuck_threads}/{pool_max} threads in RUNNABLE state, CPU {cpu_pct}%"},
            {"level": "WARN", "msg": "c.a.s.retry.ExponentialBackoff [retry-pool-{thread_n}] - attempt {retry_max}/{retry_max} for operation '{op_name}' (elapsed: {elapsed2}ms), max retries exhausted"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {root_ip}:{root_port} (latency=10003ms)"},
            {"level": "WARN", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {root_ip}:{root_port} health check failed: timeout after 5000ms"},
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
            {"level": "WARN", "msg": "GC overhead: pause={gc_pause}ms heap={heap_used}M/{heap_max}M eden=0B survivor={survivor}M old={old_gen}M"},
            {"level": "ERROR", "msg": "c.a.s.gc.Monitor [gc-monitor-1] - Full GC #{gc_count}: {gc_pause}ms pause, freed {freed}M, heap {heap_used}M/{heap_max}M"},
            {"level": "WARN", "msg": "c.a.s.gc.Monitor [gc-monitor-1] - GC overhead {gc_overhead}% in last 60s, effective throughput {gc_throughput}%"},
            {"level": "INFO", "msg": "c.a.s.gc.Monitor [gc-monitor-1] - CMS concurrent-mark: {gc_pause}ms, heap {mem_pct}% occupied"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} (latency={aff_latency}ms) SLOW"},
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {root_ip}:{root_port} (latency=30001ms)"},
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
            {"level": "ERROR", "msg": "c.a.s.dns.Resolver [dns-cache-1] - lookup '{affected_svc}.internal' on {dns_ip}:53: SERVFAIL (rcode=2)"},
            {"level": "ERROR", "msg": "c.a.s.dns.Resolver [dns-cache-1] - lookup '{affected_db}.internal' on {dns_ip}:53: NXDOMAIN (rcode=3)"},
            {"level": "ERROR", "msg": "c.a.s.dns.Resolver [dns-cache-1] - {error_count} SERVFAIL responses in last 60s from {dns_ip}:53"},
            {"level": "FATAL", "msg": "c.a.s.dns.ZoneLoader [zone-sync-1] - zone file checksum mismatch after {version} sync: expected {checksum_exp}, got {checksum_got}"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.net.ConnectionFactory [conn-pool-{thread_n}] - java.net.UnknownHostException: {dep}.internal: Name or service not known"},
            {"level": "FATAL", "msg": "c.a.s.db.HikariPool [pool-3-housekeeper] - HikariPool-1 - Failed to validate connection: java.net.UnknownHostException: {service}-db.internal"},
            {"level": "FATAL", "msg": "c.a.s.net.ConnectionFactory [conn-pool-{thread_n}] - java.net.UnknownHostException: {root_service}.internal: Name or service not known"},
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
            {"level": "ERROR", "msg": "c.a.s.tls.CertValidator [tls-handshake-{thread_n}] - peer certificate rejected: notAfter={expiry_time} (expired {expired_hours} hours ago)"},
            {"level": "ERROR", "msg": "c.a.s.tls.CertValidator [tls-handshake-{thread_n2}] - CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate (depth=1, CN=*.{service}.internal)"},
            {"level": "WARN", "msg": "c.a.s.tls.AutoRenew [cert-renew-1] - ACME challenge failed: timeout connecting to ca.internal:443 after 30000ms"},
            {"level": "ERROR", "msg": "c.a.s.tls.CertValidator [tls-handshake-{thread_n}] - javax.net.ssl.SSLHandshakeException: PKIX path validation failed: java.security.cert.CertPathValidatorException: validity check failed"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - javax.net.ssl.SSLHandshakeException connecting to {root_ip}:{root_port}: peer not authenticated"},
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - TLS handshake failed to {root_ip}:{root_port}, falling back to cached response (age={cache_age}s)"},
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
            {"level": "ERROR", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {backend_ip1}:{root_port} health check failed: HTTP {hc_code} (expected 200)"},
            {"level": "WARN", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {backend_ip2}:{root_port} health check failed: HTTP {hc_code} (expected 200)"},
            {"level": "ERROR", "msg": "c.a.s.lb.Router [routing-{thread_n}] - all backends unhealthy for pool '{service}-pool', routing to last-known-good {backend_ip3}:{root_port}"},
            {"level": "WARN", "msg": "c.a.s.lb.ConfigSync [config-watch-1] - pool '{service}-pool' config version {version}: health_check_path=/healthz, but backends serve /health"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n}] - request queue depth {queue_depth}: accepting requests but response time degraded ({aff_latency}ms)"},
            {"level": "WARN", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n2}] - thread pool {pool_max}/{pool_max} active, {waiting} requests queued"},
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
            {"level": "ERROR", "msg": "c.a.s.config.Loader [config-watch-1] - loaded key='db.connection.url' value='jdbc:postgresql://{bad_db_ip}:{db_port}/{service}_db?ssl=true' source=configmap/{service}-{version}"},
            {"level": "ERROR", "msg": "c.a.s.db.HikariPool [pool-3-thread-{thread_n}] - Failed to obtain JDBC Connection: java.net.ConnectException: Connection refused (Connection refused) to {bad_db_ip}:{db_port}"},
            {"level": "FATAL", "msg": "c.a.s.db.HikariPool [pool-3-thread-{thread_n}] - HikariPool-1 - Exception during pool initialization: java.net.ConnectException: Connection refused"},
            {"level": "WARN", "msg": "c.a.s.config.Validator [startup-1] - WARN: unrecognized config key 'db.connection.pool.maxSize' in namespace '{service}', using default"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 500 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "WARN", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {root_ip}:{root_port} health check failed: HTTP 500 (expected 200)"},
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
            {"level": "ERROR", "msg": "c.a.s.client.ApiVersionNegotiator [api-{thread_n}] - server responded with API version {api_ver_actual}, client expected {api_ver_expected}"},
            {"level": "ERROR", "msg": "c.a.s.proto.SchemaRegistry [schema-{thread_n}] - incompatible schema: field 'metadata.trace_id' type changed from STRING to INT64"},
            {"level": "WARN", "msg": "c.a.s.proto.SchemaRegistry [schema-{thread_n}] - schema version {schema_ver_old} != {schema_ver_new}, {schema_diff_count} breaking changes detected"},
            {"level": "ERROR", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n2}] - java.lang.NullPointerException: Cannot invoke method on null reference at c.a.s.model.Response.getTraceId(Response.java:{line_num})"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} but response deserialization failed: com.fasterxml.jackson.databind.exc.MismatchedInputException"},
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} response body invalid: expected field 'data.items' as ARRAY, got OBJECT"},
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
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 429 from {ext_api_ip}:{ext_api_port} (latency={aff_latency}ms), Retry-After: {retry_after}s"},
            {"level": "WARN", "msg": "c.a.s.retry.ExponentialBackoff [retry-pool-{thread_n}] - attempt {retry_attempt}/{retry_max} for operation 'externalApi.call' (elapsed: {elapsed}ms)"},
            {"level": "ERROR", "msg": "c.a.s.retry.ExponentialBackoff [retry-pool-{thread_n2}] - backoff disabled in config v{version}: retrying immediately (delay=0ms)"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-external-api] - state: CLOSED -> OPEN (failures={failure_count}/{window})"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 503 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "ERROR", "msg": "c.a.s.circuit.CircuitBreaker [cb-{root_service}] - state: CLOSED -> OPEN after {failure_count} consecutive failures"},
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
            {"level": "ERROR", "msg": "c.a.s.cdn.OriginFetcher [origin-pool-{thread_n}] - HTTP 502 from origin {origin_ip}:{root_port} for path '/api/v1/assets/{asset_id}'"},
            {"level": "WARN", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {origin_ip}:{root_port} health check failed: HTTP 500 (expected 200)"},
            {"level": "ERROR", "msg": "c.a.s.cdn.CacheManager [cache-ttl-1] - cache MISS for key={asset_id}, origin unreachable, serving stale (age={cache_age}s, max-age=300)"},
            {"level": "WARN", "msg": "c.a.s.cdn.OriginFetcher [origin-pool-{thread_n}] - origin health endpoint GET {origin_ip}:{root_port}/healthz returned 404 (expected 200)"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 502 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {root_ip}:{root_port} (latency=30002ms), upstream origin timeout"},
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
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {gw_ip}:{gw_port} (latency=30001ms), transaction {txn_id}"},
            {"level": "WARN", "msg": "c.a.s.lb.HealthChecker [hc-monitor-1] - backend {gw_ip}:{gw_port} health check degraded: {hc_success}% success rate (threshold: 95%)"},
            {"level": "ERROR", "msg": "c.a.s.retry.ExponentialBackoff [retry-pool-{thread_n}] - attempt {retry_attempt}/{retry_max} for operation 'payment.charge' (elapsed: {elapsed}ms)"},
            {"level": "WARN", "msg": "c.a.s.handler.RequestHandler [http-nio-8080-exec-{thread_n2}] - thread pool {pool_max}/{pool_max} active, {waiting} requests queued (pending gateway calls)"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 504 from {root_ip}:{root_port} (latency=30003ms)"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-{root_service}] - state: CLOSED -> OPEN after {failure_count} consecutive failures"},
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
            {"level": "ERROR", "msg": "c.a.s.cache.Validator [cache-validator-1] - entry key=user:{user_id} checksum mismatch: stored={checksum_exp} computed={checksum_got} (serializer v{version})"},
            {"level": "ERROR", "msg": "c.a.s.cache.Validator [cache-validator-1] - com.fasterxml.jackson.core.JsonParseException: Unexpected character at position {line_num} for key=user:{user_id2}"},
            {"level": "WARN", "msg": "c.a.s.cache.Manager [cache-evictor-1] - TTL bypass detected: key=session:{user_id} age={cache_age}s exceeds max-age=300s, still served"},
            {"level": "WARN", "msg": "c.a.s.cache.Warmer [cache-warm-1] - populated {cache_warm_count} entries using serializer v{version} (expected v{schema_ver_old})"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} but response body failed validation: com.fasterxml.jackson.databind.exc.MismatchedInputException"},
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 200 from {root_ip}:{root_port} response field 'user.email' type STRING expected, got NULL"},
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
            {"level": "ERROR", "msg": "c.a.s.queue.Consumer [consumer-{thread_n}] - partition {partition} offset lag: {queue_depth} messages behind head (broker {broker_ip}:{broker_port})"},
            {"level": "WARN", "msg": "c.a.s.queue.ConsumerGroup [group-coordinator-1] - consumer group '{service}-consumers' rebalancing: {partition_count} partitions, {consumer_count} consumers, all lagging"},
            {"level": "ERROR", "msg": "c.a.s.queue.Consumer [consumer-{thread_n2}] - com.fasterxml.jackson.databind.exc.InvalidFormatException: Cannot deserialize value of type 'long' from String at offset {queue_depth} (serializer v{version})"},
            {"level": "WARN", "msg": "c.a.s.queue.DLQ [dlq-writer-1] - dead letter queue '{service}-dlq' depth: {dlq_count} messages in last 300s (broker {broker_ip}:{broker_port})"},
        ],
        "log_patterns_affected": [
            {"level": "WARN", "msg": "c.a.s.queue.Consumer [consumer-{thread_n}] - event from topic '{root_service}-events' delayed: consumer lag {lag}s at partition {partition} (broker {broker_ip}:{broker_port})"},
            {"level": "ERROR", "msg": "c.a.s.handler.EventHandler [event-pool-{thread_n}] - timeout processing event from '{root_service}-events': {lag}s behind real-time"},
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
            {"level": "ERROR", "msg": "c.a.s.db.HikariPool [pool-3-thread-{thread_n}] - Failed to validate connection: org.postgresql.util.PSQLException: FATAL: password authentication failed for user \"{service}_svc\" at {db_ip}:{db_port}"},
            {"level": "ERROR", "msg": "c.a.s.auth.CredentialManager [cred-refresh-1] - credential rotation completed at {rotation_time} but cached credential hash {checksum_exp} != vault hash {checksum_got}"},
            {"level": "WARN", "msg": "c.a.s.auth.CredentialManager [cred-refresh-1] - vault secret version {schema_ver_new} loaded, service still using version {schema_ver_old}"},
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 401 from {db_ip}:{db_port} (latency={aff_latency}ms), WWW-Authenticate: Bearer error=\"invalid_token\""},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 401 from {root_ip}:{root_port} (latency={aff_latency}ms)"},
            {"level": "WARN", "msg": "c.a.s.circuit.CircuitBreaker [cb-{root_service}] - state: CLOSED -> OPEN after {failure_count} consecutive failures (all HTTP 401)"},
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
            {"level": "ERROR", "msg": "c.a.s.waf.RuleEngine [waf-worker-{thread_n}] - rule {rule_id} BLOCK: src={client_ip} uri='/api/v1/checkout' match='content-type: application/json; charset=utf-8' (pattern: /json.*charset/i)"},
            {"level": "WARN", "msg": "c.a.s.waf.RuleEngine [waf-worker-{thread_n2}] - rule {rule_id} match rate {block_rate}% in last 60s (ruleset v{version})"},
            {"level": "ERROR", "msg": "c.a.s.waf.RuleEngine [waf-worker-{thread_n}] - rule {rule_id} BLOCK: src={client_ip2} uri='/api/v1/search' user-agent='Mozilla/5.0 (compatible; Chrome/{chrome_ver})' match='Mozilla.*Chrome' (pattern: /Mozilla.*Chrome/i)"},
            {"level": "WARN", "msg": "c.a.s.waf.AuditLog [waf-audit-1] - {block_count} requests blocked in last 300s by rule {rule_id} (top src: {client_ip} x{block_src_count})"},
        ],
        "log_patterns_affected": [
            {"level": "ERROR", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 403 from {root_ip}:{root_port} (latency={aff_latency}ms), body: '{{\"error\":\"blocked_by_waf\",\"rule\":\"{rule_id}\"}}'"},
            {"level": "WARN", "msg": "c.a.s.client.ServiceClient [http-{thread_n}] - HTTP 403 from {root_ip}:{root_port}: {failure_count} consecutive 403 responses"},
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
    {"level": "INFO", "msg": "c.a.s.health.Monitor [health-check-1] - status=UP, checks=[db:UP, cache:UP, queue:UP]"},
    {"level": "INFO", "msg": "c.a.s.cache.Stats [cache-stats-1] - hit_ratio={hit_ratio}% entries=24531 evictions=12/s"},
    {"level": "INFO", "msg": "c.a.s.batch.Executor [batch-pool-3] - job 'daily-reconcile' completed: {batch_count} records processed in 142s"},
    {"level": "INFO", "msg": "c.a.s.search.Indexer [reindex-pool-1] - reindex progress: 40% (12403/31008 docs), est. remaining: 340s"},
    {"level": "INFO", "msg": "c.a.s.ml.ModelServer [model-serve-1] - inference latency p99=23ms, model=v4.2, batch_size=32"},
    {"level": "WARN", "msg": "c.a.s.db.SlowQueryLog [query-monitor-1] - slow query: 2103ms SELECT * FROM audit_log WHERE created_at > '2026-03-25' (non-critical)"},
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

    # Generate stable IPs for this scenario using the seeded rng
    root_ip = rand_ip(rng)
    root_port = rand_port(rng, 8080)
    db_ip = rand_ip(rng)
    db_port = rand_port(rng, 5432)
    replica_ip = rand_ip(rng)
    replica_ip2 = rand_ip(rng)
    dns_ip = rand_ip(rng)
    broker_ip = rand_ip(rng)
    broker_port = rand_port(rng, 9092)
    gw_ip = rand_ip(rng)
    gw_port = rand_port(rng, 443)
    origin_ip = rand_ip(rng)
    ext_api_ip = rand_ip(rng)
    ext_api_port = rand_port(rng, 443)
    bad_db_ip = rand_ip(rng)
    backend_ip1 = rand_ip(rng)
    backend_ip2 = rand_ip(rng)
    backend_ip3 = rand_ip(rng)
    client_ip = rand_ip(rng)
    client_ip2 = rand_ip(rng)

    # Root cause logs
    root_logs = []
    root_ts = make_timestamps(base_hour, base_min, num_logs)
    for i, lp in enumerate(template["log_patterns_root"][:num_logs]):
        # Shared format variables for Option C style logs
        heap_max = rng.randint(4096, 16384)
        heap_used = rng.randint(int(heap_max * 0.8), heap_max)
        pool_max = rng.randint(20, 100)
        fmt_vars = dict(
            service=root_name, version=root_version,
            # Thread/pool identifiers
            thread_n=rng.randint(1, 48), thread_n2=rng.randint(1, 48),
            pool_max=pool_max, waiting=rng.randint(10, 200),
            # Database / connection
            db_ip=db_ip, db_port=db_port, bad_db_ip=bad_db_ip,
            timeout=rng.choice([5000, 10000, 15000, 30000]),
            txn_id=rng.randint(10000, 99999), txn_id2=rng.randint(10000, 99999),
            row_id1=rng.randint(100000, 999999), row_id2=rng.randint(100000, 999999),
            # Replication
            replica_ip=replica_ip, replica_ip2=replica_ip2,
            repl_lag=rng.randint(30, 120), repl_lag2=rng.randint(30, 120),
            row_ver_old=rng.randint(1000, 9999), row_ver_new=rng.randint(10000, 19999),
            cache_key=f"user:{rng.randint(1000, 99999)}",
            # Memory / GC
            mem_pct=rng.randint(85, 98), heap_used=heap_used, heap_max=heap_max,
            survivor=rng.randint(10, 200), old_gen=rng.randint(2000, heap_used),
            gc_pause=rng.randint(500, 5000), gc_count=rng.randint(10, 200),
            freed=rng.randint(10, 500), gc_overhead=rng.randint(40, 95),
            gc_throughput=rng.randint(5, 60), evict_rate=rng.randint(100, 5000),
            kern_ts=f"{rng.uniform(1000, 9999):.6f}",
            pid=rng.randint(1000, 65000), vm_kb=rng.randint(2000000, 16000000),
            rss_kb=rng.randint(1000000, 8000000),
            # CPU spin
            cpu_pct=rng.randint(90, 100), retry_attempt=rng.randint(3, 20),
            retry_max=rng.choice([5, 10, 20]), op_name=rng.choice(["db.query", "cache.get", "api.call", "payment.charge"]),
            elapsed=rng.randint(5000, 60000), elapsed2=rng.randint(60000, 300000),
            stuck_secs=rng.randint(30, 600), line_num=rng.randint(100, 999),
            stuck_threads=rng.randint(10, pool_max),
            # DNS
            dns_ip=dns_ip, error_count=rng.randint(100, 5000),
            checksum_exp=f"sha256:{rng.randint(10000000, 99999999):08x}",
            checksum_got=f"sha256:{rng.randint(10000000, 99999999):08x}",
            # Network / IPs
            root_ip=root_ip, root_port=root_port,
            broker_ip=broker_ip, broker_port=broker_port,
            gw_ip=gw_ip, gw_port=gw_port, origin_ip=origin_ip,
            ext_api_ip=ext_api_ip, ext_api_port=ext_api_port,
            backend_ip1=backend_ip1, backend_ip2=backend_ip2, backend_ip3=backend_ip3,
            client_ip=client_ip, client_ip2=client_ip2,
            # LB / CDN
            hc_code=rng.choice([404, 500, 502, 503]),
            hc_success=rng.randint(40, 70),
            asset_id=f"asset-{rng.randint(10000, 99999)}", cache_age=rng.randint(300, 86400),
            # Queue
            queue_depth=rng.randint(10000, 100000), lag=rng.randint(60, 600),
            dlq_count=rng.randint(100, 5000), partition=rng.randint(0, 15),
            partition_count=rng.randint(8, 32), consumer_count=rng.randint(2, 8),
            # Rate limiting
            retry_after=rng.randint(30, 300), rate_limit=rng.randint(100, 1000),
            retry_count=rng.randint(100, 5000), backoff=rng.choice([1000, 2000, 5000]),
            # Auth / creds
            rotation_time=f"2026-03-26T{max(base_hour-1,0):02d}:00:00Z",
            expiry_time=f"2026-03-26T{max(base_hour-2,0):02d}:00:00Z",
            expired_hours=rng.randint(1, 48),
            # WAF
            rule_id=f"WAF-{rng.randint(1000,9999)}", block_rate=rng.randint(30, 80),
            block_count=rng.randint(500, 5000), block_src_count=rng.randint(50, 500),
            chrome_ver=f"{rng.randint(90, 130)}.0.{rng.randint(1000, 9999)}.{rng.randint(10, 99)}",
            # Version mismatch
            api_ver_actual=f"v{rng.randint(1, 3)}", api_ver_expected=f"v{rng.randint(3, 5)}",
            schema_ver_old=f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
            schema_ver_new=f"{rng.randint(5, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
            schema_diff_count=rng.randint(2, 12),
            # Cache
            user_id=rng.randint(1000, 99999), user_id2=rng.randint(1000, 99999),
            cache_warm_count=rng.randint(10000, 100000),
            # General / shared
            error_rate=round(rng.uniform(10, 80), 1),
            failure_count=rng.randint(50, 500), window=rng.choice([10, 20, 50, 100]),
            affected_svc=affected[0]["name"] if affected else root_name,
            affected_db=f"{affected[0]['name']}-db" if affected else f"{root_name}-db",
            dep=f"{root_name}-db", root_service=root_name,
            # Legacy compat
            backend_count=rng.randint(3, 8), batch_count=rng.randint(1000, 50000),
            hit_ratio=rng.randint(30, 60), rss_mb=rng.randint(2000, 8000),
            restart_count=rng.randint(1, 5), cache_size=rng.randint(100000, 999999),
            req_id=rng.randint(10000, 99999),
        )
        msg = lp["msg"].format(**fmt_vars)
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
            aff_pool_max = rng.randint(20, 100)
            aff_fmt_vars = dict(
                root_service=root_name, service=svc["name"],
                req_id=rng.randint(10000, 99999),
                dep=f"{svc['name']}-db.internal",
                root_ip=root_ip, root_port=root_port,
                thread_n=rng.randint(1, 48), thread_n2=rng.randint(1, 48),
                aff_latency=rng.randint(1000, 30000),
                failure_count=rng.randint(10, 200),
                rule_id=f"WAF-{rng.randint(1000,9999)}",
                row_ver_old=rng.randint(1000, 9999),
                row_ver_new=rng.randint(10000, 19999),
                cache_key=f"user:{rng.randint(1000, 99999)}",
                cache_age=rng.randint(300, 86400),
                backoff=rng.choice([1000, 2000, 5000]),
                lag=rng.randint(60, 600),
                partition=rng.randint(0, 15),
                broker_ip=broker_ip, broker_port=broker_port,
                queue_depth=rng.randint(10000, 100000),
                pool_max=aff_pool_max, waiting=rng.randint(10, 200),
                version=root_version,
            )
            msg = lp["msg"].format(**aff_fmt_vars)
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
            # Keep original scenario_001 and scenario_002 (hand-crafted Option C style)
            if old.name in ("scenario_001.json", "scenario_002.json"):
                continue
            old.unlink()

        print(f"\n--- Generating {args.count - 1} {difficulty} scenarios (003-{args.count+1:03d}) ---")

        # Keep scenario_001 and scenario_002 as-is, generate scenario_003 through scenario_{count+1}
        for i in range(args.count - 1):
            file_num = i + 3  # start at 003
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
