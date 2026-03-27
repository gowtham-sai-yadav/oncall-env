"""Dynamic simulation engine for service degradation and cascading recovery."""

from __future__ import annotations

from typing import Any

# Degradation factor per step (1.1 = 10% worse each step)
DEGRADATION_FACTOR = 1.1

# Status transition thresholds
ERROR_RATE_DOWN_THRESHOLD = 80.0  # degraded -> down when error_rate exceeds this


def degrade_services(services: list[dict], dependencies: dict[str, Any]) -> list[dict]:
    """Degrade unhealthy services each step (simulates real-time worsening).

    - degraded services: error_rate *= DEGRADATION_FACTOR, latency *= DEGRADATION_FACTOR
    - down services: stay at 100% error rate
    - healthy services: no change
    - Status transitions: degraded -> down when error_rate > ERROR_RATE_DOWN_THRESHOLD
    """
    for svc in services:
        if svc["status"] == "degraded":
            svc["error_rate"] = min(svc.get("error_rate", 0) * DEGRADATION_FACTOR, 100.0)
            svc["latency_ms"] = svc.get("latency_ms", 0) * DEGRADATION_FACTOR
            # Trigger status transition if error_rate exceeds threshold
            if svc["error_rate"] > ERROR_RATE_DOWN_THRESHOLD:
                svc["status"] = "down"
                svc["error_rate"] = 100.0
        elif svc["status"] == "down":
            svc["error_rate"] = 100.0
    return services


def propagate_recovery(
    services: list[dict],
    dependencies: dict[str, Any],
    fixed_service: str,
) -> list[dict]:
    """Propagate recovery when a root-cause service is fixed.

    Dependencies format: {"service_name": ["dep1", "dep2"]}
    meaning service_name DEPENDS ON dep1, dep2.

    When fixed_service becomes healthy, find all services that depend on it
    and begin their recovery (set to degraded with reduced error rate).
    Recovery is not instant — downstream services improve but don't jump to healthy.
    """
    svc_map = {s["name"]: s for s in services}

    # Find services whose dependency list mentions the fixed_service
    for svc_name, deps in dependencies.items():
        # Match both exact name and .internal suffix patterns
        depends_on_fixed = any(
            fixed_service == d or fixed_service in d or d in fixed_service
            for d in deps
        )
        if not depends_on_fixed:
            continue

        svc = svc_map.get(svc_name)
        if svc is None or svc["status"] == "healthy":
            continue

        # Check if ALL dependencies of this service are now healthy (or external)
        all_deps_ok = True
        for dep in deps:
            dep_svc = svc_map.get(dep)
            if dep_svc is not None and dep_svc["status"] != "healthy":
                all_deps_ok = False
                break

        if all_deps_ok:
            # Begin recovery: not instant, but significant improvement
            if svc["status"] == "down":
                svc["status"] = "degraded"
                svc["error_rate"] = min(svc.get("error_rate", 100) * 0.3, 30.0)
                svc["latency_ms"] = max(svc.get("latency_ms", 0) * 0.4, 100)
            elif svc["status"] == "degraded":
                svc["error_rate"] = max(svc.get("error_rate", 0) * 0.3, 0.5)
                svc["latency_ms"] = max(svc.get("latency_ms", 0) * 0.4, 50)

    return services
