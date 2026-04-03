"""Grading functions for OnCallEnv -- deterministic, returning 0.0-1.0."""

from __future__ import annotations

from typing import Any


def grade_episode(
    scenario: dict[str, Any],
    actions_taken: list[dict],
    alerts_state: list[dict],
    services_state: list[dict],
    severity_set: str | None,
    summary: str,
    escalated_to: str | None,
    resolved: bool,
) -> float:
    """Compute weighted reward for a completed episode.

    reward = weighted_sum(
        triage_accuracy      * 0.15,
        diagnostic_quality   * 0.25,
        root_cause_correct   * 0.25,
        remediation_quality  * 0.20,
        efficiency           * 0.10,
        documentation        * 0.05,
    )
    """
    rubric = scenario.get("grading_rubric", {})
    weights = {
        "triage": rubric.get("triage_weight", 0.15),
        "diagnostic": rubric.get("diagnostic_weight", 0.25),
        "root_cause": rubric.get("root_cause_weight", 0.25),
        "remediation": rubric.get("remediation_weight", 0.20),
        "efficiency": rubric.get("efficiency_weight", 0.10),
        "documentation": rubric.get("documentation_weight", 0.05),
    }

    triage = _grade_triage(scenario, actions_taken, alerts_state, severity_set)
    diagnostic = _grade_diagnostic(scenario, actions_taken)
    root_cause = _grade_root_cause(scenario, actions_taken, summary)
    remediation = _grade_remediation(scenario, actions_taken, services_state, resolved)
    efficiency = _grade_efficiency(actions_taken)
    documentation = _grade_documentation(summary)

    reward = (
        triage * weights["triage"]
        + diagnostic * weights["diagnostic"]
        + root_cause * weights["root_cause"]
        + remediation * weights["remediation"]
        + efficiency * weights["efficiency"]
        + documentation * weights["documentation"]
    )

    return round(min(max(reward, 0.0), 1.0), 4)


def _grade_triage(
    scenario: dict, actions: list[dict], alerts_state: list[dict], severity_set: str | None
) -> float:
    """Score triage quality: alert acknowledgment, correct severity classification."""
    score = 0.0
    total_checks = 0

    # Check if critical alerts were acknowledged (not silenced)
    critical_alerts = [a for a in scenario.get("initial_alerts", []) if a.get("severity") == "critical"]
    ack_actions = [a for a in actions if a["action_type"] == "acknowledge_alert"]
    acked_ids = {a["params"].get("alert_id") for a in ack_actions}

    for alert in critical_alerts:
        total_checks += 1
        if alert["alert_id"] in acked_ids:
            score += 1.0

    # Penalize silencing critical alerts
    silence_actions = [a for a in actions if a["action_type"] == "silence_alert"]
    silenced_ids = {a["params"].get("alert_id") for a in silence_actions}
    for alert in critical_alerts:
        if alert["alert_id"] in silenced_ids:
            score -= 0.5

    # Check severity setting
    expected_severity = scenario.get("expected_severity")
    if expected_severity:
        total_checks += 1
        if severity_set == expected_severity:
            score += 1.0
        elif severity_set is not None:
            score += 0.3  # partial credit for setting any severity

    # Check info alerts silenced (good practice)
    info_alerts = [a for a in scenario.get("initial_alerts", []) if a.get("severity") == "info"]
    for alert in info_alerts:
        total_checks += 0.5
        if alert["alert_id"] in silenced_ids or alert["alert_id"] in acked_ids:
            score += 0.5

    if total_checks == 0:
        return 0.5  # neutral if no triage expected
    return min(max(score / total_checks, 0.0), 1.0)


def _grade_diagnostic(scenario: dict, actions: list[dict]) -> float:
    """Score diagnostic quality: did agent query the right logs/metrics/dependencies?"""
    expected_diagnostics = scenario.get("expected_diagnostics", [])
    if not expected_diagnostics:
        return 0.5

    score = 0.0
    for diag in expected_diagnostics:
        action_type = diag.get("action_type")
        required_params = diag.get("params", {})

        for action in actions:
            if action["action_type"] != action_type:
                continue
            # Check if required params are a subset of action params
            match = all(
                action["params"].get(k, "").lower() == v.lower()
                for k, v in required_params.items()
                if isinstance(v, str)
            )
            if match:
                score += 1.0
                break
        else:
            # Partial credit for querying the right action type
            if any(a["action_type"] == action_type for a in actions):
                score += 0.3

    return min(score / len(expected_diagnostics), 1.0)


def _grade_root_cause(scenario: dict, actions: list[dict], summary: str) -> float:
    """Score whether agent identified the root cause.

    Requires both evidence of investigation AND correct identification.
    An agent that just names every service in the summary shouldn't score high.
    """
    root_cause = scenario.get("root_cause", {})
    if not root_cause:
        return 0.5

    rc_service = root_cause.get("service", "").lower()
    rc_keywords = [kw.lower() for kw in root_cause.get("keywords", [])]

    score = 0.0
    summary_lower = summary.lower()

    # Check if summary mentions root cause service (+0.3)
    if rc_service and rc_service in summary_lower:
        score += 0.3

    # Check if summary contains root cause keywords (+0.3)
    # Require matching at least half the keywords for full credit
    if rc_keywords:
        matched = sum(1 for kw in rc_keywords if kw in summary_lower)
        keyword_ratio = matched / len(rc_keywords)
        score += 0.3 * keyword_ratio

    # Did agent actually investigate the root cause service? (+0.2)
    # Without this, the agent might just guess from alert text
    investigation_types = {"query_logs", "check_metrics", "view_dependencies"}
    investigated_rc = any(
        a["action_type"] in investigation_types
        and (a["params"].get("service", "").lower() == rc_service
             or a["params"].get("service_name", "").lower() == rc_service)
        for a in actions
    )
    if investigated_rc:
        score += 0.2

    # Check if remediation targeted the right service (+0.2)
    remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}
    remediation_actions = [a for a in actions if a["action_type"] in remediation_types]
    for ra in remediation_actions:
        target_svc = ra["params"].get("service_name", "").lower()
        if target_svc == rc_service:
            score += 0.2
            break

    return min(score, 1.0)


def _grade_remediation(
    scenario: dict, actions: list[dict], services_state: list[dict], resolved: bool
) -> float:
    """Score remediation quality with blast-radius and EGAR penalties.

    A surgical agent that investigates first and remediates precisely should
    score much higher than one that shotguns rollbacks at every service.
    """
    valid_remediations = scenario.get("valid_remediations", [])
    remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}
    investigation_types = {"query_logs", "check_metrics", "view_dependencies"}

    if not valid_remediations:
        return 0.5 if resolved else 0.0

    remediation_actions = [a for a in actions if a["action_type"] in remediation_types]

    score = 0.0

    # Check if any valid remediation was executed (+0.4)
    hit_valid = False
    for vr in valid_remediations:
        for action in actions:
            if action["action_type"] == vr.get("action") and action["params"].get("service_name") == vr.get("service"):
                hit_valid = True
                score += 0.4
                break

    score = min(score, 0.4)

    # Check if services are now healthy (+0.15)
    all_healthy = all(s.get("status") == "healthy" for s in services_state)
    if all_healthy:
        score += 0.15

    # Bonus for resolving (+0.15)
    if resolved:
        score += 0.15

    # --- Blast radius penalty: penalize wrong remediations ---
    # An agent that tries 5 remediations and gets 1 right should score worse
    # than one that tries 1 and gets 1 right.
    if remediation_actions:
        valid_targets = {
            (vr.get("action"), vr.get("service"))
            for vr in valid_remediations
        }
        wrong_count = sum(
            1 for ra in remediation_actions
            if (ra["action_type"], ra["params"].get("service_name", "")) not in valid_targets
        )
        blast_ratio = wrong_count / len(remediation_actions)
        # Penalty scales: 0 wrong = 0.0 penalty, all wrong = -0.3 penalty
        score -= 0.3 * blast_ratio

    # --- EGAR penalty: penalize remediating without investigating that service first ---
    if remediation_actions:
        gated = 0
        for ra in remediation_actions:
            target = ra["params"].get("service_name", "")
            ra_step = ra.get("step", float("inf"))
            investigated = any(
                a["action_type"] in investigation_types
                and (a["params"].get("service", "") == target or a["params"].get("service_name", "") == target)
                and a.get("step", 0) < ra_step
                for a in actions
            )
            if investigated:
                gated += 1
        egar = gated / len(remediation_actions)
        # Bonus for investigating first: up to +0.3
        score += 0.3 * egar

    # Penalize harmful actions (restarting healthy services)
    healthy_services_at_start = {
        s["name"] for s in scenario.get("services", []) if s.get("status") == "healthy"
    }
    for action in actions:
        if action["action_type"] in remediation_types:
            svc = action["params"].get("service_name", "")
            if svc in healthy_services_at_start:
                score -= 0.1

    # Penalize premature resolution (resolving without any investigation)
    has_any_investigation = any(a["action_type"] in investigation_types for a in actions)
    if resolved and not has_any_investigation:
        score -= 0.3

    return min(max(score, 0.0), 1.0)


def _grade_efficiency(actions: list[dict]) -> float:
    """Score efficiency: reward thorough-but-focused investigation.

    Too few actions (< 5) means the agent didn't investigate properly.
    Too many actions (> 20) means the agent is flailing.
    Sweet spot is 6-12 actions: triage + investigate + remediate + document.
    """
    n = len(actions)
    if n == 0:
        return 0.0
    if n < 5:
        return 0.3  # too few -- likely skipped investigation
    if n <= 12:
        return 1.0  # sweet spot
    elif n <= 16:
        return 0.7
    elif n <= 20:
        return 0.4
    else:
        return 0.2


def _grade_documentation(summary: str) -> float:
    """Score incident summary completeness."""
    if not summary:
        return 0.0

    score = 0.0
    words = summary.split()
    # Minimal summary
    if len(words) >= 5:
        score += 0.3
    # Reasonable summary
    if len(words) >= 15:
        score += 0.3
    # Detailed summary
    if len(words) >= 30:
        score += 0.2
    # Contains technical keywords
    tech_keywords = ["root cause", "service", "error", "fix", "deploy", "config", "restart", "rollback", "latency", "alert"]
    matched = sum(1 for kw in tech_keywords if kw in summary.lower())
    score += 0.2 * min(matched / 3, 1.0)

    return min(score, 1.0)


# ── Supplementary Metrics (not part of main reward, for analysis) ────────


def compute_supplementary_metrics(
    scenario: dict[str, Any],
    actions_taken: list[dict],
    services_state: list[dict],
    resolved: bool,
) -> dict[str, float]:
    """Compute supplementary metrics for episode analysis.

    These do NOT affect the main reward. They provide additional insight
    into agent behavior for debugging, research, and leaderboard display.
    """
    return {
        "egar": _compute_egar(actions_taken),
        "blast_radius": _compute_blast_radius(scenario, actions_taken),
    }


def _compute_egar(actions: list[dict]) -> float:
    """Evidence-Gated Action Rate: did agent investigate before remediating?

    For each remediation action, checks whether the agent queried logs or
    metrics for that specific service beforehand. Returns fraction of
    remediation actions that were preceded by investigation of the target.

    Score: 1.0 = always investigated first, 0.0 = never investigated.
    """
    investigation_types = {"query_logs", "check_metrics", "view_dependencies"}
    remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}

    remediation_actions = [a for a in actions if a["action_type"] in remediation_types]
    if not remediation_actions:
        return 0.5  # No remediation attempted -- neutral

    gated_count = 0
    for ra in remediation_actions:
        target_svc = ra["params"].get("service_name", "")
        ra_step = ra.get("step", float("inf"))
        investigated = any(
            a["action_type"] in investigation_types
            and (a["params"].get("service", "") == target_svc or a["params"].get("service_name", "") == target_svc)
            and a.get("step", 0) < ra_step
            for a in actions
        )
        if investigated:
            gated_count += 1

    return round(gated_count / len(remediation_actions), 3)


def _compute_blast_radius(scenario: dict, actions: list[dict]) -> float:
    """Blast radius: ratio of incorrect remediation actions to total remediations.

    Lower is better. 0.0 = every remediation targeted the right service.
    1.0 = every remediation was wrong.
    """
    remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}
    remediation_actions = [a for a in actions if a["action_type"] in remediation_types]

    if not remediation_actions:
        return 0.0

    valid_targets = {
        (vr.get("action"), vr.get("service"))
        for vr in scenario.get("valid_remediations", [])
    }

    wrong_count = 0
    for ra in remediation_actions:
        key = (ra["action_type"], ra["params"].get("service_name", ""))
        if key not in valid_targets:
            wrong_count += 1

    return round(wrong_count / len(remediation_actions), 3)
