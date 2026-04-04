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
    documentation = _grade_documentation(summary, scenario)

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

    Investigation-gated: naming the root cause without investigating it
    yields at most 0.2. Agents must actually query_logs / check_metrics /
    view_dependencies for the root-cause service to unlock full credit.

    Shotgun penalty: mentioning >3 service names from the scenario in the
    summary applies a 0.5x multiplier (prevents spamming every service name).
    """
    root_cause = scenario.get("root_cause", {})
    if not root_cause:
        return 0.5

    rc_service = root_cause.get("service", "").lower()
    rc_keywords = [kw.lower() for kw in root_cause.get("keywords", [])]

    summary_lower = summary.lower()

    # --- Did agent investigate the root cause service? ---
    investigation_types = {"query_logs", "check_metrics", "view_dependencies"}
    investigated_rc = any(
        a["action_type"] in investigation_types
        and (a["params"].get("service", "").lower() == rc_service
             or a["params"].get("service_name", "").lower() == rc_service)
        for a in actions
    )

    score = 0.0

    if not investigated_rc:
        # --- NOT investigated: cap entire root_cause score at 0.2 ---
        if rc_service and rc_service in summary_lower:
            score = 0.2  # lucky guess credit
        # Nothing else contributes -- early return after shotgun check
    else:
        # --- Investigated: full rubric unlocked ---

        # Service in summary: +0.2
        if rc_service and rc_service in summary_lower:
            score += 0.2

        # Keywords in summary: +0.2, require >=50% match
        if rc_keywords:
            matched = sum(1 for kw in rc_keywords if kw in summary_lower)
            keyword_ratio = matched / len(rc_keywords)
            if keyword_ratio >= 0.5:
                score += 0.2 * keyword_ratio
            # Below 50% match: no keyword credit

        # Investigation bonus: +0.3
        score += 0.3

        # Remediation targeted right service: +0.2
        remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}
        remediation_actions = [a for a in actions if a["action_type"] in remediation_types]
        for ra in remediation_actions:
            target_svc = ra["params"].get("service_name", "").lower()
            if target_svc == rc_service:
                score += 0.2
                break

    # --- Shotgun penalty: mentioning >3 scenario service names ---
    all_service_names = {
        s.get("name", "").lower()
        for s in scenario.get("services", [])
        if s.get("name")
    }
    mentioned_count = sum(1 for sn in all_service_names if sn in summary_lower)
    if mentioned_count > 3:
        score *= 0.5

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

    # --- Blast radius penalty: heavily penalize wrong remediations ---
    # An agent that tries 5 remediations and gets 1 right should score much worse
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
        # Per-action penalty: each wrong remediation costs -0.08
        score -= wrong_count * 0.08
        # Ratio penalty: if majority of remediations are wrong, additional penalty
        blast_ratio = wrong_count / len(remediation_actions)
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
    """Score efficiency: investigation-aware step counting.

    Requirements to unlock higher scores:
    - Must investigate at least 2 different services to score above 0.3
    - Investigation-to-remediation ratio must be >= 1.5 to score above 0.5
    - Sweet spot: 6-14 steps = 1.0
    - Below 5: 0.2 (too few = didn't investigate)
    - 15-18: 0.6
    - 19+: 0.2
    """
    n = len(actions)
    if n == 0:
        return 0.0

    investigation_types = {"query_logs", "check_metrics", "view_dependencies"}
    remediation_types = {"restart_service", "scale_service", "rollback_deploy", "update_config"}

    # Count distinct services investigated
    investigated_services = set()
    investigation_count = 0
    for a in actions:
        if a["action_type"] in investigation_types:
            investigation_count += 1
            svc = a["params"].get("service", "") or a["params"].get("service_name", "")
            if svc:
                investigated_services.add(svc.lower())

    remediation_count = sum(1 for a in actions if a["action_type"] in remediation_types)

    # --- Step-count base score (tighter curve) ---
    if n < 6:
        base_score = 0.15  # too few = definitely skipped investigation
    elif n <= 12:
        base_score = 1.0  # sweet spot: triage + investigate + remediate + document
    elif n <= 16:
        base_score = 0.5  # slightly over
    elif n <= 20:
        base_score = 0.3  # flailing
    else:
        base_score = 0.1  # way too many steps

    # --- Gate: must investigate >=3 services to score above 0.3 ---
    if len(investigated_services) < 3:
        base_score = min(base_score, 0.3)

    # --- Gate: investigation-to-remediation ratio >= 2.0 to score above 0.5 ---
    if remediation_count > 0:
        ratio = investigation_count / remediation_count
        if ratio < 2.0:
            base_score = min(base_score, 0.5)

    # --- Penalty: duplicate investigation (re-querying same service) ---
    investigation_targets = [
        a["params"].get("service", "") or a["params"].get("service_name", "")
        for a in actions if a["action_type"] in investigation_types
    ]
    duplicates = len(investigation_targets) - len(set(investigation_targets))
    if duplicates > 0:
        base_score -= 0.05 * duplicates

    return max(base_score, 0.0)


def _grade_documentation(summary: str, scenario: dict | None = None) -> float:
    """Score incident summary completeness.

    When a scenario is provided, scenario-specific keywords from
    root_cause["keywords"] are worth much more than generic tech keywords.
    """
    if not summary:
        return 0.0

    score = 0.0
    words = summary.split()
    summary_lower = summary.lower()

    # --- Word count: up to 0.4 ---
    # Minimal summary
    if len(words) >= 5:
        score += 0.15
    # Reasonable summary
    if len(words) >= 15:
        score += 0.15
    # Detailed summary
    if len(words) >= 30:
        score += 0.1

    # --- Keywords ---
    if scenario:
        # Generic tech keywords: worth 0.1 max (down from 0.2)
        tech_keywords = [
            "root cause", "service", "error", "fix", "deploy",
            "config", "restart", "rollback", "latency", "alert",
        ]
        generic_matched = sum(1 for kw in tech_keywords if kw in summary_lower)
        score += 0.1 * min(generic_matched / 3, 1.0)

        # Scenario-specific keywords: worth 0.5
        rc_keywords = [
            kw.lower()
            for kw in scenario.get("root_cause", {}).get("keywords", [])
        ]
        if rc_keywords:
            specific_matched = sum(1 for kw in rc_keywords if kw in summary_lower)
            specific_ratio = specific_matched / len(rc_keywords)
            score += 0.5 * specific_ratio
        else:
            # No scenario keywords defined -- fall back to generous generic credit
            score += 0.1 * min(generic_matched / 3, 1.0)
    else:
        # No scenario provided -- original generic keyword scoring (0.2 max)
        tech_keywords = [
            "root cause", "service", "error", "fix", "deploy",
            "config", "restart", "rollback", "latency", "alert",
        ]
        matched = sum(1 for kw in tech_keywords if kw in summary_lower)
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
