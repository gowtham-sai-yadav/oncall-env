"""Tests for the OnCallEnv environment."""

import re

import pytest
from oncall_env.models import OnCallAction, OnCallObservation, OnCallState
from oncall_env.server.environment import OnCallEnvironment
from oncall_env.server.graders import grade_episode
from oncall_env.server.scenario_loader import load_scenario_by_task


def _attr(obj, key, default=""):
    """Access field as attribute or dict key (handles both typed models and raw dicts)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── Scenario Loading ────────────────────────────────────────────────────────

def test_load_task1_scenario():
    scenario = load_scenario_by_task(1, 0)
    assert scenario["incident_id"] == "INC-20260326-001"
    assert scenario["difficulty"] == "easy"
    assert len(scenario["initial_alerts"]) >= 5


def test_load_task2_scenario():
    scenario = load_scenario_by_task(2, 0)
    assert scenario["difficulty"] == "medium"
    assert "root_cause" in scenario


def test_load_task3_scenario():
    scenario = load_scenario_by_task(3, 0)
    assert scenario["difficulty"] == "hard"


def test_load_task4_scenario():
    scenario = load_scenario_by_task(4, 0)
    assert scenario["difficulty"] == "expert"


def test_load_invalid_task():
    with pytest.raises(ValueError):
        load_scenario_by_task(5, 0)


# ── Environment Reset/Step ──────────────────────────────────────────────────

def test_environment_reset():
    env = OnCallEnvironment()
    obs = env.reset(task_id=1, scenario_idx=0)
    assert isinstance(obs, OnCallObservation)
    assert not obs.done
    assert obs.reward is None
    assert len(obs.alerts) >= 5
    assert len(obs.services) >= 3
    assert obs.message != ""
    # Partial observability: uninvestigated services should show "unknown" status
    for svc in obs.services:
        status = _attr(svc, "status")
        assert status == "unknown", f"Service {_attr(svc, 'name')} should be 'unknown' before investigation, got '{status}'"


def test_environment_step_query_logs():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="query_logs", params={"service": "payment-api"})
    obs = env.step(action)
    assert isinstance(obs, OnCallObservation)
    assert obs.log_results is not None
    assert len(obs.log_results) > 0
    assert not obs.done
    # After querying logs, the investigated service should be revealed with real name
    found = any(_attr(svc, "name") == "payment-api" and _attr(svc, "status") != "unknown" for svc in obs.services)
    assert found, "payment-api should be revealed after query_logs"


def test_alias_resolution():
    """Queries using aliases should resolve to real service data."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    # Find the alias for payment-api
    alias = env._service_alias_map["payment-api"]
    # Query using the alias
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": alias}))
    # Should return payment-api's logs (resolved internally)
    assert obs.log_results is not None
    assert len(obs.log_results) > 0
    # payment-api should now be revealed in the services list
    found = any(_attr(svc, "name") == "payment-api" and _attr(svc, "status") != "unknown" for svc in obs.services)
    assert found, "payment-api should be revealed when queried via alias"


def test_environment_step_check_metrics():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="check_metrics", params={"service": "payment-api", "metric_name": "error_rate"})
    obs = env.step(action)
    assert obs.metric_results is not None


def test_environment_step_view_dependencies():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="view_dependencies", params={"service_name": "payment-api"})
    obs = env.step(action)
    assert obs.dependency_graph is not None


def test_environment_step_acknowledge_alert():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-001"})
    obs = env.step(action)
    assert "acknowledged" in obs.message.lower()


def test_environment_step_silence_alert():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="silence_alert", params={"alert_id": "alert-006"})
    obs = env.step(action)
    assert "silenced" in obs.message.lower()


def test_environment_step_set_severity():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="set_severity", params={"level": "SEV2"})
    obs = env.step(action)
    assert obs.current_severity == "SEV2"


def test_environment_step_restart_service():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="restart_service", params={"service_name": "payment-api"})
    obs = env.step(action)
    assert "restart" in obs.message.lower()


def test_environment_step_resolve_incident():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    action = OnCallAction(action_type="resolve_incident", params={"resolution_note": "Fixed by restarting payment-api"})
    obs = env.step(action)
    assert obs.done
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_environment_step_limit():
    """Test that environment ends after MAX_STEPS."""
    from oncall_env.server.environment import MAX_STEPS
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    for i in range(MAX_STEPS + 1):
        action = OnCallAction(action_type="query_logs", params={"service": "payment-api"})
        obs = env.step(action)
        if obs.done:
            break
    assert obs.done
    assert obs.reward is not None


# ── Full Episode Test ───────────────────────────────────────────────────────

def test_full_episode_task1():
    """Test a realistic Task 1 episode with good triage."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)

    actions = [
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-001"}),
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-002"}),
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-007"}),
        OnCallAction(action_type="silence_alert", params={"alert_id": "alert-004"}),
        OnCallAction(action_type="silence_alert", params={"alert_id": "alert-006"}),
        OnCallAction(action_type="set_severity", params={"level": "SEV2"}),
        OnCallAction(action_type="query_logs", params={"service": "payment-api"}),
        OnCallAction(action_type="check_metrics", params={"service": "payment-api", "metric_name": "error_rate"}),
        OnCallAction(action_type="restart_service", params={"service_name": "payment-api"}),
        OnCallAction(action_type="write_summary", params={"text": "Root cause: payment-api DB connection pool exhaustion caused by connection leak in v2.4.1. Fix: restarted payment-api service to reset connections."}),
        OnCallAction(action_type="resolve_incident", params={"resolution_note": "Resolved by restarting payment-api"}),
    ]

    for action in actions:
        obs = env.step(action)
        if obs.done:
            break

    assert obs.done
    assert obs.reward is not None
    assert obs.reward > 0.3, f"Expected good score for proper triage, got {obs.reward}"


def test_full_episode_task2():
    """Test a realistic Task 2 episode with investigation."""
    env = OnCallEnvironment()
    env.reset(task_id=2, scenario_idx=0)

    actions = [
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-201"}),
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-202"}),
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-204"}),
        OnCallAction(action_type="set_severity", params={"level": "SEV1"}),
        OnCallAction(action_type="view_dependencies", params={"service_name": "order-service"}),
        OnCallAction(action_type="query_logs", params={"service": "order-service"}),
        OnCallAction(action_type="query_logs", params={"service": "inventory-service"}),
        OnCallAction(action_type="check_metrics", params={"service": "inventory-service", "metric_name": "memory_percent"}),
        OnCallAction(action_type="rollback_deploy", params={"service_name": "inventory-service", "target_version": "2.5.0"}),
        OnCallAction(action_type="write_summary", params={"text": "Root cause: inventory-service v2.5.1 deployment introduced a memory leak in the DB connection pooling layer. The inventory-db ran out of connections and was OOM-killed, causing cascading failures. Fix: rolled back inventory-service to v2.5.0."}),
        OnCallAction(action_type="resolve_incident", params={"resolution_note": "Rolled back inventory-service"}),
    ]

    for action in actions:
        obs = env.step(action)
        if obs.done:
            break

    assert obs.done
    assert obs.reward is not None
    assert obs.reward > 0.2, f"Expected decent score for Task 2, got {obs.reward}"


# ── Grader Tests ────────────────────────────────────────────────────────────

def test_grader_empty_episode():
    """An agent that does nothing should score low."""
    scenario = load_scenario_by_task(1, 0)
    reward = grade_episode(
        scenario=scenario,
        actions_taken=[],
        alerts_state=scenario["initial_alerts"],
        services_state=scenario["services"],
        severity_set=None,
        summary="",
        escalated_to=None,
        resolved=False,
    )
    assert 0.0 <= reward <= 0.3, f"Empty episode should score low, got {reward}"


def test_grader_perfect_triage():
    """An agent with perfect triage should score well on triage component."""
    scenario = load_scenario_by_task(1, 0)
    actions = [
        {"action_type": "acknowledge_alert", "params": {"alert_id": "alert-001"}, "step": 1},
        {"action_type": "acknowledge_alert", "params": {"alert_id": "alert-002"}, "step": 2},
        {"action_type": "acknowledge_alert", "params": {"alert_id": "alert-007"}, "step": 3},
        {"action_type": "silence_alert", "params": {"alert_id": "alert-004"}, "step": 4},
        {"action_type": "silence_alert", "params": {"alert_id": "alert-006"}, "step": 5},
        {"action_type": "set_severity", "params": {"level": "SEV2"}, "step": 6},
    ]
    alerts = scenario["initial_alerts"][:]
    alerts[0]["acknowledged"] = True
    alerts[1]["acknowledged"] = True
    alerts[6]["acknowledged"] = True
    alerts[3]["silenced"] = True
    alerts[5]["silenced"] = True

    reward = grade_episode(
        scenario=scenario,
        actions_taken=actions,
        alerts_state=alerts,
        services_state=scenario["services"],
        severity_set="SEV2",
        summary="",
        escalated_to=None,
        resolved=False,
    )
    assert reward > 0.15, f"Good triage should score above 0.15, got {reward}"


def test_grader_varying_scores():
    """Different actions should produce different scores."""
    scenario = load_scenario_by_task(1, 0)

    # Minimal actions
    r1 = grade_episode(scenario, [], scenario["initial_alerts"], scenario["services"], None, "", None, False)

    # Some triage
    r2 = grade_episode(
        scenario,
        [{"action_type": "acknowledge_alert", "params": {"alert_id": "alert-001"}, "step": 1}],
        scenario["initial_alerts"],
        scenario["services"],
        "SEV2",
        "",
        None,
        False,
    )

    # Full resolution
    r3 = grade_episode(
        scenario,
        [
            {"action_type": "acknowledge_alert", "params": {"alert_id": "alert-001"}, "step": 1},
            {"action_type": "query_logs", "params": {"service": "payment-api"}, "step": 2},
            {"action_type": "check_metrics", "params": {"service": "payment-api", "metric_name": "error_rate"}, "step": 3},
            {"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 4},
            {"action_type": "write_summary", "params": {"text": "Root cause: payment-api connection pool exhaustion. Fix: restarted service."}, "step": 5},
            {"action_type": "resolve_incident", "params": {"resolution_note": "fixed"}, "step": 6},
        ],
        scenario["initial_alerts"],
        [{"name": "payment-api", "status": "healthy"}, *scenario["services"][1:]],
        "SEV2",
        "Root cause: payment-api connection pool exhaustion. Fix: restarted service.",
        None,
        True,
    )

    assert r1 < r2 < r3, f"Scores should increase with better actions: {r1} < {r2} < {r3}"


# ── State Tests ─────────────────────────────────────────────────────────────

def test_state_tracking():
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    assert env.state.step_count == 0
    assert env.state.task_id == "task1"

    env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    assert env.state.step_count == 1

    env.step(OnCallAction(action_type="set_severity", params={"level": "SEV2"}))
    assert env.state.step_count == 2


def test_determinism():
    """Same actions on same scenario should produce same reward."""
    env1 = OnCallEnvironment()
    env2 = OnCallEnvironment()

    env1.reset(task_id=1, scenario_idx=0)
    env2.reset(task_id=1, scenario_idx=0)

    actions = [
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-001"}),
        OnCallAction(action_type="query_logs", params={"service": "payment-api"}),
        OnCallAction(action_type="resolve_incident", params={"resolution_note": "test"}),
    ]

    for action in actions:
        obs1 = env1.step(action)
        obs2 = env2.step(action)

    assert obs1.reward == obs2.reward, f"Expected deterministic reward: {obs1.reward} != {obs2.reward}"


# ── Edge Case Tests ──────────────────────────────────────────────────────────

def test_action_after_resolved():
    """Actions after resolution should return done with reward."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    env.step(OnCallAction(action_type="resolve_incident", params={"resolution_note": "done"}))
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    assert obs.done
    assert obs.reward is not None
    assert "already resolved" in obs.message.lower()


def test_empty_params():
    """Actions with empty params should not crash."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    obs = env.step(OnCallAction(action_type="query_logs", params={}))
    assert isinstance(obs, OnCallObservation)
    assert not obs.done


def test_missing_service_name():
    """Restart with empty service_name returns 'not found' message."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    obs = env.step(OnCallAction(action_type="restart_service", params={}))
    assert "not found" in obs.message.lower()


def test_long_string_truncation():
    """Extremely long summary text should be truncated, not crash."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    long_text = "A" * 50000
    obs = env.step(OnCallAction(action_type="write_summary", params={"text": long_text}))
    assert "summary recorded" in obs.message.lower()


def test_nonexistent_alert_id():
    """Acknowledge a non-existent alert returns not found."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    obs = env.step(OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-nonexistent"}))
    assert "not found" in obs.message.lower()


def test_new_scenarios_load():
    """New scenarios (task3 idx=1, task4 idx=1) load and work."""
    env = OnCallEnvironment()

    obs = env.reset(task_id=3, scenario_idx=1)
    assert isinstance(obs, OnCallObservation)
    assert len(obs.alerts) >= 5

    obs = env.reset(task_id=4, scenario_idx=1)
    assert isinstance(obs, OnCallObservation)
    assert len(obs.alerts) >= 5


def test_full_episode_new_task3_scenario():
    """Run a basic episode on the new task3 scenario."""
    env = OnCallEnvironment()
    env.reset(task_id=3, scenario_idx=1)
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "cache-layer"}))
    assert not obs.done
    obs = env.step(OnCallAction(action_type="resolve_incident", params={"resolution_note": "fixed cache"}))
    assert obs.done
    assert 0.0 <= obs.reward <= 1.0


def test_full_episode_new_task4_scenario():
    """Run a basic episode on the new task4 scenario."""
    env = OnCallEnvironment()
    env.reset(task_id=4, scenario_idx=1)
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "service-mesh"}))
    assert not obs.done
    obs = env.step(OnCallAction(action_type="resolve_incident", params={"resolution_note": "fixed mesh"}))
    assert obs.done
    assert 0.0 <= obs.reward <= 1.0


# ── Reward Signals & Rubric Tests ─────────────────────────────────────────

def test_reward_signals_present_after_step():
    """Verify per-step reward signals are in observation metadata."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    assert "reward_signals" in obs.metadata
    signals = obs.metadata["reward_signals"]
    assert "oncall.triage_progress" in signals
    assert "oncall.investigation_depth" in signals
    assert "oncall.premature_action" in signals
    assert "oncall.severity_set" in signals
    assert "oncall.summary_written" in signals
    assert "oncall.resolved" in signals


def test_reward_signals_progress_over_episode():
    """Verify reward signals reflect actual progress."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)

    # Before any action: triage_progress should be 0
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    assert obs.metadata["reward_signals"]["oncall.triage_progress"] == 0.0
    assert obs.metadata["reward_signals"]["oncall.investigation_depth"] > 0.0  # queried expected diagnostic

    # Acknowledge a critical alert
    obs = env.step(OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-001"}))
    assert obs.metadata["reward_signals"]["oncall.triage_progress"] > 0.0

    # Set severity
    obs = env.step(OnCallAction(action_type="set_severity", params={"level": "SEV2"}))
    assert obs.metadata["reward_signals"]["oncall.severity_set"] == 1.0

    # Write summary
    obs = env.step(OnCallAction(action_type="write_summary", params={"text": "Root cause identified."}))
    assert obs.metadata["reward_signals"]["oncall.summary_written"] == 1.0


def test_rubric_produces_same_score_as_direct_grading():
    """Rubric score_trajectory should match grade_episode."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)

    actions = [
        OnCallAction(action_type="acknowledge_alert", params={"alert_id": "alert-001"}),
        OnCallAction(action_type="query_logs", params={"service": "payment-api"}),
        OnCallAction(action_type="restart_service", params={"service_name": "payment-api"}),
        OnCallAction(action_type="resolve_incident", params={"resolution_note": "fixed"}),
    ]
    for action in actions:
        obs = env.step(action)

    # Direct grading
    direct_score = grade_episode(
        scenario=env._scenario,
        actions_taken=env._actions_taken,
        alerts_state=env._alerts,
        services_state=env._services,
        severity_set=env._severity,
        summary=env._summary,
        escalated_to=env._escalated_to,
        resolved=env._resolved,
    )

    # Rubric trajectory score
    rubric_score = env._rubric.score_trajectory(env._rubric.trajectory)

    assert direct_score == rubric_score, f"Direct={direct_score} vs Rubric={rubric_score}"


def test_rubric_compute_step_rewards():
    """Rubric compute_step_rewards should return one reward per step."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)

    for _ in range(5):
        env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    env.step(OnCallAction(action_type="resolve_incident", params={"resolution_note": "done"}))

    step_rewards = env._rubric.compute_step_rewards()
    assert len(step_rewards) == 6  # 5 queries + 1 resolve
    assert all(isinstance(r, float) for r in step_rewards)


# ── Dynamic Simulation Tests ──────────────────────────────────────────────

def test_services_degrade_over_steps():
    """Degraded services should get worse if not remediated."""
    env = OnCallEnvironment()
    env.reset(task_id=1, scenario_idx=0)

    # payment-api starts at error_rate=12.3, status=degraded
    initial_error_rate = None
    for svc in env._services:
        if svc["name"] == "payment-api":
            initial_error_rate = svc["error_rate"]
            break

    # Take 5 non-remediation steps
    for _ in range(5):
        env.step(OnCallAction(action_type="query_logs", params={"service": "user-service"}))

    current_error_rate = None
    for svc in env._services:
        if svc["name"] == "payment-api":
            current_error_rate = svc["error_rate"]
            break

    assert current_error_rate > initial_error_rate, (
        f"Expected degradation: initial={initial_error_rate}, current={current_error_rate}"
    )


def test_cascading_recovery_after_fix():
    """Fixing root cause should improve downstream services whose deps are all healthy."""
    from oncall_env.server.simulator import propagate_recovery

    # Simulate: api-gateway depends on [order-service]. order-service is healthy.
    # api-gateway is degraded. After fixing order-service, api-gateway should recover.
    services = [
        {"name": "order-service", "status": "healthy", "error_rate": 0.0, "latency_ms": 50},
        {"name": "api-gateway", "status": "degraded", "error_rate": 30.0, "latency_ms": 2000},
    ]
    deps = {"api-gateway": ["order-service"]}

    propagate_recovery(services, deps, "order-service")

    gw = next(s for s in services if s["name"] == "api-gateway")
    assert gw["error_rate"] < 30.0, f"Expected recovery: error_rate={gw['error_rate']}"
    assert gw["latency_ms"] < 2000, f"Expected recovery: latency={gw['latency_ms']}"


def test_status_transition_degraded_to_down():
    """Services should transition from degraded to down when error_rate is extreme."""
    from oncall_env.server.simulator import degrade_services

    services = [{"name": "test-svc", "status": "degraded", "error_rate": 75.0, "latency_ms": 1000}]
    # After enough degradation steps, should transition to down
    for _ in range(10):
        degrade_services(services, {})
    assert services[0]["status"] == "down"
    assert services[0]["error_rate"] == 100.0


# ── Partial Observability Tests ───────────────────────────────────────────

def test_partial_observability():
    """Uninvestigated services should show aliases and unknown status."""
    env = OnCallEnvironment()
    obs = env.reset(task_id=1, scenario_idx=0)

    # Before investigation: all services should be aliases with "unknown" status
    for svc in obs.services:
        name = _attr(svc, "name")
        assert _attr(svc, "status") == "unknown", f"Service {name} should be 'unknown' before investigation"
        assert name.startswith("service-"), f"Uninvestigated service should show alias, got '{name}'"

    # After investigating payment-api: it should show real name + full details
    obs = env.step(OnCallAction(action_type="query_logs", params={"service": "payment-api"}))
    found_payment = False
    for svc in obs.services:
        name = _attr(svc, "name")
        if name == "payment-api":
            assert _attr(svc, "status") != "unknown", "payment-api should be revealed after query_logs"
            found_payment = True
        elif not name.startswith("service-"):
            # Other investigated services (shouldn't happen here)
            pass
        else:
            assert _attr(svc, "status") == "unknown", f"{name} should still be unknown"
    assert found_payment, "payment-api should appear by real name after investigation"


def test_alert_redaction():
    """Alert messages should not contain exact metric values or real service names."""
    env = OnCallEnvironment()
    obs = env.reset(task_id=1, scenario_idx=0)

    real_service_names = {s["name"] for s in env._services}

    for alert in obs.alerts:
        msg = _attr(alert, "message")
        svc = _attr(alert, "service")
        # Should not contain percentage numbers like "12.3%"
        assert not re.search(r'\d+\.\d+%', msg), f"Alert contains exact percentage: {msg}"
        # Service field should be anonymized (service-A, service-B, etc.)
        assert svc.startswith("service-"), f"Alert service not anonymized: {svc}"
        # Real service names should not appear in the service field
        assert svc not in real_service_names, f"Alert service field contains real name: {svc}"


# ── Supplementary Metrics Tests ───────────────────────────────────────────

def test_egar_metric():
    """EGAR should score higher when agent investigates before remediating."""
    from oncall_env.server.graders import _compute_egar

    # Agent investigated payment-api before restarting it
    good_actions = [
        {"action_type": "query_logs", "params": {"service": "payment-api"}, "step": 1},
        {"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 2},
    ]
    # Agent restarted without investigating
    bad_actions = [
        {"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 1},
    ]

    assert _compute_egar(good_actions) == 1.0
    assert _compute_egar(bad_actions) == 0.0


def test_blast_radius_metric():
    """Blast radius should be 0 when all remediations are correct."""
    from oncall_env.server.graders import _compute_blast_radius

    scenario = {"valid_remediations": [{"action": "restart_service", "service": "payment-api"}]}

    # Correct remediation
    correct = [{"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 1}]
    assert _compute_blast_radius(scenario, correct) == 0.0

    # Wrong remediation
    wrong = [{"action_type": "restart_service", "params": {"service_name": "user-service"}, "step": 1}]
    assert _compute_blast_radius(scenario, wrong) == 1.0


def test_premature_resolution_penalty():
    """Resolving without investigation should lower the remediation score."""
    scenario = load_scenario_by_task(1, 0)

    # With investigation
    actions_with_invest = [
        {"action_type": "query_logs", "params": {"service": "payment-api"}, "step": 1},
        {"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 2},
    ]
    # Without investigation
    actions_no_invest = [
        {"action_type": "restart_service", "params": {"service_name": "payment-api"}, "step": 1},
    ]

    r_with = grade_episode(scenario, actions_with_invest, scenario["services"], scenario["services"], "SEV2", "fix", None, True)
    r_without = grade_episode(scenario, actions_no_invest, scenario["services"], scenario["services"], "SEV2", "fix", None, True)

    assert r_with > r_without, f"Investigation should yield higher score: with={r_with} vs without={r_without}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
