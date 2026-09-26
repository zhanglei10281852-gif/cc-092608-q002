from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from app.core.clock import FrozenClock, to_storage
from app.core.errors import ConflictError
from app.database import get_connection
from app.network.rules import DEFAULT_RULES, allocation_for, judge_quality
from app.network.service import NetworkAccelerationService


def scenario_payload(**overrides):
    payload = {
        "code": "gdh-rail",
        "name": "广深高铁",
        "scene_type": "railway",
        "timezone": "Asia/Shanghai",
        "max_concurrent_sessions": 10,
        "capacity_mbps": 3000,
    }
    payload.update(overrides)
    return payload


def app_payload(**overrides):
    payload = {
        "app_code": "video-call",
        "name": "视频通话",
        "category": "video_call",
        "latency_target_ms": 100,
        "packet_loss_target": 0.01,
        "min_downlink_mbps": 8,
        "min_uplink_mbps": 4,
        "default_priority": 70,
    }
    payload.update(overrides)
    return payload


def sample_payload(**overrides):
    payload = {
        "sample_key": "sample-000001",
        "scenario_code": "gdh-rail",
        "segment_code": "gz-sz-01",
        "app_code": "video-call",
        "subscriber_hash": "subscriber-000000000001",
        "device_class": "phone",
        "train_speed_kmh": 300,
        "latency_ms": 350,
        "packet_loss": 0.08,
        "downlink_mbps": 1.5,
        "uplink_mbps": 0.5,
        "observed_at": "2026-09-26T05:30:00Z",
    }
    payload.update(overrides)
    return payload


def entitlement_payload(**overrides):
    payload = {
        "subscriber_hash": sample_payload()["subscriber_hash"],
        "scenario_code": "gdh-rail",
        "product_code": "rail-boost-day",
        "valid_from": "2026-09-26T00:00:00Z",
        "valid_until": "2026-09-27T00:00:00Z",
        "source_order_id": "order-replay-0001",
    }
    payload.update(overrides)
    return payload


def prepare(client):
    scenario = client.post("/api/network/scenarios", json=scenario_payload())
    assert scenario.status_code == 201, scenario.text
    segment = client.post(
        "/api/network/scenarios/gdh-rail/segments",
        json={"code": "gz-sz-01", "name": "广州南至虎门", "sequence_no": 1, "expected_dwell_seconds": 900, "capacity_mbps": 1200},
    )
    assert segment.status_code == 201, segment.text
    app = client.post("/api/network/applications", json=app_payload())
    assert app.status_code == 201, app.text
    policy = client.post("/api/network/scenarios/gdh-rail/policies", json={"rules": DEFAULT_RULES, "actor": "tests"})
    assert policy.status_code == 201, policy.text
    published = client.post(
        f"/api/network/policies/{policy.json()['id']}/publish",
        json={"actor": "tests", "effective_from": "2026-09-26T00:00:00Z"},
    )
    assert published.status_code == 200, published.text
    return {"scenario": scenario.json(), "app": app.json(), "policy": published.json()}


def test_quality_rule_engine_is_deterministic():
    profile = app_payload()
    healthy = judge_quality(sample_payload(latency_ms=80, packet_loss=0.005, downlink_mbps=20, uplink_mbps=8), profile, DEFAULT_RULES)
    assert healthy.degraded is False
    degraded = judge_quality(sample_payload(), profile, DEFAULT_RULES)
    assert degraded.degraded is True
    assert degraded.severity in {"major", "critical"}
    assert set(degraded.reasons) == {"latency", "packet_loss", "downlink", "uplink"}
    allocation = allocation_for(profile, degraded.severity, DEFAULT_RULES)
    assert allocation.downlink_mbps >= profile["min_downlink_mbps"]
    assert allocation.priority > profile["default_priority"]


def test_scenario_app_policy_and_idempotent_sample(client):
    prepare(client)
    first = client.post("/api/network/samples", json=sample_payload())
    assert first.status_code == 202, first.text
    assert first.json()["incident_id"] is not None
    duplicate = client.post("/api/network/samples", json=sample_payload())
    assert duplicate.status_code == 202
    assert duplicate.json()["sample_id"] == first.json()["sample_id"]
    conflict = client.post("/api/network/samples", json=sample_payload(latency_ms=999))
    assert conflict.status_code == 409


def test_acceleration_requires_entitlement_and_releases_capacity(client):
    prepare(client)
    sample = client.post("/api/network/samples", json=sample_payload()).json()
    denied = client.post(f"/api/network/incidents/{sample['incident_id']}/accelerate", json={"actor": "tests"})
    assert denied.status_code == 409
    entitlement = client.post(
        "/api/network/entitlements",
        json={
            "subscriber_hash": sample_payload()["subscriber_hash"],
            "scenario_code": "gdh-rail",
            "product_code": "rail-boost-day",
            "valid_from": "2026-09-26T00:00:00Z",
            "valid_until": "2026-09-27T00:00:00Z",
            "source_order_id": "order-000001",
        },
    )
    assert entitlement.status_code == 201
    started = client.post(f"/api/network/incidents/{sample['incident_id']}/accelerate", json={"actor": "tests"})
    assert started.status_code == 200, started.text
    repeated = client.post(f"/api/network/incidents/{sample['incident_id']}/accelerate", json={"actor": "tests"})
    assert repeated.status_code == 200
    assert repeated.json()["id"] == started.json()["id"]
    finished = client.post(
        f"/api/network/sessions/{started.json()['id']}/finish",
        json={"actor": "tests", "reason": "体验恢复", "result": "completed"},
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"
    assert finished.json()["reservation"]["state"] == "released"
    assert [event["event_type"] for event in finished.json()["events"]] == ["started", "completed"]


def test_expired_session_reopens_incident_with_fixed_clock(client):
    prepare(client)
    client.post(
        "/api/network/entitlements",
        json={
            "subscriber_hash": sample_payload()["subscriber_hash"],
            "scenario_code": "gdh-rail",
            "product_code": "rail-boost-day",
            "valid_from": "2026-09-26T00:00:00Z",
            "valid_until": "2026-09-27T00:00:00Z",
            "source_order_id": "order-000002",
        },
    )
    sample = client.post("/api/network/samples", json=sample_payload(sample_key="sample-000002")).json()
    started = client.post(f"/api/network/incidents/{sample['incident_id']}/accelerate", json={"actor": "tests"}).json()
    connection = get_connection()
    expiry = datetime.fromisoformat(started["expires_at"].replace("Z", "+00:00"))
    service = NetworkAccelerationService(connection, FrozenClock(expiry + timedelta(seconds=1)))
    result = service.expire_sessions("tests")
    assert started["id"] in result["expired"]
    detail = service.get_session(started["id"])
    assert detail["status"] == "expired"
    assert detail["reservation"]["state"] == "released"
    incident = connection.execute("SELECT state FROM quality_incidents WHERE id=?", (sample["incident_id"],)).fetchone()
    assert incident["state"] == "open"


def test_capacity_limit_rejects_second_session(client):
    prepare(client)
    connection = get_connection()
    connection.execute("UPDATE network_segments SET capacity_mbps=20 WHERE code='gz-sz-01'")
    for index in (1, 2):
        subscriber = f"subscriber-{index:018d}"
        client.post(
            "/api/network/entitlements",
            json={
                "subscriber_hash": subscriber,
                "scenario_code": "gdh-rail",
                "product_code": "rail-boost-day",
                "valid_from": "2026-09-26T00:00:00Z",
                "valid_until": "2026-09-27T00:00:00Z",
                "source_order_id": f"order-capacity-{index:03d}",
            },
        )
        sample = client.post("/api/network/samples", json=sample_payload(sample_key=f"sample-capacity-{index:03d}", subscriber_hash=subscriber)).json()
        response = client.post(f"/api/network/incidents/{sample['incident_id']}/accelerate", json={"actor": "tests"})
        if index == 1:
            assert response.status_code == 200
        else:
            assert response.status_code == 409


def test_policy_versions_replace_previous_publication(client):
    prepared = prepare(client)
    changed = {**DEFAULT_RULES, "allocation": {**DEFAULT_RULES["allocation"], "duration_seconds": 240}}
    draft = client.post("/api/network/scenarios/gdh-rail/policies", json={"rules": changed, "actor": "tests"})
    assert draft.status_code == 201
    publish = client.post(
        f"/api/network/policies/{draft.json()['id']}/publish",
        json={"actor": "tests", "effective_from": "2026-09-26T01:00:00Z"},
    )
    assert publish.status_code == 200
    old = get_connection().execute("SELECT state FROM policy_versions WHERE id=?", (prepared["policy"]["id"],)).fetchone()
    assert old["state"] == "retired"


def test_demo_seed_and_summary(client):
    seeded = client.post("/api/network/demo/seed")
    assert seeded.status_code == 200
    repeated = client.post("/api/network/demo/seed")
    assert repeated.status_code == 200
    summary = client.get("/api/network/summary")
    assert summary.status_code == 200
    assert summary.json()["scenarios"]["active"] == 1


def test_entitlement_identical_retry_returns_original_record(client):
    prepare(client)
    first = client.post("/api/network/entitlements", json=entitlement_payload())
    assert first.status_code == 201, first.text
    assert first.json()["replayed"] is False
    # 完全相同的重试（含等价的时区写法）返回原权益，不新增、不修改记录
    replay = client.post(
        "/api/network/entitlements",
        json=entitlement_payload(valid_from="2026-09-26T08:00:00+08:00", valid_until="2026-09-27T08:00:00+08:00"),
    )
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == first.json()["id"]
    assert replay.json()["created_at"] == first.json()["created_at"]
    assert replay.json()["updated_at"] == first.json()["updated_at"]
    rows = get_connection().execute("SELECT COUNT(*) AS total FROM subscriber_entitlements WHERE source_order_id='order-replay-0001'").fetchone()
    assert rows["total"] == 1
    # 新订单号是真实新购，与安全重试可区分
    another = client.post("/api/network/entitlements", json=entitlement_payload(source_order_id="order-replay-0002"))
    assert another.status_code == 201
    assert another.json()["replayed"] is False
    assert another.json()["id"] != first.json()["id"]


def test_entitlement_order_conflict_is_rejected_and_audited(client):
    prepare(client)
    created = client.post("/api/network/scenarios", json=scenario_payload(code="metro-01", name="地铁一号线", scene_type="metro"))
    assert created.status_code == 201
    first = client.post("/api/network/entitlements", json=entitlement_payload())
    assert first.status_code == 201
    cases = [
        ({"subscriber_hash": "subscriber-000000000099"}, ["subscriber_hash"]),
        ({"scenario_code": "metro-01"}, ["scenario_code"]),
        ({"product_code": "rail-boost-week"}, ["product_code"]),
        ({"valid_from": "2026-09-26T06:00:00Z"}, ["valid_from"]),
        ({"valid_until": "2026-09-28T00:00:00Z"}, ["valid_until"]),
        ({"subscriber_hash": "subscriber-000000000099", "product_code": "rail-boost-week"}, ["subscriber_hash", "product_code"]),
    ]
    for overrides, fields in cases:
        response = client.post("/api/network/entitlements", json=entitlement_payload(**overrides))
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["context"]["mismatched_fields"] == fields
        assert error["context"]["entitlement_id"] == first.json()["id"]
        assert error["context"]["source_order_id"] == "order-replay-0001"
    # 原记录未被覆盖
    row = get_connection().execute("SELECT * FROM subscriber_entitlements WHERE source_order_id='order-replay-0001'").fetchone()
    assert row["subscriber_hash"] == sample_payload()["subscriber_hash"]
    assert row["product_code"] == "rail-boost-day"
    assert row["updated_at"] == first.json()["updated_at"]
    total = get_connection().execute("SELECT COUNT(*) AS total FROM subscriber_entitlements").fetchone()
    assert total["total"] == 1
    # 审计保留冲突摘要，且不泄露用户标识取值
    events = get_connection().execute(
        "SELECT * FROM operation_events WHERE resource_type='entitlement' AND event_type='order_conflict' ORDER BY id"
    ).fetchall()
    assert len(events) == len(cases)
    assert all(event["resource_id"] == first.json()["id"] for event in events)
    user_event = json.loads(events[0]["detail_json"])
    assert user_event["mismatched_fields"] == ["subscriber_hash"]
    assert user_event["incoming"] == {}
    assert user_event["recorded"] == {}
    assert "subscriber-000000000099" not in events[0]["detail_json"]
    scenario_event = json.loads(events[1]["detail_json"])
    assert scenario_event["incoming"] == {"scenario_code": "metro-01"}
    assert scenario_event["recorded"] == {"scenario_code": "gdh-rail"}
    product_event = json.loads(events[2]["detail_json"])
    assert product_event["incoming"] == {"product_code": "rail-boost-week"}
    assert product_event["recorded"] == {"product_code": "rail-boost-day"}


def test_entitlement_replay_does_not_revive_cancelled_state(client):
    prepare(client)
    created = client.post("/api/network/entitlements", json=entitlement_payload()).json()
    connection = get_connection()
    connection.execute("UPDATE subscriber_entitlements SET state='cancelled' WHERE id=?", (created["id"],))
    replay = client.post("/api/network/entitlements", json=entitlement_payload())
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["state"] == "cancelled"
    state = connection.execute("SELECT state FROM subscriber_entitlements WHERE id=?", (created["id"],)).fetchone()
    assert state["state"] == "cancelled"


def test_entitlement_concurrent_identical_requests_create_single_record(client):
    prepare(client)
    payload = entitlement_payload()

    def submit():
        return NetworkAccelerationService().add_entitlement(dict(payload))

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: submit(), range(6)))
    assert {item["id"] for item in results} == {results[0]["id"]}
    assert sum(1 for item in results if not item["replayed"]) == 1
    total = get_connection().execute("SELECT COUNT(*) AS total FROM subscriber_entitlements").fetchone()
    assert total["total"] == 1


def test_entitlement_concurrent_conflicting_requests_keep_single_record(client):
    prepare(client)
    base = entitlement_payload()

    def submit(index):
        payload = dict(base)
        if index % 2:
            payload["subscriber_hash"] = f"subscriber-{index:018d}"
        try:
            return NetworkAccelerationService().add_entitlement(payload)
        except ConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))
    successes = [item for item in results if not isinstance(item, ConflictError)]
    conflicts = [item for item in results if isinstance(item, ConflictError)]
    rows = get_connection().execute("SELECT * FROM subscriber_entitlements").fetchall()
    assert len(rows) == 1
    stored = dict(rows[0])
    assert successes
    for item in successes:
        assert item["id"] == stored["id"]
        assert item["subscriber_hash"] == stored["subscriber_hash"]
    for error in conflicts:
        assert error.context["mismatched_fields"]
        assert error.context["entitlement_id"] == stored["id"]
    events = get_connection().execute(
        "SELECT COUNT(*) AS total FROM operation_events WHERE resource_type='entitlement' AND event_type='order_conflict'"
    ).fetchone()
    assert events["total"] == len(conflicts)
