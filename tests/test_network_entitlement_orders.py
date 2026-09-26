from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from app.database import get_connection
from app.network.rules import DEFAULT_RULES
from app.network.service import NetworkAccelerationService


def prepare(client):
    for code, name in (("gdh-rail", "广深高铁"), ("metro-01", "城市地铁")):
        response = client.post(
            "/api/network/scenarios",
            json={"code": code, "name": name, "scene_type": "railway", "timezone": "Asia/Shanghai", "max_concurrent_sessions": 10, "capacity_mbps": 3000},
        )
        assert response.status_code == 201, response.text
    policy = client.post("/api/network/scenarios/gdh-rail/policies", json={"rules": DEFAULT_RULES, "actor": "tests"})
    assert policy.status_code == 201, policy.text


def entitlement_payload(**overrides):
    payload = {
        "subscriber_hash": "subscriber-order-00000001",
        "scenario_code": "gdh-rail",
        "product_code": "rail-boost-day",
        "valid_from": "2026-09-26T00:00:00Z",
        "valid_until": "2026-09-27T00:00:00Z",
        "source_order_id": "order-replay-000001",
    }
    payload.update(overrides)
    return payload


def entitlement_events(event_type=None):
    sql = "SELECT * FROM operation_events WHERE resource_type='entitlement'"
    params = ()
    if event_type:
        sql += " AND event_type=?"
        params = (event_type,)
    return get_connection().execute(sql + " ORDER BY id", params).fetchall()


def test_identical_retry_returns_original_record(client):
    prepare(client)
    first = client.post("/api/network/entitlements", json=entitlement_payload())
    assert first.status_code == 201, first.text
    assert first.json()["replayed"] is False

    replay = client.post("/api/network/entitlements", json=entitlement_payload())
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == first.json()["id"]
    assert replay.json()["created_at"] == first.json()["created_at"]
    assert replay.json()["updated_at"] == first.json()["updated_at"]

    # 时间格式不同但等价的重试仍是安全重放
    equivalent = client.post("/api/network/entitlements", json=entitlement_payload(valid_from="2026-09-26T08:00:00+08:00"))
    assert equivalent.status_code == 201
    assert equivalent.json()["replayed"] is True
    assert equivalent.json()["id"] == first.json()["id"]

    rows = get_connection().execute("SELECT * FROM subscriber_entitlements").fetchall()
    assert len(rows) == 1
    assert [row["event_type"] for row in entitlement_events()] == ["created", "replay_served", "replay_served"]


def test_conflicting_reuse_is_rejected_with_field_summary(client):
    prepare(client)
    first = client.post("/api/network/entitlements", json=entitlement_payload())
    assert first.status_code == 201, first.text

    cases = [
        ({"subscriber_hash": "subscriber-order-00000099"}, ["subscriber_hash"]),
        ({"scenario_code": "metro-01"}, ["scenario_code"]),
        ({"product_code": "rail-boost-night"}, ["product_code"]),
        ({"valid_from": "2026-09-26T06:00:00Z"}, ["valid_from"]),
        ({"valid_until": "2026-09-28T00:00:00Z"}, ["valid_until"]),
        ({"subscriber_hash": "subscriber-order-00000099", "product_code": "rail-boost-night"}, ["subscriber_hash", "product_code"]),
    ]
    for overrides, fields in cases:
        response = client.post("/api/network/entitlements", json=entitlement_payload(**overrides))
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "conflict"
        assert error["context"]["mismatched_fields"] == fields
        assert error["context"]["source_order_id"] == "order-replay-000001"
        assert error["context"]["existing_entitlement_id"] == first.json()["id"]
        # 拒绝响应不能泄露任何一方的用户标识
        assert "subscriber-order-00000001" not in response.text
        assert "subscriber-order-00000099" not in response.text

    # 原记录没有被覆盖
    row = get_connection().execute("SELECT * FROM subscriber_entitlements WHERE source_order_id='order-replay-000001'").fetchone()
    assert row["subscriber_hash"] == "subscriber-order-00000001"
    assert row["scenario_id"] == first.json()["scenario_id"]
    assert row["product_code"] == "rail-boost-day"
    assert row["valid_from"] == first.json()["valid_from"]
    assert row["valid_until"] == first.json()["valid_until"]
    assert row["created_at"] == first.json()["created_at"]
    assert row["updated_at"] == first.json()["updated_at"]

    # 每次冲突都留有审计摘要，且不含用户标识
    events = entitlement_events("order_conflict_rejected")
    assert len(events) == len(cases)
    for event, (_, fields) in zip(events, cases):
        detail = json.loads(event["detail_json"])
        assert detail["source_order_id"] == "order-replay-000001"
        assert detail["mismatched_fields"] == fields
        assert "subscriber-order-00000001" not in event["detail_json"]
        assert "subscriber-order-00000099" not in event["detail_json"]
    product_conflict = json.loads(events[2]["detail_json"])
    assert product_conflict["recorded"]["product_code"] == "rail-boost-day"
    assert product_conflict["incoming"]["product_code"] == "rail-boost-night"


def test_genuine_new_purchase_creates_separate_record(client):
    prepare(client)
    first = client.post("/api/network/entitlements", json=entitlement_payload())
    assert first.status_code == 201
    second = client.post(
        "/api/network/entitlements",
        json=entitlement_payload(source_order_id="order-replay-000002", product_code="rail-boost-night"),
    )
    assert second.status_code == 201
    assert second.json()["replayed"] is False
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["product_code"] == "rail-boost-night"
    rows = get_connection().execute("SELECT COUNT(*) AS amount FROM subscriber_entitlements").fetchone()
    assert rows["amount"] == 2
    assert [row["event_type"] for row in entitlement_events()] == ["created", "created"]


def test_cancelled_or_expired_state_is_not_resurrected(client):
    prepare(client)
    created = client.post("/api/network/entitlements", json=entitlement_payload()).json()
    connection = get_connection()

    for state in ("cancelled", "expired"):
        connection.execute("UPDATE subscriber_entitlements SET state=? WHERE id=?", (state, created["id"]))
        replay = client.post("/api/network/entitlements", json=entitlement_payload())
        assert replay.status_code == 201
        assert replay.json()["replayed"] is True
        assert replay.json()["state"] == state
        conflict = client.post("/api/network/entitlements", json=entitlement_payload(product_code="rail-boost-night"))
        assert conflict.status_code == 409
        assert conflict.json()["error"]["context"]["existing_state"] == state
        row = connection.execute("SELECT state,product_code FROM subscriber_entitlements WHERE id=?", (created["id"],)).fetchone()
        assert row["state"] == state
        assert row["product_code"] == "rail-boost-day"


def test_concurrent_identical_orders_create_single_record(client):
    prepare(client)
    payload = entitlement_payload()

    def submit():
        return NetworkAccelerationService().add_entitlement(dict(payload))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit(), range(8)))

    assert {result["id"] for result in results} == {results[0]["id"]}
    assert sum(1 for result in results if result["replayed"] is False) == 1
    row = get_connection().execute("SELECT COUNT(*) AS amount FROM subscriber_entitlements WHERE source_order_id=?", (payload["source_order_id"],)).fetchone()
    assert row["amount"] == 1


def test_concurrent_conflicting_orders_keep_first_record(client):
    prepare(client)
    variants = [entitlement_payload(product_code=f"rail-boost-{index:02d}") for index in range(6)]

    def submit(payload):
        try:
            return ("accepted", NetworkAccelerationService().add_entitlement(dict(payload)))
        except Exception as exc:  # noqa: BLE001 - 需要在线程内收集冲突异常
            return ("rejected", exc)

    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(submit, variants))

    rows = get_connection().execute("SELECT * FROM subscriber_entitlements WHERE source_order_id=?", (variants[0]["source_order_id"],)).fetchall()
    assert len(rows) == 1
    winner = rows[0]["product_code"]
    accepted = [result for kind, result in outcomes if kind == "accepted"]
    rejected = [exc for kind, exc in outcomes if kind == "rejected"]
    assert len(accepted) == 1
    assert accepted[0]["product_code"] == winner
    assert accepted[0]["replayed"] is False
    assert len(rejected) == len(variants) - 1
    for exc in rejected:
        assert getattr(exc, "code", "") == "conflict"
        assert exc.context["mismatched_fields"] == ["product_code"]
    assert len(entitlement_events("order_conflict_rejected")) == len(rejected)
