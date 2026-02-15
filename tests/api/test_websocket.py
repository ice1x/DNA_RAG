"""Tests for the WebSocket streaming endpoint."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


class TestWSAnalyze:
    """WS /ws/v1/analyze."""

    def test_full_streaming_pipeline(
        self, client: TestClient, uploaded_file_id: str,
    ):
        with client.websocket_connect("/ws/v1/analyze") as ws:
            ws.send_text(json.dumps({
                "question": "test trait",
                "file_id": uploaded_file_id,
            }))

            events = []
            # Collect all events until "complete" or "error"
            for _ in range(20):
                raw = ws.receive_text()
                event = json.loads(raw)
                events.append(event)
                if event["step"] in ("complete", "error"):
                    break

            steps = [e["step"] for e in events]
            assert "parsing" in steps
            assert "snp_identification" in steps
            assert "filtering" in steps
            assert "interpretation" in steps
            assert "complete" in steps

            # The final "complete" event should contain the result
            complete = events[-1]
            assert complete["step"] == "complete"
            assert complete["data"]["snp_count_matched"] >= 1

    def test_missing_fields(self, client: TestClient):
        with client.websocket_connect("/ws/v1/analyze") as ws:
            ws.send_text(json.dumps({"question": "test"}))
            raw = ws.receive_text()
            event = json.loads(raw)
            assert event["step"] == "error"

    def test_invalid_file_id(self, client: TestClient):
        with client.websocket_connect("/ws/v1/analyze") as ws:
            ws.send_text(json.dumps({
                "question": "test",
                "file_id": "file_nonexistent",
            }))
            raw = ws.receive_text()
            event = json.loads(raw)
            assert event["step"] == "error"

    def test_invalid_json(self, client: TestClient):
        with client.websocket_connect("/ws/v1/analyze") as ws:
            ws.send_text("not json{{{")
            raw = ws.receive_text()
            event = json.loads(raw)
            assert event["step"] == "error"
