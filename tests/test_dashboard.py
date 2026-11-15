"""Exercise dashboard state and HTTP routes without a server or camera."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dashboard import DashboardState, create_app


def item_record(item_id="item-1", capture_id="capture-1", decision="trash", **extra):
    return {
        "item_id": item_id,
        "capture_id": capture_id,
        "captured_at": "2026-11-15T12:00:00+00:00",
        "decision": decision,
        "item": "paper cup",
        "material": "lined paper",
        "confidence": 0.9,
        "reason": "Not clearly recyclable.",
        "additional_info": "",
        "dry_run": False,
        **extra,
    }


class DashboardStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.history_file = Path(self.directory.name) / "history.json"
        self.state = DashboardState(self.history_file)

    def tearDown(self):
        self.state.close()
        self.directory.cleanup()

    def test_pending_note_is_claimed_once_under_concurrent_requests(self):
        self.state.set_next_note("This cup is coated with wax.")
        with ThreadPoolExecutor(max_workers=8) as executor:
            claims = list(executor.map(lambda _: self.state.claim_next_item(), range(16)))
        notes = [note for note, _ in claims]
        self.assertEqual(notes.count("This cup is coated with wax."), 1)
        self.assertEqual(notes.count(""), 15)
        self.assertEqual(len({epoch for _, epoch in claims}), 1)
        self.assertEqual(self.state.get_snapshot()["pending_note"], "")

    def test_item_retries_update_one_history_item_and_its_counts(self):
        _, epoch = self.state.claim_next_item()
        self.state.record_result(item_record(), epoch)
        self.state.record_result(item_record(capture_id="capture-2", decision="recycle"), epoch)
        snapshot = self.state.get_snapshot()
        self.assertEqual(snapshot["counts"], {"trash": 0, "recycle": 1, "total": 1})
        self.assertEqual(len(snapshot["items"]), 1)
        self.assertEqual(snapshot["items"][0]["capture_id"], "capture-2")
        self.assertEqual(snapshot["latest_result"]["decision"], "recycle")

    def test_capture_id_deduplicates_records_without_item_id(self):
        _, epoch = self.state.claim_next_item()
        record = item_record()
        del record["item_id"]
        self.state.record_result(record, epoch)
        self.state.record_result({**record, "reason": "Updated explanation."}, epoch)
        self.assertEqual(self.state.get_snapshot()["counts"]["total"], 1)
        self.assertEqual(len(self.state.get_snapshot()["items"]), 1)

    def test_history_and_counts_survive_restart(self):
        _, epoch = self.state.claim_next_item()
        self.state.record_result(item_record(), epoch)
        self.state.record_result(item_record(item_id="item-2", capture_id="capture-2", decision="recycle"), epoch)
        self.state.close()
        restarted = DashboardState(self.history_file)
        try:
            snapshot = restarted.get_snapshot()
            self.assertEqual(snapshot["counts"], {"trash": 1, "recycle": 1, "total": 2})
            self.assertEqual({item["item_id"] for item in snapshot["items"]}, {"item-1", "item-2"})
        finally:
            restarted.close()

    def test_hold_empty_and_dry_run_results_do_not_count_as_sorted_items(self):
        _, epoch = self.state.claim_next_item()
        for record in [item_record(decision="hold"), item_record(decision="empty"),
                       item_record(decision="recycle", dry_run=True)]:
            with self.subTest(record=record):
                self.state.record_result(record, epoch)
                snapshot = self.state.get_snapshot()
                self.assertEqual(snapshot["counts"], {"trash": 0, "recycle": 0, "total": 0})
                self.assertEqual(snapshot["items"], [])
                self.assertEqual(snapshot["latest_result"]["decision"], record["decision"])
                self.assertEqual(snapshot["latest_result"]["dry_run"], record["dry_run"])

    def test_reset_preserves_pending_note_and_discards_in_flight_result(self):
        _, old_epoch = self.state.claim_next_item()
        self.state.record_result(item_record(), old_epoch)
        self.state.set_next_note("A note for the next item.")
        self.state.reset_history()
        self.state.record_result(item_record(item_id="late-result"), old_epoch)
        snapshot = self.state.get_snapshot()
        self.assertEqual(snapshot["counts"], {"trash": 0, "recycle": 0, "total": 0})
        self.assertEqual(snapshot["items"], [])
        self.assertIsNone(snapshot["latest_result"])
        self.assertEqual(snapshot["pending_note"], "A note for the next item.")
        note, new_epoch = self.state.claim_next_item()
        self.assertEqual(note, "A note for the next item.")
        self.assertNotEqual(new_epoch, old_epoch)
        self.state.record_result(item_record(item_id="new-item"), new_epoch)
        self.assertEqual(self.state.get_snapshot()["counts"]["total"], 1)

    def test_recalibration_request_is_consumed_once(self):
        self.assertFalse(self.state.consume_recalibration())
        self.state.request_recalibration()
        self.assertTrue(self.state.consume_recalibration())
        self.assertFalse(self.state.consume_recalibration())

    def test_video_stream_contains_decodable_jpeg_and_stops_after_close(self):
        frame = np.zeros((16, 24, 3), dtype=np.uint8)
        frame[:, :, 1] = 180
        self.state.set_status("Waiting for an item", online=True)
        self.state.publish_frame(frame, "Watching the platform")
        stream = self.state.frames()
        part = next(stream)
        header, jpeg = part.split(b"\r\n\r\n", 1)
        self.assertIn(b"image/jpeg", header)
        decoded = cv2.imdecode(np.frombuffer(jpeg.rstrip(b"\r\n"), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, frame.shape)
        self.assertTrue(self.state.get_snapshot()["camera_online"])
        self.state.close()
        with self.assertRaises(StopIteration):
            next(stream)


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = DashboardState(Path(self.directory.name) / "history.json")
        self.app = create_app(self.state)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.state.close()
        self.directory.cleanup()

    def test_dashboard_and_state_are_reachable(self):
        with self.client.get("/") as page:
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.content_type)
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["counts"], {"trash": 0, "recycle": 0, "total": 0})

    def test_next_item_note_endpoint_accepts_note_at_size_limit(self):
        note = "x" * 2000
        response = self.client.post("/api/next-item", json={"note": note})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/state").get_json()["pending_note"], note)

    def test_next_item_note_endpoint_rejects_invalid_payloads(self):
        for payload in [{"note": "x" * 2001}, {"note": 12}, {"note": None}, {}, []]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/next-item", json=payload).status_code, 400)
        response = self.client.post("/api/next-item", data="{bad json", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_reset_and_recalibration_endpoints_change_state(self):
        _, epoch = self.state.claim_next_item()
        self.state.record_result(item_record(), epoch)
        reset = self.client.post("/api/reset", json={})
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(self.state.get_snapshot()["counts"]["total"], 0)
        recalibrate = self.client.post("/api/recalibrate", json={})
        self.assertEqual(recalibrate.status_code, 200)
        self.assertTrue(self.state.consume_recalibration())


if __name__ == "__main__":
    unittest.main()
