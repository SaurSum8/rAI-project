"""Verify API serialization and camera integration without network or hardware."""

import argparse
import base64
from concurrent.futures import Future
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np
from openai import OpenAI
from pydantic import ValidationError

from dashboard import DashboardState

from camera_sorter import (
    Classification,
    ChangeDetector,
    DEFAULT_MODEL,
    classify_image,
    crop_frame,
    run_camera,
)


class ClassificationContractTests(unittest.TestCase):
    def setUp(self):
        self.classification = {
            "decision": "recycle",
            "item": "aluminum can",
            "material": "aluminum",
            "confidence": 0.95,
            "reason": "The clean metal can meets the policy.",
        }
        self.requests = []

    def classify(self, *, status="completed", refusal=False, model=DEFAULT_MODEL, additional_info=""):
        content = (
            {"type": "refusal", "refusal": "Cannot classify this image."}
            if refusal
            else {
                "type": "output_text",
                "text": json.dumps(self.classification),
                "annotations": [],
            }
        )

        def respond(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": model,
                "status": status,
                "output": [{
                    "id": "msg_test",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [content],
                }],
            })

        with OpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            return classify_image(client, b"test-jpeg", model, "our acceptance policy", 0.75, additional_info)

    def test_real_sdk_sends_image_and_strict_schema_and_parses_result(self):
        result = self.classify()
        self.assertEqual(result.model_dump(), self.classification)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["instructions"], "our acceptance policy")
        self.assertFalse(request["store"])
        image = request["input"][0]["content"][1]
        prefix, encoded = image["image_url"].split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertEqual(base64.b64decode(encoded), b"test-jpeg")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])

    def test_low_confidence_recycling_is_changed_to_trash(self):
        self.classification["confidence"] = 0.74
        result = self.classify()
        self.assertEqual(result.decision, "trash")
        self.assertIn("75%", result.reason)

    def test_confidence_threshold_is_inclusive(self):
        self.classification["confidence"] = 0.75
        self.assertEqual(self.classify().decision, "recycle")

    def test_operational_hold_is_preserved(self):
        self.classification.update(decision="hold", confidence=0.2, reason="A hand obscures the item.")
        self.assertEqual(self.classify().decision, "hold")

    def test_refusal_never_produces_a_routing_decision(self):
        with self.assertRaises(RuntimeError):
            self.classify(refusal=True)

    def test_incomplete_response_never_produces_a_routing_decision(self):
        with self.assertRaises(RuntimeError):
            self.classify(status="incomplete")

    def test_schema_rejects_invalid_confidence(self):
        self.classification["confidence"] = 1.2
        with self.assertRaises(ValidationError):
            self.classify()

    def test_alternative_model_does_not_receive_default_model_reasoning_option(self):
        self.classify(model="another-vision-model")
        self.assertNotIn("reasoning", self.requests[0])

    def test_additional_item_information_is_sent_as_user_context(self):
        note = "The bottle previously contained cooking oil."
        self.classify(additional_info=note)
        request = self.requests[0]
        self.assertNotIn(note, request["instructions"])
        text_content = [part["text"] for part in request["input"][0]["content"] if part["type"] == "input_text"]
        self.assertTrue(any(note in text for text in text_content))


class CameraIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.zeros((160, 160, 3), dtype=np.uint8)
        self.item = self.empty.copy()
        self.item[30:90, 30:90] = 255
        self.replacement = self.empty.copy()
        self.replacement[80:140, 90:150] = 255

    @staticmethod
    def args(directory, *, dry_run):
        return argparse.Namespace(
            camera=0,
            warmup_seconds=0,
            roi=None,
            no_preview=True,
            model=DEFAULT_MODEL,
            output_dir=Path(directory),
            dry_run=dry_run,
            min_confidence=0.75,
        )

    def test_crop_uses_platform_pixels(self):
        cropped = crop_frame(self.item, [20, 25, 40, 50])
        np.testing.assert_array_equal(cropped, self.item[25:75, 20:60])

    def test_invalid_roi_is_rejected(self):
        for roi in ([-1, 0, 10, 10], [0, 0, 0, 10], [0, 0, 10, -1], [150, 150, 20, 20]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                crop_frame(self.item, roi)

    def test_dry_run_saves_once_per_item_and_releases_disconnected_camera(self):
        frames = [self.empty, self.item, self.item, self.item,
                  self.empty, self.empty, self.item, self.item]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        # First value establishes the warmup deadline; remaining values timestamp frames.
        clock = [0, 0, 0.1, 1.2, 5, 6, 7.1, 8, 9.1]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                    patch("camera_sorter.time.monotonic", side_effect=clock), \
                    patch("camera_sorter.time.sleep"), \
                    patch("camera_sorter.classify_image") as classify, \
                    contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
                    run_camera(self.args(directory, dry_run=True), ChangeDetector(), None, "policy")
            classify.assert_not_called()
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["capture_id"], records[1]["capture_id"])
            for record in records:
                self.assertEqual(record["decision"], "hold")
                self.assertTrue(record["dry_run"])
                self.assertIsNone(record["model"])
                image_path = Path(record["image_path"])
                self.assertGreater(image_path.stat().st_size, 0)
                self.assertEqual(json.loads(image_path.with_suffix(".json").read_text()), record)
        camera.release.assert_called_once()

    def test_async_result_routes_only_when_current_scene_remains_stable(self):
        for scene_changed in (False, True):
            with self.subTest(scene_changed=scene_changed), tempfile.TemporaryDirectory() as directory:
                last_frame = self.replacement if scene_changed else self.item
                camera = Mock()
                camera.isOpened.return_value = True
                camera.read.side_effect = [(True, frame) for frame in
                                           [self.empty, self.item, self.item, last_frame]] + [(False, None)]
                completed = Future()
                completed.set_result(Classification(
                    decision="recycle", item="can", material="metal", confidence=0.95, reason="Accepted."
                ))
                executor = Mock()
                executor.submit.return_value = completed
                output = io.StringIO()
                with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                        patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                        patch("camera_sorter.time.monotonic", side_effect=[0, 0, 0.1, 1.2, 1.3]), \
                        patch("camera_sorter.time.sleep"), \
                        contextlib.redirect_stdout(output):
                    with self.assertRaises(RuntimeError):
                        run_camera(self.args(directory, dry_run=False), ChangeDetector(), Mock(), "policy")
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["decision"], "hold" if scene_changed else "recycle")
                executor.submit.assert_called_once()
                camera.release.assert_called_once()

    def test_stale_result_retries_after_same_view_settles_and_cooldown_expires(self):
        camera = Mock()
        camera.isOpened.return_value = True
        frames = [self.empty, self.item, self.item, self.replacement,
                  self.item, self.item, self.item, self.item]
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        result = Classification(
            decision="recycle", item="can", material="metal", confidence=0.95, reason="Accepted."
        )
        completions = [Future(), Future()]
        for future in completions:
            future.set_result(result)
        executor = Mock()
        executor.submit.side_effect = completions
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                    patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                    patch("camera_sorter.time.monotonic", side_effect=[0, 0, 0.1, 1.2, 1.3, 1.4, 2.5, 6.3, 6.4]), \
                    patch("camera_sorter.time.sleep"), \
                    contextlib.redirect_stdout(output):
                with self.assertRaises(RuntimeError):
                    run_camera(self.args(directory, dry_run=False), ChangeDetector(), Mock(), "policy")
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record["decision"] for record in records], ["hold", "recycle"])
        self.assertEqual(executor.submit.call_count, 2)

    def test_failed_api_attempt_emits_hold_without_retrying_unchanged_scene(self):
        camera = Mock()
        camera.isOpened.return_value = True
        frames = [self.empty, self.item, self.item, self.item, self.item, self.item]
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        failed = Future()
        failed.set_exception(RuntimeError("connection failed"))
        executor = Mock()
        executor.submit.return_value = failed
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                    patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                    patch("camera_sorter.time.monotonic", side_effect=[0, 0, 0.1, 1.2, 1.3, 5, 8]), \
                    patch("camera_sorter.time.sleep"), \
                    self.assertLogs("camera_sorter", level="ERROR"), \
                    contextlib.redirect_stdout(output):
                with self.assertRaises(RuntimeError):
                    run_camera(self.args(directory, dry_run=False), ChangeDetector(), Mock(), "policy")
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["decision"], "hold")
        executor.submit.assert_called_once()

    def test_item_retry_keeps_note_and_identity_until_empty_platform_starts_next_item(self):
        camera = Mock()
        camera.isOpened.return_value = True
        frames = [self.empty, self.item, self.item, self.replacement,
                  self.item, self.item, self.item, self.item,
                  self.empty, self.empty, self.item, self.item, self.item]
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        clock = [0, 0, 0.1, 1.2, 1.3, 1.4, 2.5, 6.3, 6.4, 7, 8.1, 9, 11.4, 11.5]
        output = io.StringIO()
        submitted_notes = []
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(Path(directory) / "history.sqlite3")
            state.set_next_note("First item: clean aluminum can.")

            def complete_request(*args):
                submitted_notes.append(args[-1])
                if len(submitted_notes) == 1:
                    state.set_next_note("Second item: wax-lined cup.")
                future = Future()
                future.set_result(Classification(
                    decision="trash" if len(submitted_notes) == 3 else "recycle",
                    item="cup" if len(submitted_notes) == 3 else "can",
                    material="paper" if len(submitted_notes) == 3 else "metal",
                    confidence=0.95,
                    reason="Classified with the item's note.",
                ))
                return future

            executor = Mock()
            executor.submit.side_effect = complete_request
            try:
                with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                        patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                        patch("camera_sorter.time.monotonic", side_effect=clock), \
                        patch("camera_sorter.time.sleep"), \
                        patch.object(state, "publish_frame"), \
                        patch.object(state, "claim_next_item", wraps=state.claim_next_item) as claim, \
                        contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
                        run_camera(self.args(directory, dry_run=False), ChangeDetector(), Mock(), "policy", state)
                self.assertEqual(claim.call_count, 2)
                self.assertEqual(submitted_notes, [
                    "First item: clean aluminum can.",
                    "First item: clean aluminum can.",
                    "Second item: wax-lined cup.",
                ])
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([record["decision"] for record in records], ["hold", "recycle", "trash"])
                self.assertEqual(records[0]["item_id"], records[1]["item_id"])
                self.assertNotEqual(records[1]["item_id"], records[2]["item_id"])
                self.assertEqual([record["additional_info"] for record in records], submitted_notes)
                snapshot = state.get_snapshot()
                self.assertEqual(snapshot["counts"], {"trash": 1, "recycle": 1, "total": 2})
                self.assertEqual(snapshot["pending_note"], "")
            finally:
                state.close()


if __name__ == "__main__":
    unittest.main()
