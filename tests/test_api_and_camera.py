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

from bin_config import BinConfig, BinDefinition
from dashboard import DashboardState

from camera_sorter import (
    Classification,
    ChangeDetector,
    DEFAULT_MODEL,
    ProcessingResult,
    build_prompt,
    classify_image,
    crop_frame,
    processing_result,
    run_camera,
)


class ClassificationContractTests(unittest.TestCase):
    def setUp(self):
        self.classification = {
            "decision": "A",
            "item": "aluminum can",
            "material": "aluminum",
            "confidence": 0.95,
            "reason": "The clean metal can meets the policy.",
            "emergency": False,
        }
        self.requests = []

    def classify(self, *, status="completed", refusal=False, model=DEFAULT_MODEL,
                 additional_info="", bin_config=None):
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
            return classify_image(client, b"test-jpeg", model, "our acceptance policy", 0.75,
                                  additional_info, bin_config)

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
        schema = request["text"]["format"]["schema"]
        self.assertIn("emergency", schema["required"])
        self.assertEqual(schema["properties"]["emergency"]["type"], "boolean")
        self.assertEqual(set(schema["properties"]["decision"]["enum"]), {"A", "B", "hold", "empty"})

    def test_low_confidence_primary_bin_is_changed_to_default_fallback(self):
        self.classification["confidence"] = 0.74
        result = self.classify()
        self.assertEqual(result.decision, "B")
        self.assertEqual(result.confidence, 0.74)
        self.assertIn("75%", result.reason)

    def test_confidence_threshold_is_inclusive(self):
        self.classification["confidence"] = 0.75
        self.assertEqual(self.classify().decision, "A")

    def test_low_confidence_default_fallback_bin_is_preserved(self):
        self.classification.update(decision="B", confidence=0.2)
        result = self.classify()
        self.assertEqual(result.decision, "B")
        self.assertEqual(result.confidence, 0.2)

    def test_custom_fallback_applies_without_assuming_a_recycling_direction(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components.",)),
                  BinDefinition("B", "Reuse", ("Working components.",))),
            fallback_bin="A", hold_rules=(),
        )
        self.classification.update(decision="B", confidence=0.3)
        self.assertEqual(self.classify(bin_config=custom).decision, "A")

    def test_classifier_schema_rejects_legacy_and_processing_decisions(self):
        for old_decision in ("recycle", "trash", "error", "discarded", "preview"):
            with self.subTest(decision=old_decision):
                self.classification["decision"] = old_decision
                with self.assertRaises(ValidationError):
                    self.classify()

    def test_custom_prompt_uses_only_configured_sorting_restrictions(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components accepted.",)),
                  BinDefinition("B", "Reuse", ("Working components accepted.",))),
            fallback_bin="A", hold_rules=(),
        )
        prompt = build_prompt(custom)
        for expected in ("Repair", "Reuse", "Damaged components accepted.", "Working components accepted."):
            self.assertIn(expected, prompt)
        self.assertNotIn("recycl", prompt.lower())
        self.assertNotIn("food/liquid", prompt.lower())

    def test_non_emergency_hold_uses_configured_fallback_without_inflating_confidence(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components.",)),
                  BinDefinition("B", "Reuse", ("Working components.",))),
            fallback_bin="A", hold_rules=(),
        )
        for config, expected in [(None, "B"), (custom, "A")]:
            with self.subTest(fallback=expected):
                self.classification.update(decision="hold", confidence=0.2, emergency=False,
                                           reason="A hand partly obscures the item.")
                result = self.classify(bin_config=config)
                self.assertEqual(result.decision, expected)
                self.assertEqual(result.item, self.classification["item"])
                self.assertEqual(result.material, self.classification["material"])
                self.assertEqual(result.confidence, 0.2)
                self.assertFalse(result.emergency)
                self.assertIn("fallback", result.reason.lower())

    def test_true_emergency_forces_hold_even_if_model_selected_a_bin(self):
        for reported_decision in ("hold", "A", "B"):
            with self.subTest(reported_decision=reported_decision):
                self.classification.update(decision=reported_decision, emergency=True, confidence=0.2,
                                           reason="Visible flames require immediate intervention.")
                result = self.classify()
                self.assertEqual(result.decision, "hold")
                self.assertTrue(result.emergency)
                self.assertEqual(result.confidence, 0.2)

    def test_classifier_requires_explicit_emergency_assessment(self):
        del self.classification["emergency"]
        with self.assertRaises(ValidationError):
            self.classify()

    def test_processing_outcomes_are_not_emergency_classifications(self):
        for decision in ("error", "discarded", "preview"):
            with self.subTest(decision=decision):
                result = processing_result(decision, "Operational status.")
                self.assertIsInstance(result, ProcessingResult)
                self.assertEqual(result.decision, decision)
                self.assertFalse(result.emergency)
                self.assertEqual(result.confidence, 0)
                with self.assertRaises(ValidationError):
                    ProcessingResult(**{**result.model_dump(), "emergency": True})

    def test_prompt_excludes_body_parts_and_reserves_hold_for_visible_emergencies(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components.",)),
                  BinDefinition("B", "Reuse", ("Working components.",))),
            fallback_bin="B", hold_rules=(),
        )
        prompt = " ".join(build_prompt(custom).lower().split())
        self.assertRegex(prompt, r"ignore (?:incidental )?hands")
        self.assertIn("arms", prompt)
        self.assertIn("body parts", prompt)
        self.assertIn("if only body parts", prompt)
        self.assertIn("return empty", prompt)
        self.assertIn("low confidence are not emergencies", prompt)
        self.assertIn("hold: use only for an immediate", prompt)
        self.assertIn("mere presence of a battery", prompt)
        self.assertIn("does not establish an emergency", prompt)
        self.assertIn("report confidence honestly", prompt)
        self.assertIn("do not inflate", prompt)

    def test_prompt_assigns_all_objects_in_one_image_to_one_combined_load(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components.",)),
                  BinDefinition("B", "Reuse", ("Working components.",))),
            fallback_bin="B", hold_rules=(),
        )
        prompt = " ".join(build_prompt(custom).lower().split())
        self.assertIn("entire load", prompt)
        self.assertIn("one item for counting", prompt)
        self.assertIn("every component", prompt)
        self.assertIn("mixed contents with conflicting destinations", prompt)
        self.assertIn("configured fallback bin", prompt)
        self.assertIn("label should briefly name the whole load", prompt)
        self.assertNotIn("focus on the main item", prompt)

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
                  self.empty, self.empty, self.empty, self.item, self.item]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        # First value establishes the warmup deadline; remaining values timestamp frames.
        clock = [0, 0, 0.1, 1.2, 5, 6, 7.1, 8.2, 9, 10.1]
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
                self.assertEqual(record["decision"], "preview")
                self.assertFalse(record["emergency"])
                self.assertTrue(record["dry_run"])
                self.assertIsNone(record["model"])
                self.assertEqual(record["item_id"], record["capture_id"])
                image_path = Path(record["image_path"])
                self.assertGreater(image_path.stat().st_size, 0)
                self.assertEqual(json.loads(image_path.with_suffix(".json").read_text()), record)
        camera.release.assert_called_once()

    def test_captured_photo_result_remains_valid_when_the_live_scene_changes(self):
        custom = BinConfig(
            bins=(BinDefinition("A", "Repair", ("Damaged components.",)),
                  BinDefinition("B", "Reuse", ("Working components.",))),
            fallback_bin="B", hold_rules=(),
        )
        cases = [(5, False, 1.3), (5, True, 1.3), (0, True, 1.3),
                 (5, False, 6.2), (5, True, 6.2)]
        for cooldown, scene_changed, completion_time in cases:
            with self.subTest(cooldown=cooldown, scene_changed=scene_changed, completion_time=completion_time), \
                    tempfile.TemporaryDirectory() as directory:
                last_frame = self.replacement if scene_changed else self.item
                camera = Mock()
                camera.isOpened.return_value = True
                camera.read.side_effect = [(True, frame) for frame in
                                           [self.empty, self.item, self.item, last_frame]] + [(False, None)]
                completed = Future()
                completed.set_result(Classification(
                    decision="A", item="can", material="metal", confidence=0.95, reason="Accepted.", emergency=False
                ))
                executor = Mock()
                executor.submit.return_value = completed
                output = io.StringIO()
                with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                        patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                        patch("camera_sorter.time.monotonic", side_effect=[0, 0, 0.1, 1.2, completion_time]), \
                        patch("camera_sorter.time.sleep"), \
                        contextlib.redirect_stdout(output):
                    with self.assertRaises(RuntimeError):
                        run_camera(self.args(directory, dry_run=False), ChangeDetector(cooldown_seconds=cooldown), Mock(), "policy",
                                   bin_config=custom)
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["decision"], "A")
                self.assertFalse(records[0]["emergency"])
                self.assertEqual(records[0]["bin_name"], "Repair")
                self.assertEqual(records[0]["item_id"], records[0]["capture_id"])
                executor.submit.assert_called_once()
                camera.release.assert_called_once()

    def test_changes_during_cooldown_do_not_queue_a_photo_or_api_call_at_expiry(self):
        frames = [self.empty, self.item, self.item, self.empty, self.replacement,
                  self.replacement, self.replacement, self.replacement]
        clock = [0, 0, 0.1, 1.2, 2, 3, 6.2, 7.3, 20]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(Path(directory) / "history.sqlite3")
            state.set_next_note("Information for the first item.")
            frame_iterator = iter(enumerate(frames))

            def read_frame():
                try:
                    index, frame = next(frame_iterator)
                except StopIteration:
                    return False, None
                if index == 3:
                    state.set_next_note("Information waiting for the next item.")
                return True, frame

            camera = Mock()
            camera.isOpened.return_value = True
            camera.read.side_effect = read_frame
            completed = Future()
            completed.set_result(Classification(
                decision="A", item="can", material="metal", confidence=0.95, reason="Accepted.", emergency=False
            ))
            executor = Mock()
            executor.submit.return_value = completed
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
                executor.submit.assert_called_once()
                self.assertEqual(claim.call_count, 1)
                self.assertEqual(len(list(Path(directory).glob("*.jpg"))), 1)
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["decision"], "A")
                self.assertEqual(state.get_snapshot()["pending_note"], "Information waiting for the next item.")
            finally:
                state.close()

    def test_delayed_photo_result_counts_its_source_load_even_after_recalibration(self):
        for recalibrate in (False, True):
            with self.subTest(recalibrate=recalibrate), tempfile.TemporaryDirectory() as directory:
                state = DashboardState(Path(directory) / "history.sqlite3")
                if recalibrate:
                    frames = [self.empty, self.item, self.item, self.empty, self.replacement]
                    clock = [0, 0, 0.1, 1.2, 2, 2.1]
                    completion_checks = [False, True]
                else:
                    frames = [self.empty, self.item, self.item, self.item,
                              self.item, self.item, self.replacement]
                    clock = [0, 0, 0.1, 1.2, 2, 6.2, 7.3, 7.4]
                    completion_checks = [False, False, False, True]
                frame_iterator = iter(enumerate(frames))

                def read_frame():
                    try:
                        index, frame = next(frame_iterator)
                    except StopIteration:
                        return False, None
                    if recalibrate and index == 3:
                        state.request_recalibration()
                    return True, frame

                camera = Mock()
                camera.isOpened.return_value = True
                camera.read.side_effect = read_frame
                delayed = Mock()
                delayed.done.side_effect = completion_checks
                delayed.result.return_value = Classification(
                    decision="A", item="can", material="metal", confidence=0.95,
                    reason="Accepted source image.", emergency=False,
                )
                executor = Mock()
                executor.submit.return_value = delayed
                output = io.StringIO()
                try:
                    with patch("camera_sorter.cv2.VideoCapture", return_value=camera), \
                            patch("camera_sorter.ThreadPoolExecutor", return_value=executor), \
                            patch("camera_sorter.time.monotonic", side_effect=clock), \
                            patch("camera_sorter.time.sleep"), \
                            patch.object(state, "publish_frame"), \
                            contextlib.redirect_stdout(output):
                        with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
                            run_camera(self.args(directory, dry_run=False), ChangeDetector(), Mock(), "policy", state)
                    records = [json.loads(line) for line in output.getvalue().splitlines()]
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["decision"], "A")
                    self.assertEqual(records[0]["item_id"], records[0]["capture_id"])
                    self.assertEqual(state.get_snapshot()["counts"], {"A": 1, "B": 0, "total": 1})
                    self.assertEqual(state.get_snapshot()["items"][0]["capture_id"], records[0]["capture_id"])
                    executor.submit.assert_called_once()
                    camera.release.assert_called_once()
                finally:
                    state.close()

    def test_failed_api_attempt_emits_error_without_retrying_unchanged_scene(self):
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
        self.assertEqual(records[0]["decision"], "error")
        self.assertFalse(records[0]["emergency"])
        executor.submit.assert_called_once()

    def test_each_captured_load_gets_its_own_count_and_note_without_an_empty_platform(self):
        batch = self.item.copy()
        batch[105:140, 25:65] = 255
        frames = [self.empty, batch, batch, batch, batch,
                  self.replacement, self.replacement, self.replacement]
        clock = [0, 0, 0.1, 1.2, 1.3, 6.2, 6.3, 7.4, 7.5]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        output = io.StringIO()
        submitted_notes = []
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(Path(directory) / "history.sqlite3")
            state.set_next_note("First captured load: two clean aluminum cans.")

            def complete_request(function, client, jpeg, model, prompt, min_confidence,
                                 additional_info="", bin_config=None):
                submitted_notes.append(additional_info)
                first = len(submitted_notes) == 1
                if first:
                    state.set_next_note("Second captured load: wax-lined cup.")
                future = Future()
                future.set_result(Classification(
                    decision="A" if first else "B",
                    item="two aluminum cans" if first else "paper cup",
                    material="metal" if first else "lined paper",
                    confidence=0.95,
                    reason="Classified the entire captured load.",
                    emergency=False,
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
                    "First captured load: two clean aluminum cans.",
                    "Second captured load: wax-lined cup.",
                ])
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([record["decision"] for record in records], ["A", "B"])
                self.assertEqual(records[0]["item"], "two aluminum cans")
                self.assertNotEqual(records[0]["capture_id"], records[1]["capture_id"])
                self.assertTrue(all(record["item_id"] == record["capture_id"] for record in records))
                self.assertEqual([record["additional_info"] for record in records], submitted_notes)
                snapshot = state.get_snapshot()
                self.assertEqual(snapshot["counts"], {"B": 1, "A": 1, "total": 2})
                self.assertEqual(len(snapshot["items"]), 2)
                self.assertEqual(snapshot["pending_note"], "")
                self.assertEqual(executor.submit.call_count, 2)
            finally:
                state.close()


if __name__ == "__main__":
    unittest.main()
