"""Validate the two-bin startup configuration without cameras or API calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from bin_config import DEFAULT_CONFIG_PATH, load_bin_config


def custom_document():
    return {
        "bins": {
            "A": {"name": "Repair", "rules": ["Damaged components."]},
            "B": {"name": "Reuse", "rules": ["Working components."]},
        },
        "fallback_bin": "A",
        "hold_rules": [],
    }


class BinConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "custom-bins.json"

    def tearDown(self):
        self.directory.cleanup()

    def load(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return load_bin_config(self.path)

    def test_default_configuration_preserves_recycling_and_trash_policy(self):
        config = load_bin_config()
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file())
        self.assertEqual(tuple(bin.id for bin in config.bins), ("A", "B"))
        self.assertEqual(config.get_bin("A").name, "Recycling")
        self.assertEqual(config.get_bin("B").name, "Trash")
        self.assertEqual(config.fallback_bin, "B")
        self.assertTrue(config.get_bin("A").rules)
        self.assertTrue(config.hold_rules)

    def test_custom_names_rules_and_fallback_round_trip(self):
        document = custom_document()
        config = self.load(document)
        self.assertEqual(config.to_dict(), document)
        self.assertEqual(config.get_bin("B").name, "Reuse")
        self.assertEqual(config.get_bin("A").rules, ("Damaged components.",))
        self.assertEqual(config.hold_rules, ())

    def test_invalid_configuration_is_rejected_with_source_path(self):
        invalid = []

        def change(label, mutate):
            document = deepcopy(custom_document())
            mutate(document)
            invalid.append((label, document))

        change("missing bin", lambda value: value["bins"].pop("B"))
        change("extra bin", lambda value: value["bins"].update(C={"name": "Other", "rules": ["Anything"]}))
        change("unknown root field", lambda value: value.update(extra=True))
        change("unknown bin field", lambda value: value["bins"]["A"].update(color="green"))
        change("missing root field", lambda value: value.pop("fallback_bin"))
        change("wrong bins type", lambda value: value.update(bins=[]))
        change("empty name", lambda value: value["bins"]["A"].update(name="  "))
        change("nonstr name", lambda value: value["bins"]["A"].update(name=12))
        change("empty rules", lambda value: value["bins"]["A"].update(rules=[]))
        change("wrong rules type", lambda value: value["bins"]["A"].update(rules="Accept parts"))
        change("blank rule", lambda value: value["bins"]["A"].update(rules=[" "]))
        change("nonstr rule", lambda value: value["bins"]["A"].update(rules=[None]))
        change("unknown fallback", lambda value: value.update(fallback_bin="C"))
        change("wrong fallback type", lambda value: value.update(fallback_bin=True))
        change("wrong hold rules type", lambda value: value.update(hold_rules="Hold hazardous items"))
        change("blank hold rule", lambda value: value.update(hold_rules=[""]))
        invalid.append(("wrong root type", []))
        for label, document in invalid:
            with self.subTest(case=label), self.assertRaises(ValueError) as raised:
                self.load(document)
            self.assertIn(str(self.path), str(raised.exception))

    def test_malformed_json_and_missing_file_errors_identify_path(self):
        self.path.write_text("{bad json", encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            load_bin_config(self.path)
        self.assertIn(str(self.path), str(raised.exception))
        missing = self.path.with_name("missing.json")
        with self.assertRaises((OSError, ValueError)) as raised:
            load_bin_config(missing)
        self.assertIn(str(missing), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
