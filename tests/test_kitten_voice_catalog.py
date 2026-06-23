import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "kitten_service" / "voice_catalog.json"


class KittenVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_has_all_female_voices_only(self):
        self.assertEqual(len(self.catalog), 4)
        self.assertTrue(all(voice["gender"] == "Female" for voice in self.catalog))

    def test_voice_ids_are_unique(self):
        ids = [voice["id"] for voice in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))

    def test_expected_female_presets_are_present(self):
        expected = {
            "bella": "expr-voice-2-f",
            "luna": "expr-voice-3-f",
            "rosie": "expr-voice-4-f",
            "kiki": "expr-voice-5-f",
        }

        actual = {voice["id"]: voice["internalId"] for voice in self.catalog}

        self.assertEqual(actual, expected)

    def test_internal_ids_are_female_styles(self):
        for voice in self.catalog:
            self.assertTrue(voice["internalId"].endswith("-f"))


if __name__ == "__main__":
    unittest.main()
