import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "kokoro_service" / "voice_catalog.json"


class KokoroVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_has_all_english_female_voice_counts(self):
        american = [voice for voice in self.catalog if voice["region"] == "american"]
        british = [voice for voice in self.catalog if voice["region"] == "british"]

        self.assertEqual(len(american), 11)
        self.assertEqual(len(british), 4)

    def test_voice_ids_are_unique(self):
        ids = [voice["id"] for voice in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))

    def test_region_matches_voice_prefix_and_lang_code(self):
        expected = {
            "american": ("af_", "a"),
            "british": ("bf_", "b"),
        }

        for voice in self.catalog:
            prefix, lang_code = expected[voice["region"]]
            self.assertTrue(voice["id"].startswith(prefix))
            self.assertEqual(voice["langCode"], lang_code)

    def test_expected_kokoro_presets_are_present(self):
        expected_ids = {
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
        }

        actual_ids = {voice["id"] for voice in self.catalog}

        self.assertEqual(actual_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
