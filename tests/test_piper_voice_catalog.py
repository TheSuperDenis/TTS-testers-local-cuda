import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "piper_service" / "voice_catalog.json"


class PiperVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_has_expected_female_voice_counts(self):
        american = [voice for voice in self.catalog if voice["region"] == "american"]
        british = [voice for voice in self.catalog if voice["region"] == "british"]

        self.assertEqual(len(american), 6)
        self.assertEqual(len(british), 5)

    def test_voice_ids_are_unique(self):
        ids = [voice["id"] for voice in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))

    def test_model_and_config_paths_match_quality(self):
        for voice in self.catalog:
            quality = voice["quality"]

            self.assertIn(f"/{quality}/", voice["modelFile"])
            self.assertTrue(voice["modelFile"].endswith(f"-{quality}.onnx"))
            self.assertEqual(f"{voice['modelFile']}.json", voice["configFile"])

    def test_highest_available_quality_choices_are_encoded(self):
        expected = {
            "gb-alba": "medium",
            "gb-aru": "medium",
            "gb-cori": "high",
            "gb-jenny-dioco": "medium",
            "gb-southern-english-female": "low",
            "us-amy": "medium",
            "us-hfc-female": "medium",
            "us-kathleen": "low",
            "us-kristin": "medium",
            "us-lessac": "high",
            "us-ljspeech": "high",
        }

        actual = {voice["id"]: voice["quality"] for voice in self.catalog}

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
