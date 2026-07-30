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
            "de-thorsten": "high",
            "es-ar-daniela": "high",
            "es-mx-claude": "high",
            "gb-alba": "medium",
            "gb-aru": "medium",
            "gb-cori": "high",
            "gb-jenny-dioco": "medium",
            "gb-southern-english-female": "low",
            "kk-issai": "high",
            "us-amy": "medium",
            "us-hfc-female": "medium",
            "us-kathleen": "low",
            "us-kristin": "medium",
            "us-lessac": "high",
            "us-libritts": "high",
            "us-ljspeech": "high",
            "us-ryan": "high",
        }

        actual = {voice["id"]: voice["quality"] for voice in self.catalog}

        self.assertEqual(actual, expected)

    def test_catalog_includes_all_official_high_quality_voice_paths(self):
        expected_high_model_files = {
            "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
            "en/en_GB/cori/high/en_GB-cori-high.onnx",
            "en/en_US/lessac/high/en_US-lessac-high.onnx",
            "en/en_US/libritts/high/en_US-libritts-high.onnx",
            "en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
            "en/en_US/ryan/high/en_US-ryan-high.onnx",
            "es/es_AR/daniela/high/es_AR-daniela-high.onnx",
            "es/es_MX/claude/high/es_MX-claude-high.onnx",
            "kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx",
        }

        actual_high_model_files = {
            voice["modelFile"] for voice in self.catalog if voice["quality"] == "high"
        }

        self.assertEqual(actual_high_model_files, expected_high_model_files)


if __name__ == "__main__":
    unittest.main()
