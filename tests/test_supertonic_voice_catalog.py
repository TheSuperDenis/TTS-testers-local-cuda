import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "supertonic_service" / "voice_catalog.json"


class SupertonicVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_contains_all_official_female_presets_only(self):
        self.assertEqual(
            [voice["internalId"] for voice in self.catalog],
            ["F1", "F2", "F3", "F4", "F5"],
        )
        self.assertTrue(all(voice["gender"] == "Female" for voice in self.catalog))

    def test_voice_ids_and_orders_are_unique(self):
        ids = [voice["id"] for voice in self.catalog]
        orders = [voice["order"] for voice in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(orders), len(set(orders)))


if __name__ == "__main__":
    unittest.main()
