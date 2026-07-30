import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "kitten_service" / "model_catalog.json"


class KittenModelCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_contains_official_v08_sizes(self):
        expected = {
            "nano": ("KittenML/kitten-tts-nano-0.8", "15M"),
            "micro": ("KittenML/kitten-tts-micro-0.8", "40M"),
            "mini": ("KittenML/kitten-tts-mini-0.8", "80M"),
        }
        actual = {
            model["id"]: (model["modelId"], model["parameters"])
            for model in self.catalog
        }

        self.assertEqual(actual, expected)

    def test_model_ids_and_orders_are_unique(self):
        ids = [model["id"] for model in self.catalog]
        orders = [model["order"] for model in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(orders), len(set(orders)))

    def test_nano_is_first_low_memory_choice(self):
        first = min(self.catalog, key=lambda model: model["order"])

        self.assertEqual(first["id"], "nano")


if __name__ == "__main__":
    unittest.main()
