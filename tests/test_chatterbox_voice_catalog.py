import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "chatterbox_service" / "voice_catalog.json"


class ChatterboxVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_starts_with_reference_clone(self):
        self.assertEqual(self.catalog[0]["id"], "reference-clone")
        self.assertEqual(self.catalog[0]["mode"], "clone")

    def test_builtin_fallback_is_not_claimed_as_preferred_female_preset(self):
        fallback = self.catalog[1]

        self.assertEqual(fallback["id"], "builtin-default")
        self.assertEqual(fallback["mode"], "builtin")
        self.assertIn("does not label", fallback["description"].lower())

    def test_voice_ids_are_unique(self):
        ids = [voice["id"] for voice in self.catalog]

        self.assertEqual(len(ids), len(set(ids)))

    def test_no_private_files_are_referenced(self):
        text = CATALOG_PATH.read_text(encoding="utf-8").lower()

        self.assertNotRegex(text, r"c:\\users|/users/|desktop|downloads")


if __name__ == "__main__":
    unittest.main()
