import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "sopro_service" / "voice_catalog.json"


class SoproVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_has_reference_clone_only(self):
        self.assertEqual(len(self.catalog), 1)
        self.assertEqual(self.catalog[0]["id"], "reference-clone")
        self.assertEqual(self.catalog[0]["mode"], "clone")

    def test_no_private_files_are_referenced(self):
        text = CATALOG_PATH.read_text(encoding="utf-8").lower()

        self.assertNotRegex(text, r"c:\\users|/users/|desktop|downloads")


if __name__ == "__main__":
    unittest.main()
