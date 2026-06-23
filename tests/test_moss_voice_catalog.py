import json
import unittest
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "moss_service" / "voice_catalog.json"


class MossVoiceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_catalog_starts_with_reference_clone(self):
        self.assertEqual(self.catalog[0]["id"], "reference-clone")
        self.assertEqual(self.catalog[0]["mode"], "clone")

    def test_expected_english_female_presets_are_present(self):
        presets = {voice["id"]: voice for voice in self.catalog[1:]}

        self.assertEqual(set(presets), {"ava", "bella"})
        self.assertEqual(presets["ava"]["voiceId"], "Ava")
        self.assertEqual(presets["bella"]["voiceId"], "Bella")
        self.assertTrue(all(voice["group"] == "English Female" for voice in presets.values()))

    def test_no_private_files_are_referenced(self):
        text = CATALOG_PATH.read_text(encoding="utf-8").lower()

        self.assertNotRegex(text, r"c:\\users|/users/|desktop|downloads")


if __name__ == "__main__":
    unittest.main()
