import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CloneModelVoiceCatalogTests(unittest.TestCase):
    def test_qwen_catalog_starts_with_reference_clone(self):
        catalog = json.loads((ROOT / "qwen_service" / "voice_catalog.json").read_text(encoding="utf-8"))

        self.assertEqual(catalog[0]["id"], "reference-clone")
        self.assertEqual(catalog[0]["mode"], "clone")

    def test_qwen_catalog_contains_only_documented_female_presets_after_clone(self):
        catalog = json.loads((ROOT / "qwen_service" / "voice_catalog.json").read_text(encoding="utf-8"))
        presets = catalog[1:]

        self.assertEqual({voice["speaker"] for voice in presets}, {"Serena", "Vivian", "Sohee", "Ono_Anna"})
        self.assertTrue(all(voice["mode"] == "custom_voice" for voice in presets))
        self.assertTrue(all("female" in voice["description"].lower() for voice in presets))

    def test_miotts_catalog_starts_with_reference_clone_and_includes_en_female(self):
        catalog = json.loads((ROOT / "miotts_service" / "voice_catalog.json").read_text(encoding="utf-8"))

        self.assertEqual(catalog[0]["id"], "reference-clone")
        self.assertEqual(catalog[0]["mode"], "clone")
        self.assertEqual(catalog[1]["presetId"], "en_female")
        self.assertEqual(catalog[1]["mode"], "preset")

    def test_no_private_files_are_referenced_by_catalogs(self):
        catalogs = [
            ROOT / "qwen_service" / "voice_catalog.json",
            ROOT / "miotts_service" / "voice_catalog.json",
        ]

        for catalog_path in catalogs:
            text = catalog_path.read_text(encoding="utf-8").lower()
            self.assertNotRegex(text, r"c:\\users|/users/|desktop|downloads")


if __name__ == "__main__":
    unittest.main()
