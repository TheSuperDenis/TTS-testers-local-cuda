import re
import unittest

from kokoro_service.cli import DEFAULT_VOICE_ID, output_name, split_kokoro_text


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class KokoroCliTests(unittest.TestCase):
    def test_default_voice_is_american_female(self):
        self.assertTrue(DEFAULT_VOICE_ID.startswith("af_"))

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alex: Kokoro should leave room for a natural reply.\n"
            "Blair: Yes, it should not rush through long turns."
        )

        chunks = split_kokoro_text(text, max_chars=120)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alex:"))
        self.assertTrue(any(chunk.startswith("Blair:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_sentence_splits_into_speakable_phrases(self):
        text = (
            "This longer Kokoro sentence should keep its meaning, because the voice can rush on long input, "
            "and shorter chunks make the pacing easier to compare."
        )

        chunks = split_kokoro_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_output_name_uses_kokoro_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-kokoro"))


if __name__ == "__main__":
    unittest.main()
