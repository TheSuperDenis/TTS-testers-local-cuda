import re
import unittest

from kitten_service.cli import DEFAULT_MODEL_ID, DEFAULT_MODEL_NAME, DEFAULT_VOICE_ID, output_name, split_kitten_text


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class KittenCliTests(unittest.TestCase):
    def test_default_model_and_voice_are_kitten_female(self):
        self.assertEqual(DEFAULT_MODEL_ID, "KittenML/kitten-tts-nano-0.8")
        self.assertEqual(DEFAULT_MODEL_NAME, "Nano 15M")
        self.assertTrue(DEFAULT_VOICE_ID.endswith("-f"))

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alice: KittenTTS should preserve a natural first turn.\n"
            "Beth: And this reply should stay separate for easier pacing."
        )

        chunks = split_kitten_text(text, max_chars=120)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_sentence_splits_into_speakable_phrases(self):
        text = (
            "This longer KittenTTS sentence should keep its meaning, because the voice is easier to compare "
            "when the launcher gives it shorter chunks with pauses between them."
        )

        chunks = split_kitten_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_output_name_uses_kitten_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-kitten"))


if __name__ == "__main__":
    unittest.main()
