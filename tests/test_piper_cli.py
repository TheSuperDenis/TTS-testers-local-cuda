import re
import unittest

from piper_service.cli import DEFAULT_MODEL_FILE, DEFAULT_VOICE_NAME, env_bool, output_name, split_piper_text


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class PiperCliTests(unittest.TestCase):
    def test_default_voice_is_british_preset(self):
        self.assertIn("en_GB", DEFAULT_MODEL_FILE)
        self.assertIn("Cori", DEFAULT_VOICE_NAME)

    def test_env_bool_accepts_common_true_values(self):
        self.assertTrue(env_bool("PIPER_TEST_BOOL_MISSING", True))

    def test_output_name_uses_piper_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-piper"))

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alice: I need you to slow down and make this sound like a real turn.\n"
            "Beth: I can do that, and I will leave a little breathing room."
        )

        chunks = split_piper_text(text, max_chars=120)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_inline_dialogue_labels_start_new_turns(self):
        text = (
            "Alice: I need this to be one turn. "
            "Beth: This should start a separate turn even without a newline."
        )

        chunks = split_piper_text(text, max_chars=120)

        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_sentence_splits_into_speakable_phrases(self):
        text = (
            "I want this longer sentence to keep its meaning, because the voice should pause around clauses, "
            "and it should not rush through everything as one flat uninterrupted line."
        )

        chunks = split_piper_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))


if __name__ == "__main__":
    unittest.main()
