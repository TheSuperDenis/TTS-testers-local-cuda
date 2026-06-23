import re
import unittest

from chatterbox_service.cli import DEFAULT_VOICE_MODE, env_bool, normalize_voice_mode, output_name, split_chatterbox_text


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class ChatterboxCliTests(unittest.TestCase):
    def test_default_voice_mode_is_clone(self):
        self.assertEqual(DEFAULT_VOICE_MODE, "clone")

    def test_voice_mode_aliases(self):
        self.assertEqual(normalize_voice_mode("reference-clone"), "clone")
        self.assertEqual(normalize_voice_mode("voice_clone"), "clone")
        self.assertEqual(normalize_voice_mode("builtin-default"), "builtin")
        self.assertEqual(normalize_voice_mode("default"), "builtin")

    def test_env_bool_accepts_common_true_values(self):
        self.assertTrue(env_bool("CHATTERBOX_TEST_BOOL_MISSING", True))

    def test_output_name_uses_chatterbox_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-chatterbox"))

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alice: This Chatterbox chunk should stay as one speaker turn.\n"
            "Beth: This second turn should not get glued to the first one."
        )

        chunks = split_chatterbox_text(text, max_chars=130)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_sentence_splits_into_speakable_phrases(self):
        text = (
            "I want this longer sentence to keep its meaning, because Chatterbox should pause around clauses, "
            "and it should not push the whole thought through as one rushed uninterrupted line."
        )

        chunks = split_chatterbox_text(text, max_chars=130)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 130 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))


if __name__ == "__main__":
    unittest.main()
