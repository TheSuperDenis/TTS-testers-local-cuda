import re
import unittest

from pockettts_service.cli import (
    DEFAULT_VOICE_ID,
    DEFAULT_VOICE_MODE,
    env_bool,
    friendly_startup_error,
    normalize_voice_mode,
    output_name,
    split_pocket_text,
)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class PocketTtsCliTests(unittest.TestCase):
    def test_default_model_and_voice_mode(self):
        self.assertEqual(DEFAULT_VOICE_MODE, "clone")
        self.assertEqual(DEFAULT_VOICE_ID, "reference-clone")

    def test_voice_mode_aliases(self):
        self.assertEqual(normalize_voice_mode("reference-clone"), "clone")
        self.assertEqual(normalize_voice_mode("voice_clone"), "clone")
        self.assertEqual(normalize_voice_mode("preset"), "preset")
        self.assertEqual(normalize_voice_mode("predefined"), "preset")

    def test_env_bool_accepts_common_true_values(self):
        self.assertTrue(env_bool("POCKETTTS_TEST_BOOL_MISSING", True))

    def test_output_name_uses_pockettts_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-pockettts"))

    def test_clone_auth_error_is_actionable(self):
        error = friendly_startup_error(
            ValueError(
                "We could not download the weights for the model with voice cloning. "
                "Without voice cloning, you can use our catalog of voices."
            )
        )

        self.assertIn("https://huggingface.co/kyutai/pocket-tts", error)
        self.assertIn("HF_TOKEN", error)

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alice: PocketTTS should keep this as one speaker turn.\n"
            "Beth: This second line should start another turn."
        )

        chunks = split_pocket_text(text, max_chars=120)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_sentence_splits_into_speakable_phrases(self):
        text = (
            "I want this longer sentence to keep its meaning, because PocketTTS should pause around clauses, "
            "and it should not push the whole thought through as one rushed uninterrupted line."
        )

        chunks = split_pocket_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))


if __name__ == "__main__":
    unittest.main()
