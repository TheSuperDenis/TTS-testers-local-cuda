import re
import unittest

from supertonic_service.cli import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VOICE_ID,
    output_name,
    select_onnx_providers,
    split_supertonic_text,
)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class SupertonicCliTests(unittest.TestCase):
    def test_defaults_use_supertonic3_female_preset(self):
        self.assertEqual(DEFAULT_MODEL_NAME, "supertonic-3")
        self.assertEqual(DEFAULT_VOICE_ID, "F1")

    def test_cuda_provider_is_required_when_requested(self):
        providers = select_onnx_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            use_cuda=True,
        )

        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_missing_cuda_provider_fails_instead_of_silent_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "does not expose CUDAExecutionProvider"):
            select_onnx_providers(["CPUExecutionProvider"], use_cuda=True)

    def test_dialogue_lines_stay_separate(self):
        text = (
            "Alice: Supertonic should preserve this first dialogue turn.\n"
            "Beth: This reply should be synthesized as its own chunk."
        )

        chunks = split_supertonic_text(text, max_chars=120)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Alice:"))
        self.assertTrue(any(chunk.startswith("Beth:") for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_long_text_stays_under_chunk_limit(self):
        text = " ".join(f"word{index}" for index in range(70))

        chunks = split_supertonic_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_output_name_uses_supertonic_fallback_slug(self):
        self.assertTrue(output_name("!!!").endswith("-supertonic"))


if __name__ == "__main__":
    unittest.main()
