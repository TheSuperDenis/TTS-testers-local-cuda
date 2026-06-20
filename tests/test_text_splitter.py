import re
import unittest

from xtts_service.text_splitter import split_long_text


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class SplitLongTextTests(unittest.TestCase):
    def test_keeps_chunks_under_limit(self):
        text = (
            "This is the first sentence. "
            "This second sentence is intentionally longer so the packer has real work to do. "
            "The final sentence should still be included."
        )

        chunks = split_long_text(text, max_chars=90)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 90 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_splits_long_unpunctuated_text(self):
        text = " ".join(f"word{i}" for i in range(80))

        chunks = split_long_text(text, max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(compact(" ".join(chunks)), compact(text))

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_long_text("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
