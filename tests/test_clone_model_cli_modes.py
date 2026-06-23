import unittest

from miotts_service.cli import normalize_voice_mode as normalize_mio_voice_mode
from qwen_service.cli import normalize_voice_mode as normalize_qwen_voice_mode


class CloneModelCliModeTests(unittest.TestCase):
    def test_qwen_voice_mode_aliases(self):
        self.assertEqual(normalize_qwen_voice_mode("reference-clone"), "clone")
        self.assertEqual(normalize_qwen_voice_mode("custom"), "custom_voice")
        self.assertEqual(normalize_qwen_voice_mode("preset"), "custom_voice")

    def test_miotts_voice_mode_aliases(self):
        self.assertEqual(normalize_mio_voice_mode("reference-clone"), "clone")
        self.assertEqual(normalize_mio_voice_mode("official-preset"), "preset")
        self.assertEqual(normalize_mio_voice_mode("preset"), "preset")


if __name__ == "__main__":
    unittest.main()
