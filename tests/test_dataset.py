import json
import tempfile
import unittest
from pathlib import Path

from whatsapp_llm.dataset import build_examples, clean_and_group, parse_chat


class WhatsAppDatasetTests(unittest.TestCase):
    def test_parse_group_redact_and_build_example(self):
        export = (
            "1/2/2025, 9:00 AM - Me: Hello\n"
            "1/2/2025, 9:02 AM - Friend: Email friend@example.com\n"
            "continued message with +1 415 555 0134\n"
            "1/2/2025, 9:05 AM - Me: Visit https://example.com\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.txt"
            path.write_text(export, encoding="utf-8")
            messages, unmatched = parse_chat(path)

        self.assertEqual(unmatched, 0)
        turns = clean_and_group(messages, "Me")
        examples = build_examples(turns)

        self.assertEqual(len(examples), 1)
        self.assertIn("[EMAIL]", examples[0]["prompt"][1]["content"])
        self.assertIn("[PHONE]", examples[0]["prompt"][1]["content"])
        self.assertEqual(examples[0]["completion"][0]["content"], "Visit [URL]")
        self.assertEqual(set(examples[0]), {"prompt", "completion"})
        json.dumps(examples[0], ensure_ascii=False)

    def test_parses_bracketed_month_first_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.txt"
            path.write_text("[12/31/2025, 23:59] Me: End of year\n", encoding="utf-8")
            messages, _ = parse_chat(path, day_first=False)

        self.assertEqual(messages[0].timestamp.month, 12)
        self.assertEqual(messages[0].timestamp.day, 31)


if __name__ == "__main__":
    unittest.main()