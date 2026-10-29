import unittest

from whatsapp_llm.chat import _encode_prompt


class PlainTokenizer:
    chat_template = None

    def __call__(self, text, return_tensors):
        return {"text": text, "return_tensors": return_tensors}


class ChatTemplateTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        return {"messages": messages, **kwargs}


class ChatPromptTests(unittest.TestCase):
    def test_plain_prompt_fallback(self):
        result = _encode_prompt(
            PlainTokenizer(),
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )

        self.assertEqual(result["text"], "User: Hi\nAssistant: Hello\nAssistant:")
        self.assertEqual(result["return_tensors"], "pt")

    def test_chat_template_receives_generation_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _encode_prompt(ChatTemplateTokenizer(), messages)

        self.assertEqual(result["messages"], messages)
        self.assertTrue(result["add_generation_prompt"])
        self.assertEqual(result["return_tensors"], "pt")


if __name__ == "__main__":
    unittest.main()