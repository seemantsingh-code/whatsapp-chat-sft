# WhatsApp Chat SFT

A standalone, local-first Python package for turning a one-to-one WhatsApp text export into a dataset and fine-tuning a compatible Hugging Face causal language model. It has no Supabase, Ollama, upload, or telemetry integration.

"Any model" here means a text-generation model supported by Transformers' `AutoModelForCausalLM`. Chat models work best when their tokenizer includes a chat template. Common LoRA projection layers are detected automatically; less common architectures can specify their own target modules.

## Install

Python 3.10 or newer is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train]"
```

For data preparation only, install without the optional training dependencies:

```bash
python -m pip install -e .
```

## Prepare WhatsApp data

Export a one-to-one chat as a text file, then run:

```bash
whatsapp-llm prepare \
  --input /path/to/WhatsApp-chat.txt \
  --my-name "Your name in the export" \
  --output dataset
```

The command writes chronological 80/10/10 `train.jsonl`, `validation.jsonl`, and `test.jsonl` splits. Each row has `prompt` and `completion` message arrays. Phone numbers, email addresses, and URLs are masked by default.

Useful options include `--month-first` for MM/DD dates, `--history` to adjust context, `--gap-hours` to segment conversations, `--merge-minutes` to combine consecutive messages, and `--no-redact-contacts` to disable contact masking.

## Fine-tune a model

Choose a model ID or a local Transformers model directory:

```bash
whatsapp-llm train \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset dataset \
  --output outputs/my-adapter
```

The trainer uses LoRA with 4-bit quantization by default. It supports changing the base model and common training settings:

```bash
whatsapp-llm train \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset dataset \
  --output outputs/mistral-adapter \
  --quantization 4bit \
  --dtype bfloat16 \
  --epochs 3
```

4-bit training requires a compatible CUDA PyTorch and bitsandbytes setup. For hardware without CUDA, use `--quantization none` and choose a model that fits in memory. If LoRA layer detection does not recognize an architecture, pass `--lora-target-modules name1,name2`. Some model families require custom targets or training code and are not directly compatible.

## Chat with your adapter

After training, start an interactive session with the adapter directory:

```bash
whatsapp-llm chat --adapter outputs/my-adapter
```

The adapter records which base model it was trained on, and the package loads that base plus the saved tokenizer. Quantization defaults to `auto`: 4-bit on CUDA, otherwise unquantized. Use `/reset` to clear the conversation or `/quit` to exit. You can set `--system`, `--history`, `--max-new-tokens`, `--temperature`, and `--top-p` on the command line.

## Privacy

All preprocessing and training run locally; the package does not upload chats. Contact masking is not full anonymization: names, places, personal events, and distinctive phrases may still identify people. Get consent from everyone whose messages you use, inspect samples before training, keep raw exports out of Git, and treat trained adapters as sensitive. Group exports currently treat all other participants as one `user` role.

## Development

```bash
python -m unittest discover -s tests
```
