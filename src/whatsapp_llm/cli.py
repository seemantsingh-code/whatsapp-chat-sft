"""Command-line interface for WhatsApp dataset preparation and model training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .dataset import prepare_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="whatsapp-llm",
        description="Prepare WhatsApp exports and fine-tune compatible causal language models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create chronological JSONL splits")
    prepare.add_argument("--input", required=True, help="WhatsApp exported .txt file")
    prepare.add_argument("--my-name", required=True, help="your exact sender label in the export")
    prepare.add_argument("--output", default="dataset", help="output directory")
    prepare.add_argument("--month-first", action="store_true", help="parse ambiguous dates as MM/DD")
    prepare.add_argument("--history", type=int, default=6, help="maximum context turns")
    prepare.add_argument("--gap-hours", type=int, default=12, help="conversation gap limit")
    prepare.add_argument("--merge-minutes", type=int, default=5, help="merge adjacent messages")
    prepare.add_argument("--min-examples", type=int, default=10, help="minimum usable replies")
    prepare.add_argument(
        "--no-redact-contacts",
        action="store_true",
        help="do not mask URLs, email addresses, and phone numbers",
    )

    train = commands.add_parser("train", help="fine-tune a Transformers causal language model")
    train.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    train.add_argument("--dataset", default="dataset", help="directory containing train/validation JSONL")
    train.add_argument("--output", default="outputs/whatsapp-adapter", help="adapter output directory")
    train.add_argument("--epochs", type=float, default=2)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=512)
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    train.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    train.add_argument(
        "--lora-target-modules",
        help="comma-separated projection layer names for architectures not detected automatically",
    )

    chat = commands.add_parser("chat", help="chat with a saved LoRA adapter")
    chat.add_argument("--adapter", required=True, help="directory containing the trained adapter")
    chat.add_argument("--quantization", choices=("auto", "4bit", "none"), default="auto")
    chat.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    chat.add_argument("--max-new-tokens", type=int, default=200)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--history", type=int, default=10, help="maximum messages retained in context")
    chat.add_argument("--system", help="optional system prompt")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        try:
            summary = prepare_dataset(
                args.input,
                args.my_name,
                args.output,
                day_first=not args.month_first,
                history=args.history,
                gap_hours=args.gap_hours,
                merge_minutes=args.merge_minutes,
                min_examples=args.min_examples,
                redact_contacts=not args.no_redact_contacts,
            )
        except (OSError, ValueError) as error:
            parser.exit(2, f"whatsapp-llm: error: {error}\n")
        for key, value in summary.items():
            print(f"{key.replace('_', ' ')}: {value}")
        print(f"Wrote JSONL splits to {args.output}")
        print("Review samples locally for private details before training.")
        return

    if args.command == "chat":
        from .chat import chat_session

        try:
            chat_session(
                adapter_dir=args.adapter,
                quantization=args.quantization,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                history_limit=args.history,
                system_prompt=args.system,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(2, f"whatsapp-llm: error: {error}\n")
        return

    from .training import train_model

    try:
        train_model(
            model_id=args.model,
            dataset_dir=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            quantization=args.quantization,
            dtype=args.dtype,
            target_modules=args.lora_target_modules,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"whatsapp-llm: error: {error}\n")