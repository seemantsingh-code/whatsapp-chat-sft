"""Fine-tune a Transformers causal language model with LoRA or QLoRA."""

from __future__ import annotations

from pathlib import Path


_TARGET_GROUPS = (
    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ("q_proj", "k_proj", "v_proj", "o_proj"),
    ("query_key_value",),
    ("c_attn",),
    ("Wqkv",),
    ("qkv_proj",),
)


def _target_modules(model, override: str | None) -> list[str]:
    if override:
        return [name.strip() for name in override.split(",") if name.strip()]
    module_names = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    for group in _TARGET_GROUPS:
        if all(name in module_names for name in group):
            return list(group)
    for group in _TARGET_GROUPS[2:]:
        if group[0] in module_names:
            return [group[0]]
    raise ValueError(
        "Could not infer attention projection layers for LoRA. "
        "Pass --lora-target-modules with comma-separated module names."
    )


def _plain_prompt(messages: list[dict[str, str]]) -> str:
    lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    return "\n".join(lines) + "\nAssistant:"


def _prepare_for_tokenizer(dataset, tokenizer):
    if tokenizer.chat_template:
        return dataset

    def format_row(row):
        return {
            "prompt": _plain_prompt(row["prompt"]),
            "completion": "\n" + "\n".join(
                message["content"] for message in row["completion"]
            ),
        }

    return dataset.map(format_row, remove_columns=dataset["train"].column_names)


def train_model(
    *,
    model_id: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: float = 2,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    lora_rank: int = 8,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
    target_modules: str | None = None,
) -> None:
    """Train a text causal LM from prepared JSONL data using Transformers and TRL."""
    try:
        import torch
        import transformers
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install 'whatsapp-chat-sft[train]'"
        ) from error

    if quantization == "4bit" and not torch.cuda.is_available():
        raise ValueError("4-bit QLoRA requires a CUDA-enabled PyTorch install; use --quantization none.")
    torch_dtypes = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_argument = "dtype" if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5 else "torch_dtype"
    model_kwargs = {dtype_argument: torch_dtypes[dtype], "device_map": "auto"}
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtypes[dtype]
            if dtype in {"float16", "bfloat16"}
            else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dataset_path = Path(dataset_dir)
    datasets = load_dataset(
        "json",
        data_files={
            "train": str(dataset_path / "train.jsonl"),
            "validation": str(dataset_path / "validation.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = _prepare_for_tokenizer(datasets, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=_target_modules(model, target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        max_length=max_length,
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        optim="paged_adamw_8bit" if quantization == "4bit" else "adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))