"""Load a trained adapter and run an interactive local chat session."""

from __future__ import annotations

from pathlib import Path


def _encode_prompt(tokenizer, messages):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    rendered = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in messages
    )
    return tokenizer(f"{rendered}\nAssistant:", return_tensors="pt")


def _load_adapter(adapter_dir: Path, quantization: str, dtype: str):
    try:
        import torch
        import transformers
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Chat dependencies are missing. Install with: pip install 'whatsapp-chat-sft[train]'"
        ) from error

    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_id = adapter_config.base_model_name_or_path
    if not base_model_id:
        raise ValueError(f"Adapter at {adapter_dir} does not specify its base model.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selected_quantization = quantization
    if selected_quantization == "auto":
        selected_quantization = "4bit" if torch.cuda.is_available() else "none"
    if selected_quantization == "4bit" and not torch.cuda.is_available():
        raise ValueError("4-bit loading requires CUDA. Use --quantization none on this machine.")

    torch_dtypes = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_argument = "dtype" if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5 else "torch_dtype"
    model_kwargs = {dtype_argument: torch_dtypes[dtype], "device_map": "auto"}
    if selected_quantization == "4bit":
        compute_dtype = torch.bfloat16 if dtype in {"auto", "bfloat16"} else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model, torch


def chat_session(
    *,
    adapter_dir: str | Path,
    quantization: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    history_limit: int = 10,
    system_prompt: str | None = None,
) -> None:
    """Start an interactive chat with a locally saved LoRA adapter."""
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        raise ValueError(f"Adapter directory does not exist: {adapter_path}")
    if history_limit < 1:
        raise ValueError("History limit must be at least 1.")
    if max_new_tokens < 1:
        raise ValueError("Maximum new tokens must be at least 1.")
    if temperature < 0:
        raise ValueError("Temperature cannot be negative.")
    if not 0 < top_p <= 1:
        raise ValueError("Top-p must be greater than 0 and at most 1.")

    tokenizer, model, torch = _load_adapter(adapter_path, quantization, dtype)
    history: list[dict[str, str]] = []
    print("Chat ready. Commands: /reset clears history, /quit exits.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            history.clear()
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": text})
        context = history[-history_limit:]
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + context
        inputs = _encode_prompt(tokenizer, messages)
        input_device = model.get_input_embeddings().weight.device
        inputs = {name: value.to(input_device) for name, value in inputs.items()}
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0:
            generation_args.update(temperature=temperature, top_p=top_p)

        with torch.inference_mode():
            output = model.generate(**inputs, **generation_args)
        prompt_length = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()
        print(f"Assistant: {answer}\n")
        history.append({"role": "assistant", "content": answer})
        history = history[-history_limit:]