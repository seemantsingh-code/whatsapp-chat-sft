"""Parse WhatsApp text exports and create local prompt/completion JSONL splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


_TIME = r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?"
_DATE = r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}"
_HEADERS = [
    re.compile(
        rf"^(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\s+-\s+(?P<sender>[^:]+):\s*(?P<text>.*)$"
    ),
    re.compile(
        rf"^\[(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\]\s+(?P<sender>[^:]+):\s*(?P<text>.*)$"
    ),
]
_SKIP = re.compile(
    r"^(?:<Media omitted>|image omitted|video omitted|audio omitted|"
    r"sticker omitted|GIF omitted|This message was deleted|"
    r"You deleted this message|Missed voice call|Missed video call)$",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    timestamp: datetime
    sender: str
    text: str


def _parse_timestamp(date: str, time: str, day_first: bool) -> datetime:
    normalized_date = date.replace("-", "/")
    normalized_time = re.sub(r"\s+", " ", time.strip()).upper()
    if len(normalized_date.split("/", maxsplit=1)[0]) == 4:
        date_formats = ("%Y/%m/%d",)
    elif day_first:
        date_formats = ("%d/%m/%Y", "%d/%m/%y")
    else:
        date_formats = ("%m/%d/%Y", "%m/%d/%y")
    time_formats = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p")
    for date_format in date_formats:
        for time_format in time_formats:
            try:
                return datetime.strptime(
                    f"{normalized_date} {normalized_time}",
                    f"{date_format} {time_format}",
                )
            except ValueError:
                continue
    raise ValueError(f"Unrecognized timestamp: {date} {time}")


def parse_chat(path: str | Path, day_first: bool = True) -> tuple[list[ChatMessage], int]:
    """Read Android/iOS .txt exports, joining continuation lines to their message."""
    messages: list[ChatMessage] = []
    unmatched = 0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = None
        for pattern in _HEADERS:
            match = pattern.match(line)
            if match:
                break
        if match is not None:
            fields = match.groupdict()
            try:
                timestamp = _parse_timestamp(fields["date"], fields["time"], day_first)
            except ValueError:
                unmatched += 1
                continue
            messages.append(
                ChatMessage(timestamp, fields["sender"].strip(), fields["text"].strip())
            )
        elif messages and line.strip():
            previous = messages[-1]
            messages[-1] = replace(previous, text=f"{previous.text}\n{line.strip()}".strip())
        elif line.strip():
            unmatched += 1
    return messages, unmatched


def redact(text: str) -> str:
    """Mask common contact details; this is a convenience, not full anonymization."""
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    return _PHONE.sub("[PHONE]", text)


def clean_and_group(
    messages: list[ChatMessage],
    my_name: str,
    merge_minutes: int = 5,
    redact_contacts: bool = True,
) -> list[ChatMessage]:
    """Map the owner's messages to assistant and merge adjacent same-speaker texts."""
    turns: list[ChatMessage] = []
    for message in messages:
        text = message.text.strip()
        if not text or _SKIP.fullmatch(text) or "\u200e" in text:
            continue
        sender = "assistant" if message.sender.casefold() == my_name.casefold() else "user"
        cleaned_text = redact(text) if redact_contacts else text
        current = ChatMessage(message.timestamp, sender, cleaned_text)
        if (
            turns
            and turns[-1].sender == sender
            and 0 <= (current.timestamp - turns[-1].timestamp).total_seconds() <= merge_minutes * 60
        ):
            previous = turns[-1]
            turns[-1] = replace(
                previous,
                timestamp=current.timestamp,
                text=f"{previous.text}\n{current.text}",
            )
        else:
            turns.append(current)
    return turns


def build_examples(
    turns: list[ChatMessage], history: int = 6, gap_hours: int = 12
) -> list[dict[str, Any]]:
    """Create conversational prompt/completion rows from adjacent user/assistant turns."""
    examples: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if index == 0 or turn.sender != "assistant" or turns[index - 1].sender != "user":
            continue
        start = index - 1
        while start > 0 and index - start < history:
            gap = turns[start].timestamp - turns[start - 1].timestamp
            if gap.total_seconds() > gap_hours * 3600:
                break
            start -= 1
        context = turns[start:index]
        if (turn.timestamp - context[-1].timestamp).total_seconds() > gap_hours * 3600:
            continue
        examples.append(
            {
                "prompt": [
                    {"role": item.sender, "content": item.text} for item in context
                ],
                "completion": [{"role": "assistant", "content": turn.text}],
            }
        )
    return examples


def prepare_dataset(
    input_path: str | Path,
    my_name: str,
    output_dir: str | Path = "dataset",
    *,
    day_first: bool = True,
    history: int = 6,
    gap_hours: int = 12,
    merge_minutes: int = 5,
    min_examples: int = 10,
    redact_contacts: bool = True,
) -> dict[str, int]:
    """Parse, clean, and write chronological train/validation/test JSONL splits."""
    messages, unmatched = parse_chat(input_path, day_first=day_first)
    if my_name.casefold() not in {message.sender.casefold() for message in messages}:
        names = sorted({message.sender for message in messages})
        raise ValueError(f"Sender {my_name!r} was not found. Export contains: {names}")

    turns = clean_and_group(messages, my_name, merge_minutes, redact_contacts)
    examples = build_examples(turns, history=history, gap_hours=gap_hours)
    if len(examples) < min_examples:
        raise ValueError(
            f"Only {len(examples)} examples were created; at least {min_examples} are required."
        )

    first_split = int(0.8 * len(examples))
    second_split = int(0.9 * len(examples))
    splits = {
        "train": examples[:first_split],
        "validation": examples[first_split:second_split],
        "test": examples[second_split:],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (destination / f"{split_name}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "parsed_messages": len(messages),
        "grouped_turns": len(turns),
        "unmatched_lines": unmatched,
        **{f"{name}_examples": len(rows) for name, rows in splits.items()},
    }