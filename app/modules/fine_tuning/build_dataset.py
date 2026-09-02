"""JSONL training set for the fine-tuned Exit Advisor.

    python -m app.modules.fine_tuning.build_dataset

Trained on all `end` labels, not only the disengaged ones. Only 4 of the 15
conversations end that way; the rest end on an accepted slot. Training on
disengagement alone leaves 2 positive examples - below OpenAI's minimum, and
far below anything learnable.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agents.prompts import EXIT_ADVISOR_SYSTEM, render_history
from ..agents.rule_based import ACCEPTANCE, DISINTEREST
from ..config.settings import PROJECT_ROOT
from ..evaluation.dataset import iter_labelled, split

OUTPUT_DIR = PROJECT_ROOT / "data" / "fine_tuning"
TRAIN_PATH = OUTPUT_DIR / "exit_advisor_train.jsonl"
VALID_PATH = OUTPUT_DIR / "exit_advisor_valid.jsonl"

USER_TEMPLATE = "Conversation so far:\n{history}\n\nToday is {anchor}.\nShould this conversation end now?"


def _reason(should_end: bool, last_candidate_message: str) -> str:
    """A consistent rationale, so the model learns one vocabulary, not fifteen."""
    if not should_end:
        return "the candidate is still engaged"
    if DISINTEREST.search(last_candidate_message):
        return "the candidate has disengaged"
    if ACCEPTANCE.search(last_candidate_message):
        return "the candidate accepted a slot; confirm and close"
    return "the conversation has reached its natural end"


def build_examples(conversations) -> list[dict]:
    examples = []
    for conversation, turn in iter_labelled(conversations):
        history = conversation.history_before(turn.turn_id)
        if not history:
            continue
        last_candidate = next(
            (t.text for t in reversed(history) if t.speaker == "candidate"), ""
        )
        should_end = turn.label == "end"
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": EXIT_ADVISOR_SYSTEM},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            history=render_history(history),
                            anchor=turn.timestamp.strftime("%A %d %B %Y"),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "decision": should_end,
                                "reason": _reason(should_end, last_candidate),
                            }
                        ),
                    },
                ]
            }
        )
    return examples


def write(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    train_conversations, test_conversations = split()
    train = build_examples(train_conversations)
    valid = build_examples(test_conversations)

    write(TRAIN_PATH, train)
    write(VALID_PATH, valid)

    positives = sum(
        1 for e in train if json.loads(e["messages"][-1]["content"])["decision"]
    )
    approx_tokens = sum(
        len(m["content"]) for e in train for m in e["messages"]
    ) // 4

    print(f"train: {len(train)} examples ({positives} end / {len(train) - positives} not-end)")
    print(f"       {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"valid: {len(valid)} examples (held-out conversations, never trained on)")
    print(f"       {VALID_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\n~{approx_tokens:,} training tokens per epoch")
    if len(train) < 10:
        print("\nWARNING: OpenAI requires at least 10 training examples.")


if __name__ == "__main__":
    main()
