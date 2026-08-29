"""Run an evaluation and cache the result to results/.

    python -m app.modules.evaluation.run_eval --system rule    # free
    python -m app.modules.evaluation.run_eval --system llm     # spends tokens
    python -m app.modules.evaluation.run_eval --system llm --exit-model ft:...

Every run is scored on all 15 conversations and the JSON keeps per-turn rows, so
train/held-out slices are computed afterwards without paying twice. The notebook
reads these files rather than re-running the agent.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config.settings import PROJECT_ROOT, TEST_CONVERSATION_IDS
from ..db.store import get_store
from .dataset import load_conversations
from .harness import Report, baseline_reports, evaluate

RESULTS_DIR = PROJECT_ROOT / "results"


def slice_report(report: Report, conversation_ids) -> Report:
    """Re-score an existing run over a subset of conversations - costs nothing."""
    wanted = set(conversation_ids)
    return Report(
        name=f"{report.name} [conversations {min(wanted)}-{max(wanted)}]",
        predictions=[p for p in report.predictions if p.conversation_id in wanted],
    )


def build_agent(system: str, exit_model: str | None):
    store = get_store(read_only=True)
    if system == "rule":
        from ..agents.rule_based import build_rule_based_agent

        return build_rule_based_agent(store), "rule-based agent (no API)"

    from ..agents.llm import build_llm_agent
    from ..embedding.retriever import get_retriever

    label = "LLM agent"
    if exit_model:
        label += " + fine-tuned exit advisor"
    return build_llm_agent(store, get_retriever(), exit_model=exit_model), label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("rule", "llm"), default="rule")
    parser.add_argument("--exit-model", default=None, help="fine-tuned model id")
    parser.add_argument("--tag", default=None, help="filename tag for the cached result")
    args = parser.parse_args()

    conversations = load_conversations()
    agent, label = build_agent(args.system, args.exit_model)

    started = time.time()
    report = evaluate(agent, conversations, name=label)
    elapsed = time.time() - started

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = args.tag or ("llm_finetuned" if args.exit_model else args.system)
    report.to_json(RESULTS_DIR / f"{tag}.json")

    print(report.render())
    print(f"\n  ran in {elapsed:.1f}s")

    held_out = slice_report(report, TEST_CONVERSATION_IDS)
    print()
    print(held_out.render())
    print(
        f"\n  NOTE: {len(held_out.predictions)} held-out turns means one turn is "
        f"worth {100 / len(held_out.predictions):.1f} accuracy points."
    )

    if args.system == "rule":
        print("\n" + "=" * 62)
        for baseline in baseline_reports(conversations):
            print(f"{baseline.accuracy:>7.1%}   {baseline.name}")
        baselines = {b.name: b.accuracy for b in baseline_reports(conversations)}
        Path(RESULTS_DIR / "baselines.json").write_text(
            __import__("json").dumps(baselines, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
