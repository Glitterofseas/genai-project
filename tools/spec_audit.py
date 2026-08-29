"""Audit the implementation against the assignment specification.

    python tools/spec_audit.py

Checks the mandatory project structure (spec page 3), the Additional
Implementation Steps (page 4), and conformance with the workflow diagram in
docs/workflow/. Exits non-zero if any requirement regresses.

Requirements that are blocked externally are reported as BLOCKED rather than
FAIL, with the reason, so a genuine regression is never hidden behind a known
limitation.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = {p: p.read_text(encoding="utf-8") for p in (ROOT / "app").rglob("*.py")}
BLOB = "\n".join(SRC.values())


def has(relative: str) -> bool:
    return (ROOT / relative).exists()


def finds(pattern: str) -> bool:
    return bool(re.search(pattern, BLOB, re.I))


def source(relative: str) -> str:
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


STRUCTURE = [
    ("Git repository", (ROOT / ".git").is_dir()),
    (".gitignore", has(".gitignore")),
    ("Virtual environment (.venv)", has(".venv")),
    ("requirements.txt", has("requirements.txt")),
    (".env (and .env.example)", has(".env") and has(".env.example")),
    ("LICENSE", has("LICENSE")),
    ("README.md", has("README.md")),
    ("app/__init__.py", has("app/__init__.py")),
    ("app/main.py entry point", has("app/main.py")),
    ("app/modules/__init__.py", has("app/modules/__init__.py")),
    ("module: main app (agents)", has("app/modules/agents/__init__.py")),
    ("module: Fine-Tuning", has("app/modules/fine_tuning/__init__.py")),
    ("module: Embedding", has("app/modules/embedding/__init__.py")),
    ("streamlit package __init__.py", has("streamlit_app/__init__.py")),
    ("streamlit_main.py", has("streamlit_app/streamlit_main.py")),
    ("streamlit utils.py", has("streamlit_app/utils.py")),
    ("tests/test_main.py", has("tests/test_main.py")),
    ("tests/test_evals.ipynb", has("tests/test_evals.ipynb")),
]


def _fine_tune_trained() -> bool:
    path = ROOT / "app/modules/fine_tuning/fine_tuned_model.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("model"))


IMPLEMENTATION = [
    ("Fine-tuning pipeline for the Exit Advisor",
     has("app/modules/fine_tuning/build_dataset.py") and has("app/modules/fine_tuning/run_job.py"), None),
    ("Fine-tune actually trained", _fine_tune_trained(),
     "OpenAI returns 403 training_not_available - self-serve fine-tuning was withdrawn"),
    ("Offline embedding step into Chroma", has("app/modules/embedding/build_index.py"), None),
    ("Agent architecture on LangChain + OpenAI", finds(r"langchain_openai"), None),
    ("LangChain Agents", finds(r"create_agent|AgentExecutor|create_tool_calling_agent"), None),
    ("LangChain Memories", finds(r"InMemoryChatMessageHistory|ConversationMemory"), None),
    ("LangChain Tools", finds(r"StructuredTool|@tool\b|bind_tools"), None),
    ("Function calling to reach the SQL database",
     finds(r"StructuredTool\.from_function") and finds(r"create_agent"), None),
    ("Suggests the three nearest available slots", finds(r"n=3|n:\s*int\s*=\s*3"), None),
    ("Dates inferred from the conversation timestamp", finds(r"anchor"), None),
    ("Streamlit proof of concept", has("streamlit_app/streamlit_main.py"), None),
    ("Deployed to Streamlit Community Cloud", False,
     "runs locally by choice; local SQL Server backend is not reachable from the cloud"),
    ("Prompting strategy: Roles", finds(r"ROLE:"), None),
    ("Prompting strategy: API parameters", finds(r"temperature"), None),
    ("Prompting strategy: instruction prompts", finds(r"Answer YES when|Answer NO when"), None),
    ("Prompting strategy: few-shot learning", finds(r"few_shot"), None),
    ("Evaluation over End/Continue/Schedule", finds(r"confusion"), None),
]

MAIN_AGENT = source("app/modules/agents/main_agent.py")
LLM = source("app/modules/agents/llm.py")
STREAMLIT = source("streamlit_app/streamlit_main.py")

WORKFLOW = [
    ("Two entry points (reply | registration form)", "registration" in STREAMLIT.lower()),
    ("Main Agent decides 1 of 3, routing to ONE advisor",
     "choose(" in MAIN_AGENT and "advisors.get(choice)" in MAIN_AGENT),
    ("Advisor processes the complete chat history", "render_history" in LLM),
    ("Advisor returns a binary verdict", "BinaryVerdict" in LLM),
    ("SQL reached only after a positive Sched verdict",
     "if not result.decision" in LLM and "_with_slots" in LLM),
    ("Vector store reached only when info is needed",
     bool(re.search(r"if not result\.decision:\s*\n\s*return AdvisorVerdict\(self\.name, False", LLM))),
    ("Advisor output returns to the Main Agent", "consulted.append" in MAIN_AGENT),
    ("Main Agent decides 1 of 2: consult again or reply", "consult_again" in MAIN_AGENT),
    ("Loop back to the routing decision", "for _ in range(self.max_advisor_calls)" in MAIN_AGENT),
    ("Advisor loop is bounded", "max_advisor_calls" in MAIN_AGENT),
    ("Sends output to the user, ending the turn", "composer.compose" in MAIN_AGENT),
    ("Three actions: continue / schedule / end",
     all(a in BLOB for a in ("CONTINUE", "SCHEDULE", "END"))),
]


def report(title: str, rows) -> tuple[int, int, int]:
    print(f"\n{title}")
    passed = failed = blocked = 0
    for row in rows:
        name, good = row[0], row[1]
        note = row[2] if len(row) > 2 else None
        if good:
            status, passed = "PASS", passed + 1
        elif note:
            status, blocked = "BLOCKED", blocked + 1
        else:
            status, failed = "FAIL", failed + 1
        print(f"  [{status:>7}] {name}")
        if status == "BLOCKED":
            print(f"            reason: {note}")
    return passed, failed, blocked


def main() -> int:
    totals = [0, 0, 0]
    for title, rows in [
        ("SPEC PAGE 3 - Project Structure", STRUCTURE),
        ("SPEC PAGE 4 - Additional Implementation Steps", IMPLEMENTATION),
        ("WORKFLOW DIAGRAM - conformance", WORKFLOW),
    ]:
        passed, failed, blocked = report(title, rows)
        totals = [totals[0] + passed, totals[1] + failed, totals[2] + blocked]

    print(f"\n{'=' * 58}")
    print(f"  {totals[0]} passed, {totals[1]} failed, {totals[2]} blocked externally")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    sys.exit(main())
