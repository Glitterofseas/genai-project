"""Verify the supplied source materials are present and unaltered.

    python tools/verify_sources.py

Sizes are those recorded when each file was first seen in the project folder.
See docs/SOURCE_FILES.md for the full provenance table.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED: dict[str, int] = {
    # documents
    "data/Python Developer Job Description.pdf": 99_900,
    "data/sms_conversations.json": 26_957,
    "data/db_Tech.sql": 2_018,
    "docs/README_template_original.txt": 3_483,
    # assignment specification
    "docs/spec/01_project_overview.png": 77_515,
    "docs/spec/02_data_and_resources.png": 177_277,
    "docs/spec/03_project_structure.png": 112_086,
    "docs/spec/04_additional_implementation_steps.png": 149_701,
    "docs/spec/05_main_components.png": 61_122,
    # workflow diagram
    "docs/workflow/01_entry_and_legend.png": 36_598,
    "docs/workflow/02_main_agent_routing.png": 32_776,
    "docs/workflow/03_advisor_internals.png": 55_654,
    "docs/workflow/04_convergence_and_exit.png": 30_495,
}


def check_sizes() -> list[str]:
    problems = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.exists():
            problems.append(f"MISSING  {relative}")
            continue
        actual = path.stat().st_size
        if actual != expected:
            problems.append(f"CHANGED  {relative}: {actual} bytes, expected {expected}")
    return problems


def check_readable() -> list[str]:
    problems = []

    try:
        from pypdf import PdfReader

        pages = len(PdfReader(str(ROOT / "data/Python Developer Job Description.pdf")).pages)
        if pages != 2:
            problems.append(f"PDF has {pages} pages, expected 2")
    except ImportError:
        pass
    except Exception as exc:
        problems.append(f"PDF unreadable: {exc}")

    try:
        data = json.loads((ROOT / "data/sms_conversations.json").read_text(encoding="utf-8"))
        turns = sum(len(c["turns"]) for c in data)
        if len(data) != 15 or turns != 103:
            problems.append(f"JSON has {len(data)} conversations / {turns} turns, expected 15 / 103")
    except Exception as exc:
        problems.append(f"JSON unreadable: {exc}")

    sql = (ROOT / "data/db_Tech.sql").read_text(encoding="utf-8-sig")
    if "CREATE DATABASE Tech" not in sql:
        problems.append("db_Tech.sql no longer creates the Tech database")

    for relative in EXPECTED:
        if not relative.endswith(".png"):
            continue
        path = ROOT / relative
        if not path.exists():
            continue
        header = path.read_bytes()[:24]
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            problems.append(f"{relative} is not a valid PNG")
        else:
            width, height = struct.unpack(">II", header[16:24])
            if width < 100 or height < 100:
                problems.append(f"{relative} has implausible dimensions {width}x{height}")

    return problems


def main() -> int:
    problems = check_sizes() + check_readable()
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print("  " + problem)
        return 1
    print(f"All {len(EXPECTED)} supplied files present, unaltered and readable.")
    print("  4 documents, 5 specification pages, 4 workflow diagram sections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
