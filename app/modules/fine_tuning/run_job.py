"""Launch and track the Exit Advisor fine-tune.

    python -m app.modules.fine_tuning.run_job --launch    # upload + start
    python -m app.modules.fine_tuning.run_job --status    # poll
    python -m app.modules.fine_tuning.run_job --wait      # poll until finished

The model id goes in fine_tuned_model.json, which is committed - it isn't a
secret, and keeping it only in the gitignored .env would mean anyone cloning
the repo silently falls back to the few-shot advisor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from ..config.settings import PROJECT_ROOT, get_settings
from .build_dataset import TRAIN_PATH, VALID_PATH

MODEL_ID_PATH = Path(__file__).parent / "fine_tuned_model.json"
BASE_MODEL = "gpt-4o-mini-2024-07-18"
TERMINAL = {"succeeded", "failed", "cancelled"}


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.has_api_key:
        raise SystemExit("OPENAI_API_KEY is not set - add it to .env")
    return OpenAI(api_key=settings.openai_api_key)


def _save(payload: dict) -> None:
    existing = json.loads(MODEL_ID_PATH.read_text()) if MODEL_ID_PATH.exists() else {}
    existing.update(payload)
    MODEL_ID_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_model_id() -> str | None:
    """The fine-tuned model id, if a job has succeeded."""
    if not MODEL_ID_PATH.exists():
        return None
    return json.loads(MODEL_ID_PATH.read_text()).get("model")


def launch() -> str:
    client = _client()
    if not TRAIN_PATH.exists():
        raise SystemExit("Run `python -m app.modules.fine_tuning.build_dataset` first")

    print(f"Uploading {TRAIN_PATH.name}...")
    train_file = client.files.create(file=TRAIN_PATH.open("rb"), purpose="fine-tune")
    valid_file = client.files.create(file=VALID_PATH.open("rb"), purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=valid_file.id,
        model=BASE_MODEL,
        suffix="exit-advisor",
    )
    _save({"job_id": job.id, "base_model": BASE_MODEL, "status": job.status})
    print(f"Launched {job.id} (status: {job.status})")
    print("This usually takes 10-30 minutes. Poll with --status.")
    return job.id


def status(job_id: str | None = None) -> dict:
    client = _client()
    job_id = job_id or (
        json.loads(MODEL_ID_PATH.read_text()).get("job_id")
        if MODEL_ID_PATH.exists()
        else None
    )
    if not job_id:
        raise SystemExit("No job id recorded - launch one first")

    job = client.fine_tuning.jobs.retrieve(job_id)
    info = {
        "job_id": job.id,
        "status": job.status,
        "model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
        "error": str(job.error) if getattr(job, "error", None) else None,
    }
    _save({k: v for k, v in info.items() if v is not None})
    return info


def wait(poll_seconds: int = 60, timeout_seconds: int = 3600) -> dict:
    started = time.time()
    while True:
        info = status()
        print(f"  [{time.time() - started:6.0f}s] {info['status']}")
        if info["status"] in TERMINAL or time.time() - started > timeout_seconds:
            return info
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    if args.launch:
        launch()
    if args.wait:
        info = wait()
    elif args.status:
        info = status()
    else:
        return

    print(json.dumps(info, indent=2))
    if info.get("model"):
        print(f"\nFine-tuned model: {info['model']}")
        print(f"Recorded in {MODEL_ID_PATH.relative_to(PROJECT_ROOT)} (committed).")
        print("Evaluate it with:")
        print(f"  python -m app.modules.evaluation.run_eval --system llm --exit-model {info['model']}")


if __name__ == "__main__":
    main()
