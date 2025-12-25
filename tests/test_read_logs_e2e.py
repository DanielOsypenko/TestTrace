import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path

import pytest

import read_logs

# Optional expected substrings for replies; fill with {question: expected_substring} as needed.
EXPECTED_CONTAINS = {}

# Matrices to exercise all combinations.
MODELS = ["gemma2:27b", "gpt-oss-safeguard:20b", "codellama:13b-instruct"]
CHUNK_SIZES = [1000, 1250, 1500]
SIMILARITY_TOP_K = [7, 12, 17, 23]


def _latest_qa_log(qa_logs_dir: Path, before: set[Path]) -> Path:
    qa_logs_dir.mkdir(exist_ok=True)
    after_files = set(qa_logs_dir.glob("qa_*.jsonl"))
    new_files = after_files - before
    if not new_files:
        pytest.fail("No new QA log file was created in qa_logs/")
    return max(new_files, key=lambda p: p.stat().st_mtime)


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_inputs(responses):
    it = iter(responses)
    return lambda _prompt="": next(it, "exit")


def _run_analyzer_subprocess(
    question: str,
    qa_log_path: str,
    model: str,
    chunk_size: int,
    similarity_top_k: int,
    log_dir: str,
):
    """Child process entry: patch questions/input, run analyzer, and exit."""
    import builtins

    # Patch inside the subprocess so the main process stays clean.
    read_logs._load_initial_questions = lambda: [question]
    builtins.input = _iter_inputs(["exit"])

    args = argparse.Namespace(
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
        qa_log_path=qa_log_path,
        log_dir=log_dir,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    read_logs.run_analyzer(args)


def _run_single_question(
    tmp_path,
    *,
    question: str,
    expected_substring: str,
    qa_filename: str,
    model: str,
    chunk_size: int,
    similarity_top_k: int,
    timeout_seconds: int = 300,
):
    logs_dir = Path("logs")
    if not logs_dir.exists() or not any(logs_dir.iterdir()):
        pytest.skip("logs/ is missing or empty; provide at least one log file to run this test")

    qa_log_override = tmp_path / qa_filename

    proc = multiprocessing.Process(
        target=_run_analyzer_subprocess,
        args=(
            question,
            str(qa_log_override),
            model,
            chunk_size,
            similarity_top_k,
            str(logs_dir.resolve()),
        ),
    )
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(f"Analyzer timed out after {timeout_seconds}s and was killed")
    if proc.exitcode != 0:
        pytest.fail(f"Analyzer subprocess exited with code {proc.exitcode}")

    assert qa_log_override.exists(), "qa_log_path override was not created"
    records = _load_jsonl(qa_log_override)

    qa_entries = [r for r in records if "qery" in r and "reply" in r]
    assert len(qa_entries) == 1, "Expected exactly one Q/A entry"
    assert qa_entries[0]["qery"] == question
    actual_reply = qa_entries[0]["reply"].strip()
    print(f"expected_substring={expected_substring}")
    print(f"actual_reply={actual_reply}")
    if expected_substring not in actual_reply:
        pytest.fail(
            f"Model reply missing expected substring '{expected_substring}'. Got '{actual_reply}'."
        )
    return actual_reply


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_single_pvc_question(model, chunk_size, similarity_top_k, tmp_path):
    """Ask one PVC deletion question, expect a precise PVC name reply, then exit."""

    question = "what PVC has stuck in deletion?"
    expected_reply = "pvc-test-ddf9b3ed18f3442b83a9e40d597b12a"
    _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_reply,
        qa_filename=f"qa_single_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_pvc_creation_count(model, chunk_size, similarity_top_k, tmp_path):
    """Ask how many PVCs were created; expect reply to include the number 6."""

    question = "How many PersistentVolumeClaim custom resources created?"
    expected_count = "6"
    _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_count,
        qa_filename=f"qa_pvc_count_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )
