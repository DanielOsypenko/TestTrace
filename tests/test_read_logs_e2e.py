import argparse
import json
import logging
import multiprocessing
from pathlib import Path
import unicodedata
from typing import Optional

import pytest

import read_logs

# Optional expected substrings for replies; fill with {question: expected_substring} as needed.
EXPECTED_CONTAINS = {}

# Matrices to exercise all combinations.
MODELS = ["gpt-oss:20b"]
CHUNK_SIZES = [1000, 1250, 1500]
SIMILARITY_TOP_K = [10, 25, 50]
QA_OUTPUT_DIR = Path("qa_logs")


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

    qa_log_override = QA_OUTPUT_DIR / qa_filename
    QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = set(QA_OUTPUT_DIR.glob(f"{qa_log_override.stem}*.jsonl"))

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

    after = set(QA_OUTPUT_DIR.glob(f"{qa_log_override.stem}*.jsonl"))
    new_files = after - before
    if not new_files:
        pytest.fail("qa_log_path override was not created")
    qa_log_path = max(new_files, key=lambda p: p.stat().st_mtime)

    records = _load_jsonl(qa_log_path)

    qa_entries = [r for r in records if "qery" in r and "reply" in r]
    assert len(qa_entries) == 1, "Expected exactly one Q/A entry"
    assert qa_entries[0]["qery"] == question
    actual_reply = qa_entries[0]["reply"].strip()
    duration = qa_entries[0].get("time")
    assert duration is not None and duration >= 0, "Reply time missing or invalid"
    print(f"expected_substring={expected_substring}")
    print(f"actual_reply={actual_reply}")
    print(f"reply_time={duration}")

    def _normalize_dashes(text: str) -> str:
        # Replace any dash-like punctuation with ASCII hyphen for tolerant comparisons.
        return "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in text)

    normalized_expected = _normalize_dashes(expected_substring)
    normalized_actual = _normalize_dashes(actual_reply)

    # Accept numeric or spelled-out number (case-insensitive) for PVC count answers.
    normalized_expected_num = NUMBER_WORDS.get(normalized_expected.lower(), normalized_expected)
    spelled_out = next((word for word, num in NUMBER_WORDS.items() if num == normalized_expected_num), None)
    actual_lower = normalized_actual.lower()

    if (
        expected_substring in actual_reply
        or normalized_expected in normalized_actual
        or normalized_expected_num in normalized_actual
        or (spelled_out is not None and spelled_out in actual_lower)
    ):
        return actual_reply, duration

    pytest.fail(
        f"Model reply missing expected substring '{expected_substring}'. Got '{actual_reply}'."
    )


def _normalize_dashes(text: str) -> str:
    return "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in text)


def _normalize_namespace(value):
    if value in {None, "", "null", "None"}:
        return None
    return _normalize_dashes(str(value))


def _resource_key(item: dict) -> tuple[str, str, Optional[str]]:
    kind = _normalize_dashes(str(item.get("kind", ""))).lower()
    name = _normalize_dashes(str(item.get("name", ""))).lower()
    ns = _normalize_namespace(item.get("namespace"))
    return (kind, name, ns)


def _format_resources(resources: set[tuple[str, str, Optional[str]]]) -> str:
    return ", ".join(f"(kind={k}, name={n}, ns={ns})" for k, n, ns in sorted(resources))


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_single_pvc_question(model, chunk_size, similarity_top_k, tmp_path):
    """Ask one PVC deletion question, expect a precise PVC name reply, then exit."""

    question = "what PVC has stuck in deletion?"
    expected_reply = "pvc-test-ddf9b3ed18f3442b83a9e40d597b12a"
    reply, duration = _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_reply,
        qa_filename=f"qa_single_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )
    assert duration >= 0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_pvc_creation_count(model, chunk_size, similarity_top_k, tmp_path):
    """Ask how many PVCs were created; expect reply to include the number 6."""

    question = "How many PersistentVolumeClaim custom resources created?"
    expected_count = "6"
    reply, duration = _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_count,
        qa_filename=f"qa_pvc_count_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )
    assert duration >= 0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_cluster_version(model, chunk_size, similarity_top_k, tmp_path):
    """Ask for cluster version; expect to see 4.20.0-ec.6 (hyphen or en dash)."""

    question = "what cluster version is used?"
    expected_version = "4.20.0-ec.6"
    reply, duration = _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_version,
        qa_filename=f"qa_cluster_version_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )
    assert duration >= 0


@pytest.mark.parametrize("run", range(1))
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("similarity_top_k", SIMILARITY_TOP_K)
def test_list_all_resources(run, model, chunk_size, similarity_top_k, tmp_path):
    """Ask to list all resources with kind/name/namespace; expect all known resources present."""

    question = """List all Kubernetes/OpenShift resources created during the test in JSON format. 
    For each resource, include: kind, name, and namespace. Name should be unique. 
    Include Namespace objects themselves if they are created in the logs with 'namespace' field null.
    """
    expected_resources = [
        {"kind": "Namespace", "name": "namespace-test-93adc4c6f438450392fa849c5", "namespace": None},
        {"kind": "Pod", "name": "pod-test-rbd-5effae8239d24aadb12a15cc22e", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "Pod", "name": "pod-test-rbd-f3f886226cbd4865b727f60d9a7", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "Pod", "name": "pod-test-rbd-1cacbf51a7284d6aa97c219bf01", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "Deployment", "name": "pod-test-rbd-246da46c93a2435490187f50b8c", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "Deployment", "name": "pod-test-rbd-ffa978959e684efc893bedebc44", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "Deployment", "name": "pod-test-rbd-cb8c50be536a4b40953e1ab6ff5", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "ServiceAccount", "name": "serviceaccount-sa-badd33bb4b6d4bbb8e2182", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-39e8fe4a32ce46e28af72bcd89c5900", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-c79d2728376849e7919d5081bec0f42", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-84cb19f5d00541a8bd8b6051743a545", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-7846e302d2f04ed4bd11eae4f5c2b8e", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-ddf9b3ed18f3442b83a9e40d597b12a", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
        {"kind": "PersistentVolumeClaim", "name": "pvc-test-dbc726b1810b497a8c7810e78656ac4", "namespace": "namespace-test-93adc4c6f438450392fa849c5"},
    ]

    reply, duration = _run_single_question(
        tmp_path,
        question=question,
        expected_substring=expected_resources[0]["name"],
        qa_filename=f"qa_list_resources_{model}_{chunk_size}_{similarity_top_k}.jsonl",
        model=model,
        chunk_size=chunk_size,
        similarity_top_k=similarity_top_k,
    )
    assert duration >= 0

    def _try_parse_array(text: str):
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None

    parsed = _try_parse_array(reply)
    expected_set = {_resource_key(item) for item in expected_resources}

    if parsed is not None:
        actual_set = {_resource_key(item) for item in parsed if isinstance(item, dict)}
        missing = expected_set - actual_set
        if missing:
            pytest.fail(
                "Missing resources in reply: "
                f"{_format_resources(missing)}"
            )
        return

    lower_reply = reply.lower()
    missing_details = []
    for item in expected_resources:
        kind = item["kind"].lower()
        name = item["name"].lower()
        ns = _normalize_namespace(item["namespace"])
        missing_fields = []
        if name not in lower_reply:
            missing_fields.append(f"name='{name}'")
        if kind not in lower_reply:
            missing_fields.append(f"kind='{kind}'")
        if ns is None:
            null_tokens = ['"namespace": null', "'namespace': null"]
            if not any(tok in lower_reply for tok in null_tokens):
                missing_fields.append("namespace=null")
        else:
            if ns.lower() not in lower_reply:
                missing_fields.append(f"namespace='{ns.lower()}'")
        if missing_fields:
            missing_details.append(f"(kind={kind}, name={name}, ns={ns}): missing {', '.join(missing_fields)}")

    if missing_details:
        pytest.fail("Missing expected resource details: " + "; ".join(missing_details))
