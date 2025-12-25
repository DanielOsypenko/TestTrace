#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import shutil
import urllib.request
import time
import json
import re
import argparse
import hashlib
import signal
import logging
from datetime import datetime, timezone
from typing import Optional

# Exit cleanly when piping output (prevents stack traces on broken pipe)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Default model used when user hits Enter in the model prompt.
DEFAULT_MODEL_NAME = "codellama:13b-instruct"
DEFAULT_CHUNK_SIZE = 1250
DEFAULT_SIMILARITY_TOP_K = 7
logger = logging.getLogger(__name__)


# --- Small helpers used throughout -------------------------------------------------

def _utc_now_iso() -> str:
    """UTC timestamp string for logs."""
    # Keep seconds precision for readability and stable diffs.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: str, obj: dict) -> None:
    """Append a single JSON object as a line to a JSONL file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        line = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"❌ Failed to serialize JSONL record: {e}")
        sys.exit(1)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except OSError as e:
        logger.error(f"❌ Failed to write JSONL log to {path}: {e}")
        sys.exit(1)


# Windows-invalid filename characters (we strip control chars separately).
_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/|?*\\]')
_WHITESPACE_RE = re.compile(r"\s+")
_UNDERSCORES_RE = re.compile(r"_+")


def _sanitize_for_filename(text: str, *, max_len: int = 120) -> str:
    """Return a path-component-safe slug derived from arbitrary text.

    Designed for model names like 'codellama:13b-instruct' and safe across
    macOS/Linux/Windows.
    """
    if text is None:
        text = ""

    s = str(text)

    # Remove ASCII control characters (0x00-0x1F) and DEL (0x7F).
    s = "".join(ch for ch in s if (ord(ch) >= 32 and ord(ch) != 127))

    # Replace whitespace with underscores, then replace invalid characters.
    s = _WHITESPACE_RE.sub("_", s.strip())
    s = _INVALID_FILENAME_CHARS_RE.sub("_", s)

    # Remove path separators just in case (defense in depth).
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")

    s = _UNDERSCORES_RE.sub("_", s).strip(" ._-")

    # Windows forbids trailing dots/spaces; also avoid reserved device names.
    s = s.rstrip(" .")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if not s:
        s = "untitled"
    if s.upper() in reserved:
        s = f"_{s}"

    if len(s) > max_len:
        # Preserve uniqueness when truncating.
        digest = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:8]
        keep = max(1, max_len - (1 + len(digest)))
        s = f"{s[:keep]}_{digest}"

    return s


def _prompt_yes_no(prompt: str, *, default_yes: bool = True) -> bool:
    """Prompt user for y/n. Empty input selects default."""
    default_hint = "Y/n" if default_yes else "y/N"
    while True:
        raw = input(f"{prompt} ({default_hint}): ").strip().lower()
        if not raw:
            return default_yes
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        logger.warning("Please enter 'y' or 'n'.")


def _select_log_file_interactively(files_sorted: list[str], *, default_index: int = 1) -> str:
    """Select a file from a sorted list by number (1-based)."""
    if not files_sorted:
        logger.error("❌ No log files available.")
        sys.exit(1)

    default_index = max(1, min(default_index, len(files_sorted)))

    logger.info("\n📄 Available log files:")
    for i, name in enumerate(files_sorted, 1):
        marker = " (default)" if i == default_index else ""
        logger.info(f"[{i}] {name}{marker}")

    while True:
        raw = input(f"Choose a file number (Enter for {default_index}): ").strip()
        if not raw:
            return files_sorted[default_index - 1]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(files_sorted):
                return files_sorted[idx - 1]
        logger.warning(f"Please enter a number between 1 and {len(files_sorted)}.")


def _all_cli_runtime_params_provided(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "model", None)
        and getattr(args, "chunk_size", None) is not None
        and getattr(args, "similarity_top_k", None) is not None
    )


def _prompt_int(prompt: str, *, default: int, min_value: int = 1) -> int:
    while True:
        raw = input(f"{prompt} (Enter for {default}): ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            logger.warning("Please enter a whole number.")
            continue
        if val < min_value:
            logger.warning(f"Please enter a value >= {min_value}.")
            continue
        return val


def _resolve_runtime_settings(runtime_args: argparse.Namespace) -> tuple[str, int, int]:
    """Resolve model name, chunk size, and similarity_top_k.

    If a setting is missing, prompt interactively.
    """
    model = getattr(runtime_args, "model", None)
    chunk_size = getattr(runtime_args, "chunk_size", None)
    top_k = getattr(runtime_args, "similarity_top_k", None)

    if model:
        model_name = model
    else:
        model_name = choose_model_interactively(DEFAULT_MODEL_NAME)

    # Validate/prompt ints.
    if chunk_size is None:
        chunk_size = _prompt_int("Chunk size", default=DEFAULT_CHUNK_SIZE, min_value=50)
    elif chunk_size < 50:
        logger.error("❌ --chunk-size must be >= 50")
        sys.exit(1)

    if top_k is None:
        top_k = _prompt_int("Similarity top-k", default=DEFAULT_SIMILARITY_TOP_K, min_value=1)
    elif top_k < 1:
        logger.error("❌ --similarity-top-k must be >= 1")
        sys.exit(1)

    return model_name, int(chunk_size), int(top_k)


def _maybe_edit_system_prompt_interactively() -> None:
    """Optionally open SYSTEM_PROMPT.txt in $EDITOR for quick edits."""
    # If stdin isn't interactive (piped), skip.
    if not sys.stdin.isatty():
        return

    if not _prompt_yes_no("Edit SYSTEM_PROMPT.txt before running?", default_yes=False):
        return

    path = _prompt_file_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        # Create an empty file so the editor opens it.
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
        except OSError as e:
            logger.error(f"❌ Failed to create {path}: {e}")
            return

    editor = os.environ.get("EDITOR")
    if not editor:
        editor = "nano" if shutil.which("nano") else "vi"

    try:
        subprocess.run([editor, path], check=False)
    except OSError as e:
        logger.warning(f"⚠️ Could not launch editor '{editor}': {e}")


def _venv_paths():
    if platform.system().lower() == "windows":
        pip_exe = os.path.join(".venv", "Scripts", "pip")
        py_exe = os.path.join(".venv", "Scripts", "python")
    else:
        pip_exe = os.path.join(".venv", "bin", "pip")
        py_exe = os.path.join(".venv", "bin", "python")
    return py_exe, pip_exe


def _restart_inside_venv_if_needed():
    py_exe, _ = _venv_paths()
    if os.path.exists(py_exe) and sys.executable != os.path.abspath(py_exe):
        logger.info("🔄 Restarting inside virtual environment...")
        os.execv(py_exe, [py_exe] + sys.argv)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_venv():
    if not os.path.exists(".venv"):
        logger.info("📦 Creating virtual environment...")
        python_cmd = None
        try:
            result = subprocess.run(
                ["pyenv", "which", "python3.11"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                python_cmd = result.stdout.strip()
        except FileNotFoundError:
            pass

        if not python_cmd:
            python_cmd = shutil.which("python3") or sys.executable

        logger.info(f"Using Python: {python_cmd}")
        subprocess.run([python_cmd, "-m", "venv", ".venv"], check=True)

    py_exe, pip_exe = _venv_paths()

    if not os.path.exists(pip_exe):
        logger.info("📦 Installing pip in virtual environment...")
        subprocess.run([py_exe, "-m", "ensurepip", "--upgrade"], check=True)

    subprocess.run([py_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    req_file = os.path.join(os.path.dirname(__file__) or ".", "requirements.txt")
    if not os.path.exists(req_file):
        logger.error(f"❌ requirements.txt not found at: {req_file}")
        sys.exit(1)

    # Avoid slow/fragile installs on every run: install only when requirements.txt changes.
    stamp_path = os.path.join(os.path.dirname(__file__) or ".", ".venv", ".requirements.sha256")
    current_hash = _sha256_file(req_file)
    previous_hash = None
    if os.path.exists(stamp_path):
        try:
            with open(stamp_path, "r", encoding="utf-8") as f:
                previous_hash = f.read().strip()
        except OSError:
            previous_hash = None

    if previous_hash == current_hash:
        return

    logger.info("⬇️ Installing pinned dependencies from requirements.txt...")
    subprocess.run(
        [
            pip_exe,
            "install",
            "-r",
            req_file,
            "--upgrade-strategy",
            "only-if-needed",
            "--quiet",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(current_hash)
    except OSError:
        pass


def ensure_ollama():
    if shutil.which("ollama"):
        logger.info("✅ Ollama is already installed")
        return

    system = platform.system().lower()
    arch = platform.machine().lower()
    logger.info(f"⬇️ Ollama not found, installing for {system} ({arch})...")

    if system == "darwin":
        subprocess.run(["brew", "install", "ollama"], check=True)
    elif system == "linux":
        url = "https://ollama.com/download/OllamaInstaller.sh"
        installer = "/tmp/OllamaInstaller.sh"
        urllib.request.urlretrieve(url, installer)
        subprocess.run(["chmod", "+x", installer], check=True)
        subprocess.run([installer], check=True)
    else:
        logger.error(f"❌ Unsupported OS: {system}")
        sys.exit(1)


def _is_ollama_healthy() -> bool:
    """Return True if the local Ollama server responds."""
    try:
        res = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_ollama_started():
    """Start Ollama service if it isn't running yet."""
    if _is_ollama_healthy():
        logger.info("✅ Ollama is running")
        return

    system = platform.system().lower()

    # On macOS/Linux, `ollama serve` runs a foreground server. We'll launch it detached.
    if system in {"darwin", "linux"}:
        logger.info("▶️ Starting Ollama service...")

        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

        # Wait briefly until the server responds.
        deadline = time.time() + 20
        while time.time() < deadline:
            if _is_ollama_healthy():
                logger.info("✅ Ollama started")
                return
            time.sleep(0.5)

        logger.error("❌ Ollama was started but is not responding yet.")
        logger.info("   Try running `ollama serve` in another terminal to see logs.")
        sys.exit(1)

    # Windows service management is different; we fail with guidance.
    logger.error(f"❌ Unsupported auto-start on OS: {system}")
    logger.info("   Please start Ollama manually, then re-run this script.")
    sys.exit(1)


def ensure_model(model_name: str):
    # Ensure server is running before calling this.
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name in result.stdout:
        logger.info(f"✅ Model already available: {model_name}")
        return

    logger.info(f"⬇️ Downloading model: {model_name}")
    pull = subprocess.run(["ollama", "pull", model_name])
    if pull.returncode == 0:
        logger.info(f"✅ Successfully pulled {model_name}")
        return

    logger.error(f"❌ Failed to pull {model_name}")
    sys.exit(1)


def _parse_ollama_list_models(output: str) -> list[str]:
    """Parse `ollama list` output and return model names."""
    models: list[str] = []
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return models

    # Expected header like: NAME ID SIZE MODIFIED
    # Data lines begin with the model name.
    for line in lines[1:]:
        name = line.split()[0]
        if name and name.lower() != "name":
            models.append(name)
    return models


def choose_model_interactively(default_model: str = DEFAULT_MODEL_NAME) -> str:
    """Prompt user to choose a model from `ollama list`.

    - Shows a numbered list of installed models.
    - Default is `default_model`.
    - If user enters a custom name, it will be used (and pulled if missing).
    """
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    installed = _parse_ollama_list_models(res.stdout)

    logger.info("\n🧠 Ollama models installed:")
    if installed:
        for i, m in enumerate(installed, 1):
            marker = " (default)" if m == default_model else ""
            logger.info(f"[{i}] {m}{marker}")
    else:
        logger.info("(none found)")

    prompt = (
        f"Choose a model by number, or type a model name. "
        f"Press Enter for default [{default_model}]: "
    )
    raw = input(prompt).strip()

    if not raw:
        selected = default_model
    elif raw.isdigit() and installed and 1 <= int(raw) <= len(installed):
        selected = installed[int(raw) - 1]
    else:
        selected = raw

    if selected == default_model and selected not in installed:
        # Per requirement: if default doesn't exist locally, pull it.
        ensure_model(selected)
    elif selected not in installed:
        pull = input(f"Model '{selected}' is not installed. Pull it now? (y/n): ").strip().lower()
        if pull in {"y", "yes"}:
            ensure_model(selected)
        else:
            logger.error("❌ Model not installed. Exiting.")
            sys.exit(1)

    return selected


def _prompt_file_path() -> str:
    return os.path.join(os.path.dirname(__file__) or ".", "SYSTEM_PROMPT.txt")


def _load_system_prompt() -> str:
    """Load system prompt from env or SYSTEM_PROMPT.txt.

    This script relies on SYSTEM_PROMPT.txt (or $SYSTEM_PROMPT) and will fail
    fast if neither is provided.
    """
    env_prompt = os.environ.get("SYSTEM_PROMPT")
    if env_prompt and env_prompt.strip():
        return env_prompt.strip()

    prompt_file = _prompt_file_path()
    if not os.path.exists(prompt_file):
        logger.error(f"❌ Missing required file: {prompt_file}")
        logger.info("   Create SYSTEM_PROMPT.txt (or set $SYSTEM_PROMPT) and re-run.")
        sys.exit(1)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError as e:
        logger.error(f"❌ Failed to read {prompt_file}: {e}")
        sys.exit(1)

    if not content:
        logger.error(f"❌ {prompt_file} is empty. Add a system prompt and re-run.")
        sys.exit(1)

    return content


def _initial_questions_file_path() -> str:
    return os.path.join(os.path.dirname(__file__) or ".", "INITIAL_QUESTIONS.txt")


def _load_initial_questions() -> list[str]:
    """Load initial questions from INITIAL_QUESTIONS.txt.

    - One question per line.
    - Empty lines and lines starting with # are ignored.

    This script relies on INITIAL_QUESTIONS.txt and will fail fast if the file
    is missing or contains no questions.
    """
    path = _initial_questions_file_path()
    if not os.path.exists(path):
        logger.error(f"❌ Missing required file: {path}")
        logger.info("   Create INITIAL_QUESTIONS.txt (one question per line) and re-run.")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            questions: list[str] = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                questions.append(line)
    except OSError as e:
        logger.error(f"❌ Failed to read {path}: {e}")
        sys.exit(1)

    if not questions:
        logger.error(f"❌ {path} contains no questions.")
        logger.info("   Add at least one question (non-empty line) and re-run.")
        sys.exit(1)

    return questions


def _default_session_log_path(model_name: str) -> str:
    logs_dir = os.path.join(os.path.dirname(__file__) or ".", "qa_logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _sanitize_for_filename(model_name)
    return os.path.join(logs_dir, f"qa_{model_slug}_{ts}.jsonl")


def _session_log_path(model_name: str, qa_log_path: Optional[str]) -> str:
    """Return the output JSONL path.

    If qa_log_path is:
    - None: use default qa_logs/qa_<model>_<ts>.jsonl
    - a directory: create file qa_<model>_<ts>.jsonl inside it
    - a file path: use it directly (create parent directories)
    """
    if not qa_log_path:
        return _default_session_log_path(model_name)

    expanded = os.path.expanduser(qa_log_path)
    expanded = os.path.abspath(expanded)

    # If it exists and is a directory, or if it ends with path separator, treat as directory.
    treat_as_dir = os.path.isdir(expanded) or qa_log_path.endswith(os.sep)

    if treat_as_dir:
        os.makedirs(expanded, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = _sanitize_for_filename(model_name)
        return os.path.join(expanded, f"qa_{model_slug}_{ts}.jsonl")

    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)

    return expanded


def run_analyzer(runtime_args=None):
    from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama

    if runtime_args is None:
        runtime_args = argparse.Namespace(model=None, chunk_size=None, similarity_top_k=None, log_dir="logs")

    cli_full = _all_cli_runtime_params_provided(runtime_args)

    # Resolve model + chunking + retrieval settings.
    model_name, chunk_size, similarity_top_k = _resolve_runtime_settings(runtime_args)

    # Allow quick prompt edits now that the model is chosen.
    if not cli_full:
        _maybe_edit_system_prompt_interactively()
    system_prompt = _load_system_prompt()

    log_dir = getattr(runtime_args, "log_dir", "logs") or "logs"
    log_dir = os.path.abspath(os.path.expanduser(log_dir))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        logger.error(f"❌ Directory not found previously; created it for you: {log_dir}")
        logger.info("   Add at least one log file there, then re-run.")
        sys.exit(1)

    files = [
        f
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f)) and not f.startswith(".")
    ]
    if not files:
        logger.error(f"❌ No files found in {log_dir}")
        logger.info("   Place the log file(s) you want to analyze in that directory and re-run.")
        sys.exit(1)

    files_sorted = sorted(files)

    if cli_full:
        # Per requirement: no prompts if all 3 runtime params were provided.
        chosen_file = files_sorted[0]
        run_initial = True
    else:
        chosen_file = _select_log_file_interactively(files_sorted, default_index=1)
        run_initial = _prompt_yes_no("Run initial queries?", default_yes=True)

    file_path = os.path.join(log_dir, chosen_file)
    logger.info(f"✅ Using log file: {file_path}")

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    llm = Ollama(
        model=model_name,
        request_timeout=600.0,
        temperature=0.1,
        system_prompt=system_prompt,
    )
    embed = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = llm
    Settings.embed_model = embed

    splitter = SentenceSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        paragraph_separator="\n\n",
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(
        f"📑 Loaded {len(documents)} document(s), split into {len(nodes)} chunks. "
        f"chunk_size = {chunk_size}, similarity_top_k = {similarity_top_k}"
    )

    index = VectorStoreIndex(nodes)
    qe = index.as_query_engine(similarity_top_k=similarity_top_k)

    session_log_path = _session_log_path(model_name, getattr(runtime_args, "qa_log_path", None))
    _append_jsonl(
        session_log_path,
        {
            "type": "session_start",
            "time": _utc_now_iso(),
            "model": model_name,
            "log_file": file_path,
            "chunk_size": chunk_size,
            "similarity_top_k": similarity_top_k,
            "system_prompt_source": "env"
            if os.environ.get("SYSTEM_PROMPT")
            else ("file" if os.path.exists(_prompt_file_path()) else "default"),
        },
    )
    logger.info(f"📝 Q&A will be saved to: {session_log_path}")

    def ask_and_log(question: str, source: str) -> str:
        start = time.perf_counter()
        reply_obj = qe.query(question)
        duration = time.perf_counter() - start
        reply_text = str(reply_obj)

        # Per requirement: each Q/A line is exactly {qery, reply, time}
        _append_jsonl(
            session_log_path,
            {
                "qery": question,
                "reply": reply_text,
                "time": duration,
            },
        )
        logger.info("🤖 Reply: %s", reply_text)
        return reply_text

    if run_initial:
        initial_questions = _load_initial_questions()
        for q in initial_questions:
            print("=" * 80)
            print("Q:", q)
            print(ask_and_log(q, source="initial"))

    print("\n💬 Enter your own questions about this log (type 'exit' to quit):")
    while True:
        try:
            user_q = input("> ")
        except EOFError:
            _append_jsonl(session_log_path, {"type": "session_end", "time": _utc_now_iso()})
            print("\n👋 Bye!")
            break

        user_q = user_q.strip()
        if user_q.lower() in {"exit", "quit"}:
            _append_jsonl(
                session_log_path,
                {"type": "session_end", "time": _utc_now_iso()},
            )
            print("👋 Bye!")
            break
        if not user_q:
            continue
        print("=" * 80)
        print("Q:", user_q)
        print(ask_and_log(user_q, source="interactive"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze log files with an Ollama-backed LLM.")
    parser.add_argument("--model", help="Ollama model name (e.g., codellama:13b-instruct)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for splitting logs (larger = fewer chunks)",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        help="Number of top similar chunks to retrieve per query",
    )
    parser.add_argument(
        "--qa-log-path",
        help="Where to save the Q&A JSONL log (directory or full file path)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing the log files to analyze (default: logs)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        args = _parse_args(sys.argv[1:])
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        ensure_venv()
        _restart_inside_venv_if_needed()
        ensure_ollama()
        ensure_ollama_started()
        run_analyzer(args)
    except BrokenPipeError:
        # Clean exit when piping output (e.g., through `head`).
        sys.exit(0)
