"""
MANA External Bridge v0.2
=========================

Read-only integration layer for inspecting a local MANA project.

v0.2 adds:
    - project overview
    - module map
    - runtime-state map
    - dependency map
    - code-evolution targets
    - evolution status
    - Git status/history
    - compact AI context package
    - structured JSON output

IMPORTANT:
    v0.2 is READ ONLY.
    It does NOT modify, create, delete or rename MANA files.
    It does NOT apply patches.
    It does NOT execute arbitrary shell commands.

Usage:
    python mana_bridge.py

Interactive commands:
    ping
    project
    modules
    runtime
    dependencies
    code-targets
    evolution
    git
    context
    manifest
    tree
    search <text>
    read <relative_path>
    info <relative_path>
    help
    exit
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "0.2.0"
BRIDGE_NAME = "MANA External Bridge"
MODE = "READ_ONLY"

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "mana"

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "tmp",
    "temp",
    "cache",
    "caches",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".exe",
    ".bin",
    ".model",
    ".gguf",
    ".safetensors",
    ".onnx",
    ".pt",
    ".pth",
    ".zip",
    ".7z",
    ".rar",
    ".iso",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyw",
    ".ps1",
    ".bat",
    ".cmd",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".txt",
    ".md",
    ".rst",
    ".xml",
    ".html",
    ".css",
    ".sql",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".cs",
    ".go",
    ".rs",
}

RUNTIME_PATHS = {
    "memory_db": "mana_memory/mana_memory.sqlite3",
    "knowledge_db": "mana_v3_4_knowledge.pkl",
    "state_file": "mana_v3_4_state.pkl",
    "experience_db": "mana_v3_4_experience.sqlite3",
    "history_file": "mana_v3_4_history.json",
    "cache_file": "mana_v3_4_cache.pkl",
    "evolution_reports": "mana_v4_6_evolution_reports.json",
    "knowledge_root": "mana_memory/knowledge",
    "code_history": "mana_code_history",
}

CONFIG_PATH = PACKAGE_ROOT / "config.py"
CODE_EVOLUTION_PATH = PACKAGE_ROOT / "code_evolution.py"
VERSION_PATH = PACKAGE_ROOT / "version.py"

MAX_SEARCH_FILE_SIZE = 5 * 1024 * 1024
MAX_READ_FILE_SIZE = 10 * 1024 * 1024
MAX_TREE_ITEMS = 1000
MAX_SEARCH_RESULTS = 100


# ============================================================
# BASIC HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return True

    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def is_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return True

    if suffix == "":
        try:
            return path.stat().st_size <= MAX_READ_FILE_SIZE
        except OSError:
            return False

    return False


def iter_project_files() -> Iterable[Path]:
    for root, dirs, files in os.walk(PROJECT_ROOT):

        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS
        ]

        root_path = Path(root)

        for name in files:
            path = root_path / name

            if is_excluded(path):
                continue

            yield path


def safe_read_text(path: Path) -> str:
    errors: List[Exception] = []

    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(
                encoding=encoding,
                errors="strict",
            )
        except Exception as exc:
            errors.append(exc)

    raise RuntimeError(
        f"Не удалось прочитать файл: {errors[-1]}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


# ============================================================
# GIT
# ============================================================

def git_command(*args: str) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )

        return (
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )

    except FileNotFoundError:
        return -1, "", "Git не найден в PATH."

    except Exception as exc:
        return -1, "", str(exc)


def get_git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "available": False,
        "branch": None,
        "commit": None,
        "remote": None,
        "clean": None,
        "status": [],
    }

    code, branch, error = git_command(
        "branch",
        "--show-current",
    )

    if code != 0:
        info["error"] = error
        return info

    info["available"] = True
    info["branch"] = branch or None

    code, commit, _ = git_command(
        "rev-parse",
        "--short",
        "HEAD",
    )

    if code == 0:
        info["commit"] = commit

    code, remote, _ = git_command(
        "remote",
        "get-url",
        "origin",
    )

    if code == 0:
        info["remote"] = remote

    code, status, _ = git_command(
        "status",
        "--short",
    )

    if code == 0:
        lines = status.splitlines() if status else []
        info["status"] = lines
        info["clean"] = not bool(lines)

    return info


def get_git_history(limit: int = 10) -> List[Dict[str, str]]:
    code, output, error = git_command(
        "log",
        f"-{limit}",
        "--pretty=format:%h|%ad|%s",
        "--date=iso",
    )

    if code != 0:
        return [{
            "error": error,
        }]

    result = []

    for line in output.splitlines():

        parts = line.split("|", 2)

        if len(parts) != 3:
            continue

        result.append({
            "commit": parts[0],
            "date": parts[1],
            "message": parts[2],
        })

    return result


# ============================================================
# PROJECT
# ============================================================

def get_project_stats() -> Dict[str, Any]:

    total = 0
    python_files = 0
    text_files = 0
    directories = set()
    total_size = 0

    for path in iter_project_files():

        total += 1

        try:
            total_size += path.stat().st_size
        except OSError:
            pass

        if path.suffix.lower() == ".py":
            python_files += 1

        if is_text_file(path):
            text_files += 1

        try:
            directories.add(
                relative_path(path.parent)
            )
        except Exception:
            pass

    return {
        "root": str(PROJECT_ROOT),
        "files": total,
        "python_files": python_files,
        "text_files": text_files,
        "directories": len(directories),
        "total_size_bytes": total_size,
        "generated_at": now_iso(),
    }


def command_project():
    data = {
        "bridge": {
            "name": BRIDGE_NAME,
            "version": VERSION,
            "mode": MODE,
        },
        "project": get_project_stats(),
        "git": get_git_info(),
    }

    print(safe_json(data))


# ============================================================
# MODULE MAP
# ============================================================

def extract_python_module_info(path: Path) -> Optional[Dict[str, Any]]:

    if path.suffix.lower() != ".py":
        return None

    try:
        source = safe_read_text(path)

        if len(source) > MAX_READ_FILE_SIZE:
            return None

        tree = ast.parse(source)

    except Exception:
        return None

    imports: List[str] = []
    functions: List[str] = []
    classes: List[str] = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):

            for item in node.names:
                imports.append(item.name)

        elif isinstance(node, ast.ImportFrom):

            module = node.module or ""

            if node.level:
                module = "." * node.level + module

            imports.append(module)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            functions.append(node.name)

        elif isinstance(node, ast.ClassDef):

            classes.append(node.name)

    return {
        "path": relative_path(path),
        "size": path.stat().st_size,
        "imports": sorted(set(imports)),
        "functions": sorted(set(functions)),
        "classes": sorted(set(classes)),
    }


def get_module_map() -> List[Dict[str, Any]]:

    modules = []

    if not PACKAGE_ROOT.exists():
        return modules

    for path in sorted(
        PACKAGE_ROOT.rglob("*.py")
    ):

        if is_excluded(path):
            continue

        item = extract_python_module_info(path)

        if item:
            modules.append(item)

    return modules


def command_modules():

    modules = get_module_map()

    output = {
        "package": relative_path(PACKAGE_ROOT),
        "module_count": len(modules),
        "modules": modules,
    }

    print(safe_json(output))


# ============================================================
# RUNTIME STATE
# ============================================================

def runtime_item(
    name: str,
    relative: str,
    kind: str,
) -> Dict[str, Any]:

    path = PROJECT_ROOT / relative

    exists = path.exists()

    item: Dict[str, Any] = {
        "name": name,
        "path": relative,
        "kind": kind,
        "exists": exists,
    }

    if exists:

        try:
            item["size"] = path.stat().st_size
            item["modified"] = datetime.fromtimestamp(
                path.stat().st_mtime
            ).astimezone().isoformat()
        except OSError:
            pass

    return item


def get_runtime_map() -> Dict[str, Any]:

    result = {}

    for name, relative in RUNTIME_PATHS.items():

        if name == "knowledge_root":
            result[name] = runtime_item(
                name,
                relative,
                "directory",
            )

        elif name == "code_history":
            result[name] = runtime_item(
                name,
                relative,
                "directory",
            )

        else:
            kind = "sqlite" if relative.endswith(
                (".sqlite3", ".sqlite", ".db")
            ) else "persistent_state"

            result[name] = runtime_item(
                name,
                relative,
                kind,
            )

    return result


def command_runtime():
    print(safe_json({
        "runtime_state": get_runtime_map(),
        "principle": (
            "Runtime state is separate from source code. "
            "This Bridge only inspects it."
        ),
    }))


# ============================================================
# CONFIG / DEPENDENCIES
# ============================================================

def extract_config_paths() -> Dict[str, str]:

    if not CONFIG_PATH.exists():
        return {}

    try:
        source = safe_read_text(CONFIG_PATH)
        tree = ast.parse(source)
    except Exception:
        return {}

    result: Dict[str, str] = {}

    for node in ast.walk(tree):

        if not isinstance(node, ast.AnnAssign):
            continue

        target = node.target

        if not isinstance(target, ast.Name):
            continue

        name = target.id

        if not (
            name.endswith("_path")
            or name.endswith("_file")
            or name.endswith("_root")
        ):
            continue

        value = node.value

        if isinstance(value, ast.Constant) and isinstance(
            value.value,
            str,
        ):
            result[name] = value.value

    return result


def get_python_import_dependencies() -> Dict[str, List[str]]:

    result: Dict[str, List[str]] = {}

    for item in get_module_map():

        result[item["path"]] = item["imports"]

    return result


def command_dependencies():

    print(safe_json({
        "config_paths": extract_config_paths(),
        "python_imports": get_python_import_dependencies(),
    }))


# ============================================================
# CODE EVOLUTION TARGETS
# ============================================================

def parse_code_evolution_targets() -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "source": relative_path(CODE_EVOLUTION_PATH),
        "loaded": False,
        "whitelist": [],
        "never_patchable": [],
        "targets": [],
    }

    if not CODE_EVOLUTION_PATH.exists():
        result["error"] = "code_evolution.py not found"
        return result

    try:
        source = safe_read_text(CODE_EVOLUTION_PATH)
        tree = ast.parse(source)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["loaded"] = True

    for node in tree.body:

        # Read _NEVER_PATCHABLE.
        if isinstance(node, ast.Assign):

            for target in node.targets:

                if not isinstance(target, ast.Name):
                    continue

                if target.id == "_NEVER_PATCHABLE":

                    try:
                        value = ast.literal_eval(node.value)

                        if isinstance(value, (set, list, tuple)):
                            result["never_patchable"] = sorted(
                                str(x) for x in value
                            )

                    except Exception:
                        pass

        # Read WHITELIST dictionary.
        if isinstance(node, ast.Assign):

            for target in node.targets:

                if not isinstance(target, ast.Name):
                    continue

                if target.id == "WHITELIST":

                    try:
                        value = ast.literal_eval(node.value)

                        if isinstance(value, dict):
                            result["whitelist"] = sorted(
                                str(x) for x in value.keys()
                            )

                    except Exception:
                        pass

    # Extract dataclass target metadata and useful function names.
    for node in tree.body:

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if node.name in {
            "apply_patch",
            "rollback",
            "history",
            "decide",
            "evaluate_candidate",
        }:

            args = []

            for arg in node.args.args:
                args.append(arg.arg)

            result["targets"].append({
                "kind": "code_evolution_function",
                "name": node.name,
                "parameters": args,
            })

    return result


def command_code_targets():

    result = parse_code_evolution_targets()

    result["note"] = (
        "Bridge does not decide what is patchable. "
        "MANA code_evolution remains authoritative."
    )

    print(safe_json(result))


# ============================================================
# EVOLUTION STATUS
# ============================================================

def command_evolution():

    runtime = get_runtime_map()

    reports_path = PROJECT_ROOT / RUNTIME_PATHS["evolution_reports"]

    report_info: Dict[str, Any] = {
        "path": relative_path(reports_path),
        "exists": reports_path.exists(),
    }

    if reports_path.exists():

        try:
            report_info["size"] = reports_path.stat().st_size

            data = json.loads(
                safe_read_text(reports_path)
            )

            if isinstance(data, list):

                report_info["count"] = len(data)

                if data:
                    report_info["latest"] = data[-1]

        except Exception as exc:
            report_info["read_error"] = str(exc)

    state_path = PROJECT_ROOT / RUNTIME_PATHS["state_file"]

    state_info: Dict[str, Any] = {
        "path": relative_path(state_path),
        "exists": state_path.exists(),
    }

    # Deliberately DO NOT unpickle arbitrary state in Bridge.
    # We only report presence/size to keep v0.2 strictly observational.
    if state_path.exists():

        try:
            state_info["size"] = state_path.stat().st_size
        except OSError:
            pass

    print(safe_json({
        "runtime": runtime,
        "state": state_info,
        "evolution_reports": report_info,
        "code_targets": parse_code_evolution_targets(),
    }))


# ============================================================
# COMPACT AI CONTEXT
# ============================================================

def build_ai_context() -> Dict[str, Any]:

    project = get_project_stats()
    git = get_git_info()
    modules = get_module_map()

    module_summary = []

    for item in modules:

        module_summary.append({
            "path": item["path"],
            "classes": item["classes"],
            "functions": item["functions"],
        })

    context = {
        "bridge": {
            "name": BRIDGE_NAME,
            "version": VERSION,
            "mode": MODE,
            "generated_at": now_iso(),
        },

        "project": project,

        "git": {
            "branch": git.get("branch"),
            "commit": git.get("commit"),
            "remote": git.get("remote"),
            "clean": git.get("clean"),
        },

        # Hand-maintained: this is the map an external reader trusts to
        # know what MANA *is*, so a subsystem missing here is worse than a
        # subsystem missing from a listing -- the reader never learns it
        # exists. It went stale once already (5.10 added the brain pool and
        # this still described a single-LLM agent), so anything added under
        # mana/ that is a subsystem rather than a helper belongs here.
        "architecture": {
            "package": "mana/",
            "entrypoint": "mana_run.py",
            "config": "mana/config.py",
            "agent": "mana/agent.py",
            "memory": "mana/memory.py",
            "graph_memory": "mana/graph_memory.py",
            "knowledge": "mana/knowledge.py",
            "experience": "mana/experience.py",
            "evolution": "mana/agent_parts/evolution.py",
            "code_evolution": "mana/code_evolution.py",
            "verification": "mana/verifier.py",
            "llm": "mana/llm.py",
            "brains": "mana/brains.py",
            "decompose": "mana/decompose.py",
            "tools": "mana/tools.py",
            "pipeline": "mana/pipeline.py",
            "tests": "tests/",
        },

        "runtime_state": get_runtime_map(),

        "code_evolution": parse_code_evolution_targets(),

        "modules": module_summary,

        "integration_contract": {
            "read": True,
            "search": True,
            "analyze": True,
            "patch": False,
            "execute_arbitrary": False,
            "git_write": False,
        },
    }

    return context


def command_context():

    print(safe_json(build_ai_context()))


# ============================================================
# MANIFEST
# ============================================================

def build_manifest() -> Dict[str, Any]:

    files = []

    for path in sorted(
        iter_project_files(),
        key=lambda x: relative_path(x).lower(),
    ):

        try:
            stat = path.stat()
        except OSError:
            continue

        files.append({
            "path": relative_path(path),
            "size": stat.st_size,
            "extension": path.suffix.lower(),
        })

    return {
        "bridge": {
            "name": BRIDGE_NAME,
            "version": VERSION,
            "mode": MODE,
        },
        "project": get_project_stats(),
        "git": get_git_info(),
        "runtime": get_runtime_map(),
        "files": files,
    }


def command_manifest():
    print(safe_json(build_manifest()))


# ============================================================
# TREE
# ============================================================

def command_tree():

    items = sorted(
        iter_project_files(),
        key=lambda x: relative_path(x).lower(),
    )

    print()
    print("MANA PROJECT TREE")
    print("=" * 70)

    for index, path in enumerate(items[:MAX_TREE_ITEMS], 1):

        rel = relative_path(path)

        depth = len(Path(rel).parts) - 1

        prefix = "  " * depth

        print(f"{prefix}{Path(rel).name}")

    if len(items) > MAX_TREE_ITEMS:
        print()
        print(
            f"... вывод ограничен {MAX_TREE_ITEMS} файлами"
        )

    print("=" * 70)
    print(f"Показано: {min(len(items), MAX_TREE_ITEMS)}")


# ============================================================
# SEARCH
# ============================================================

def command_search(query: str):

    if not query:
        print("Использование: search <text>")
        return

    q = query.lower()

    results = []

    for path in iter_project_files():

        if not is_text_file(path):
            continue

        try:
            size = path.stat().st_size

            if size > MAX_SEARCH_FILE_SIZE:
                continue

            text = safe_read_text(path)

        except Exception:
            continue

        for number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            if q in line.lower():

                results.append({
                    "file": relative_path(path),
                    "line": number,
                    "text": line.strip(),
                })

                if len(results) >= MAX_SEARCH_RESULTS:
                    break

        if len(results) >= MAX_SEARCH_RESULTS:
            break

    print()
    print("SEARCH RESULTS")
    print("=" * 70)

    if not results:

        print("Совпадений не найдено.")
        return

    for item in results:

        print(
            f"{item['file']}:{item['line']}: "
            f"{item['text']}"
        )

    print()
    print(f"Результатов: {len(results)}")


# ============================================================
# READ FILE
# ============================================================

def resolve_project_file(user_path: str) -> Path:

    requested = Path(
        user_path.strip().strip('"')
    )

    if requested.is_absolute():
        candidate = requested.resolve()
    else:
        candidate = (
            PROJECT_ROOT / requested
        ).resolve()

    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(
            "Доступ за пределы E:\\Mana запрещён."
        )

    return candidate


def command_read(user_path: str):

    if not user_path:
        print("Использование: read <relative_path>")
        return

    try:
        path = resolve_project_file(user_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    if not path.exists():
        print("ERROR: файл не найден.")
        return

    if not path.is_file():
        print("ERROR: это не файл.")
        return

    try:
        size = path.stat().st_size
    except OSError as exc:
        print(f"ERROR: {exc}")
        return

    if size > MAX_READ_FILE_SIZE:
        print(
            f"ERROR: файл слишком большой ({size} bytes)."
        )
        return

    if not is_text_file(path):
        print(
            "ERROR: файл не определён как текстовый."
        )
        return

    try:
        text = safe_read_text(path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    print()
    print(f"FILE: {relative_path(path)}")
    print(f"SIZE: {size}")
    print(f"SHA256: {sha256_file(path)}")
    print("-" * 70)
    print(text)


# ============================================================
# INFO
# ============================================================

def command_info(user_path: str):

    if not user_path:
        print("Использование: info <relative_path>")
        return

    try:
        path = resolve_project_file(user_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    if not path.exists():
        print("ERROR: файл не найден.")
        return

    try:

        stat = path.stat()

        data = {
            "path": relative_path(path),
            "absolute_path": str(path),
            "size": stat.st_size,
            "extension": path.suffix.lower(),
            "is_text": is_text_file(path),
            "sha256": sha256_file(path),
            "modified": datetime.fromtimestamp(
                stat.st_mtime
            ).astimezone().isoformat(),
        }

        print(safe_json(data))

    except Exception as exc:
        print(f"ERROR: {exc}")


# ============================================================
# PING / HELP
# ============================================================

def command_ping():

    print(safe_json({
        "bridge": BRIDGE_NAME,
        "version": VERSION,
        "status": "ONLINE",
        "mode": MODE,
        "project_root": str(PROJECT_ROOT),
        "timestamp": now_iso(),
    }))


def command_git():

    print(safe_json({
        "current": get_git_info(),
        "history": get_git_history(10),
    }))


def command_help():

    print()
    print(f"{BRIDGE_NAME} v{VERSION}")
    print("=" * 70)
    print()
    print("Основные команды:")
    print()
    print("  ping")
    print("      Состояние Bridge.")
    print()
    print("  project")
    print("      Общая информация о проекте.")
    print()
    print("  modules")
    print("      Карта Python-модулей, классов, функций и импортов.")
    print()
    print("  runtime")
    print("      Состояние persistent/runtime-файлов.")
    print()
    print("  dependencies")
    print("      Пути Config и импорты Python-модулей.")
    print()
    print("  code-targets")
    print("      Разрешённые механизмы code_evolution MANA.")
    print()
    print("  evolution")
    print("      Состояние эволюции и evolution reports.")
    print()
    print("  git")
    print("      Branch, commit, remote, status и история.")
    print()
    print("  context")
    print("      Компактный структурированный контекст для внешнего ИИ.")
    print()
    print("  manifest")
    print("      Полный список видимых файлов проекта.")
    print()
    print("  tree")
    print("      Дерево проекта.")
    print()
    print("  search <text>")
    print("      Поиск по текстовым исходникам.")
    print()
    print("  read <file>")
    print("      Чтение текстового файла.")
    print()
    print("  info <file>")
    print("      Размер, SHA256 и метаданные файла.")
    print()
    print("  help")
    print("      Эта справка.")
    print()
    print("  exit")
    print("      Выход.")
    print()
    print("Режим v0.2: READ_ONLY")


# ============================================================
# COMMAND DISPATCH
# ============================================================

def execute_command(command: str) -> bool:

    command = command.strip()

    if not command:
        return True

    parts = command.split(maxsplit=1)

    name = parts[0].lower()
    argument = parts[1] if len(parts) > 1 else ""

    try:

        if name == "ping":
            command_ping()

        elif name == "project":
            command_project()

        elif name == "modules":
            command_modules()

        elif name == "runtime":
            command_runtime()

        elif name == "dependencies":
            command_dependencies()

        elif name == "code-targets":
            command_code_targets()

        elif name == "evolution":
            command_evolution()

        elif name == "git":
            command_git()

        elif name == "context":
            command_context()

        elif name == "manifest":
            command_manifest()

        elif name == "tree":
            command_tree()

        elif name == "search":
            command_search(argument)

        elif name == "read":
            command_read(argument)

        elif name == "info":
            command_info(argument)

        elif name == "help":
            command_help()

        elif name in {"exit", "quit"}:
            return False

        else:
            print(f"Неизвестная команда: {name}")
            print("Введите help.")

    except Exception as exc:

        print(
            f"INTERNAL ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(f"{BRIDGE_NAME} v{VERSION}")
    print("=" * 70)
    print(f"PROJECT: {PROJECT_ROOT}")
    print(f"MODE:    {MODE}")
    print()
    print("Введите 'help' для списка команд.")
    print()

    while True:

        try:
            command = input("MANA-BRIDGE> ")

        except KeyboardInterrupt:
            print()
            break

        except EOFError:
            print()
            break

        if not execute_command(command):
            break

    print()
    print("MANA External Bridge stopped.")


if __name__ == "__main__":
    main()

