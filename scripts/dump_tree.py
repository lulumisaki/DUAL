#!/usr/bin/env python3
import os
import sys
import re
from pathlib import Path
from typing import Iterable

IGNORE_PATTERNS = [
    r"\.git($|/)", r"\.idea($|/)", r"__pycache__($|/)", r"\.ipynb_checkpoints($|/)",
    r"\.DS_Store$", r"venv($|/)", r"\.venv($|/)", r"env($|/)",
    r"models($|/)", r"checkpoints($|/)", r"outputs($|/)", r"logs($|/)",
    r"data($|/)", r"datasets($|/)",
    r".*\\.bin$", r".*\\.pt$", r".*\\.safetensors$", r".*\\.onnx$", r".*\\.gguf$",
    r".*\\.ckpt$", r".*\\.h5$", r".*\\.npz$",
]

def should_ignore(path: str, patterns: Iterable[str]) -> bool:
    p = path.replace("\\", "/") + ("/" if os.path.isdir(path) else "")
    return any(re.search(pat, p) for pat in patterns)


def build_tree(root: Path) -> list[str]:
    lines = []

    def walk(dir_path: Path, prefix: str = ""):
        try:
            entries = sorted(
                dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            rel = entry.relative_to(root)
            if should_ignore(str(rel), IGNORE_PATTERNS):
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    lines.append(root.name)
    walk(root)
    return lines


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "repo_tree.md"
    lines = build_tree(root)
    md = []
    md.append("# Repository Tree (filtered)\n")
    md.append("```\n")
    md.extend(line + "\n" for line in lines)
    md.append("```\n")
    out.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()