#!/usr/bin/env python3
"""
CI Import Boundary Checker.
Enforces architectural boundaries specified in LLD v2.0 Section 2.2:
- Mode Handlers must NEVER import langgraph or langgraph_workflows
- Tools must NEVER import context, grpc/clients, or prompts
- Context must NEVER import tools
- Prompts must NEVER import tools
- gRPC clients must NEVER import workflow_engine or request_analyzer
"""

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent / "app"

RULES = [
    {
        "source_dir": ROOT / "workflow_engine" / "mode_handlers",
        "forbidden_imports": ["langgraph", "app.workflow_engine.langgraph_workflows"],
        "reason": "Mode Handlers must have zero LangGraph dependency",
    },
    {
        "source_dir": ROOT / "tools",
        "forbidden_imports": ["app.context", "app.grpc.clients", "app.prompts"],
        "reason": "Tools must not wrap or duplicate baseline context or prompts",
    },
    {
        "source_dir": ROOT / "context",
        "forbidden_imports": ["app.tools"],
        "reason": "Baseline context collector must not depend on optional tools",
    },
    {
        "source_dir": ROOT / "prompts",
        "forbidden_imports": ["app.tools"],
        "reason": "Prompt builder reads state/data, not tool modules",
    },
    {
        "source_dir": ROOT / "grpc" / "clients",
        "forbidden_imports": ["app.workflow_engine", "app.request_analyzer"],
        "reason": "gRPC clients are low-level wrappers and must not import orchestration",
    },
]


def check_file(file_path: Path, forbidden: list[str]) -> list[str]:
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception as e:
        return [f"Failed to parse {file_path}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for f_imp in forbidden:
                    if alias.name == f_imp or alias.name.startswith(f_imp + "."):
                        violations.append(f"Line {node.lineno}: 'import {alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for f_imp in forbidden:
                if module == f_imp or module.startswith(f_imp + "."):
                    violations.append(f"Line {node.lineno}: 'from {module} import ...'")
    return violations


def main() -> int:
    if not ROOT.exists():
        print(f"Directory {ROOT} does not exist.")
        return 0

    all_violations = []
    for rule in RULES:
        source_dir = rule["source_dir"]
        forbidden = rule["forbidden_imports"]
        reason = rule["reason"]

        if not source_dir.exists():
            continue

        for py_file in source_dir.rglob("*.py"):
            violations = check_file(py_file, forbidden)
            if violations:
                for v in violations:
                    all_violations.append(
                        f"VIOLATION in {py_file.relative_to(ROOT.parent)}:\n  {v}\n  Reason: {reason}"
                    )

    if all_violations:
        print("\n".join(all_violations), file=sys.stderr)
        print(f"\nTotal import boundary violations: {len(all_violations)}", file=sys.stderr)
        return 1

    print("All import boundary checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
