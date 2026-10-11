"""Snapshot actual declarations for reviewing the documentation's API coverage.

This uses AST only: it does not import NumPy, load models or execute library code.
Regenerate with `python scripts/build_api_inventory.py` after an API change.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/glassllm"
OUTPUT = ROOT / "site/src/data/api-inventory.json"


def visible(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__") and name not in {"__repr__", "__post_init__"})


def function(node: ast.FunctionDef) -> dict:
    decorators = {ast.unparse(item) for item in node.decorator_list}
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return {"name": node.name, "kind": "property" if "property" in decorators else "function",
            "signature": f"{node.name}({ast.unparse(node.args)}){returns}", "line": node.lineno}


def members(node: ast.ClassDef) -> list[dict]:
    result = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and visible(item.name):
            result.append(function(item))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and visible(item.target.id):
            result.append({"name": item.target.id, "kind": "field", "signature": f"{item.target.id}: {ast.unparse(item.annotation)}", "line": item.lineno})
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and visible(target.id):
                    result.append({"name": target.id, "kind": "alias", "signature": f"{target.id} = {ast.unparse(item.value)}", "line": item.lineno})
    return result


def main() -> None:
    modules = []
    root_exports = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        symbols = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                symbols.append(function(node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                symbols.append({"name": node.name, "kind": "class", "signature": node.name,
                                "line": node.lineno, "members": members(node)})
            elif path.name == "__init__.py" and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        root_exports = ast.literal_eval(node.value)
        modules.append({"module": "glassllm" if path.stem == "__init__" else f"glassllm.{path.stem}",
                        "file": path.relative_to(ROOT).as_posix(), "symbols": symbols})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"rootExports": root_exports, "modules": modules}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = sum(len(module["symbols"]) for module in modules)
    print(f"API inventory: {len(root_exports)} root exports; {count} declarations in {len(modules)} modules")


if __name__ == "__main__":
    main()
