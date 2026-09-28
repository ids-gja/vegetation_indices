import ast
from pathlib import Path


def test_source_packages_use_only_absolute_imports():
    source = Path(__file__).resolve().parents[1] / "src"
    relative = [
        str(path.relative_to(source))
        for path in source.rglob("*.py")
        if any(
            isinstance(node, ast.ImportFrom) and node.level
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    ]
    assert relative == []
