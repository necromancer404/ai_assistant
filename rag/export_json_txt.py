import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Tuple


def _iter_leaf_paths(node: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    # Yield key-paths to scalar values
    if isinstance(node, dict):
        for k, v in node.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_leaf_paths(v, new_prefix)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_prefix = f"{prefix}[{i}]"
            yield from _iter_leaf_paths(v, new_prefix)
    else:
        yield prefix, node


def _to_scalar_text(value: Any, max_len: int = 2000) -> str:
    if value is None or isinstance(value, (int, float, bool, str)):
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False)
    if len(text) > max_len:
        return text[:max_len] + " ...[truncated]"
    return text


def export_json_to_txt(root: Path) -> int:
    count = 0
    for path in root.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            is_valid = True
        except Exception:
            is_valid = False

        # 1) Pretty TXT (raw JSON pretty-printed or as-is if invalid)
        if is_valid:
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                pretty = f.read()
        txt_path = path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as out:
            out.write(pretty)

        # 2) Flattened TXT (key paths -> scalar values), helps retrieval
        flat_lines = []
        if is_valid:
            for key_path, value in _iter_leaf_paths(data):
                # Only include scalars or short reprs
                if isinstance(value, (str, int, float, bool)) or value is None:
                    val_text = _to_scalar_text(value)
                    flat_lines.append(f"{key_path}: {val_text}")
                else:
                    # Non-scalar (nested object/list) summarized to JSON string (capped)
                    val_text = _to_scalar_text(value)
                    flat_lines.append(f"{key_path}: {val_text}")
        else:
            flat_lines.append("(invalid JSON - raw content mirrored in .txt)")

        flat_txt_path = path.with_suffix(".flatten.txt")
        with open(flat_txt_path, "w", encoding="utf-8") as out:
            out.write("\n".join(flat_lines))

        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Export JSON files to pretty-indented TXT copies.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing JSON files")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    if not root.exists():
        print(f"Data dir not found: {root}")
        return
    n = export_json_to_txt(root)
    print(f"Exported {n} JSON files to .txt in {root}")


if __name__ == "__main__":
    main()


