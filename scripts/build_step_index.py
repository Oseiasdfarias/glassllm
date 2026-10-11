"""Write src/glassllm/step_index.json from source comment markers."""
import json

from glassllm.steps import INDEX_PATH, scan_markers


def main() -> None:
    index = scan_markers()
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(index)} steps -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
