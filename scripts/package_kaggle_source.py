"""Package source only for the Task 3 Kaggle GPU notebook; never include data."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "outputs" / "task3_kaggle_source.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "tasks", "scripts", "configs"):
            for path in sorted((ROOT / folder).rglob("*")):
                if (path.is_file() and path.suffix in {".py", ".txt", ".yaml", ".md"}
                        and "__pycache__" not in path.parts):
                    archive.write(path, path.relative_to(ROOT))
        guide = ROOT / "docs/project/TASK3_T4X2.md"
        archive.write(guide, "TASK3_T4X2.md")
    print(target)


if __name__ == "__main__":
    main()
