from __future__ import annotations

from pathlib import Path

from .constants import RAW_EXTENSIONS

def collect_raw_files(
    input_path: Path,
) -> list[Path]:

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            return [input_path]

        return []

    files = []

    for path in input_path.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(path)

    files.sort()

    return files
