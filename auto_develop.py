#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from raw_develop import AutoDeveloper, collect_raw_files


def main():
    parser = argparse.ArgumentParser(
        description="Automatic RAW developer v29"
    )

    parser.add_argument(
        "input",
        type=Path,
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("developed"),
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Segmentation device",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images/statistics",
    )

    args = parser.parse_args()

    raw_files = collect_raw_files(args.input)

    if not raw_files:
        print("No RAW files found.")
        return 1

    developer = AutoDeveloper(
        device=args.device,
        debug=args.debug,
    )

    for raw_path in raw_files:
        if args.input.is_file():
            relative = raw_path.name
        else:
            try:
                relative = raw_path.relative_to(args.input)
            except ValueError:
                relative = raw_path.name

        output_name = Path(relative).with_suffix(".jpg")
        output_path = args.output / output_name

        if args.debug:
            developer.debug_dir = (
                args.output
                / Path(relative).with_suffix("")
                / "debug"
            )

        try:
            developer.process_file(
                raw_path,
                output_path,
            )
        except Exception as exc:
            print()
            print(f"[ERROR] {raw_path}")
            print(f"{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 72)
    print("Finished.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
