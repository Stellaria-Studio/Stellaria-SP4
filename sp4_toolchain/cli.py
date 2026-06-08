from __future__ import annotations

import argparse

from .benchmark import add_benchmark_parser
from .compiler import add_compile_parser
from .inspector import add_inspect_parser
from .validator import add_validate_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sp4", description="Stellaria SP4 A1P toolchain")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_compile_parser(subparsers)
    add_inspect_parser(subparsers)
    add_validate_parser(subparsers)
    add_benchmark_parser(subparsers)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
