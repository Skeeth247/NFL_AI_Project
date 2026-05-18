#!/usr/bin/env python3
"""
check_leakage.py — CLI runner for data-leakage audit checks.

Usage:
    python scripts/check_leakage.py
    python scripts/check_leakage.py --csv data/modeling_table.csv
    python scripts/check_leakage.py --csv data/modeling_table.csv --feature-json data/features.json

Exit codes:
    0 — all checks PASS or WARN/SKIP (no FAILs)
    1 — one or more checks FAIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_chatbot.evaluation.leakage_checks import run_all_checks

# ── ANSI colours (disabled on non-tty) ───────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

_COLOR = {
    "PASS":  "\033[32m",   # green
    "FAIL":  "\033[31m",   # red
    "WARN":  "\033[33m",   # yellow
    "SKIP":  "\033[90m",   # dark grey
    "reset": "\033[0m",
    "bold":  "\033[1m",
    "dim":   "\033[2m",
}


def _c(text: str, *styles: str) -> str:
    if not _USE_COLOR:
        return text
    prefix = "".join(_COLOR.get(s, "") for s in styles)
    return f"{prefix}{text}{_COLOR['reset']}"


# ── Formatting ────────────────────────────────────────────────────────────────

_STATUS_WIDTH = 5    # "PASS " / "FAIL " / "WARN " / "SKIP "
_CAT_WIDTH    = 18
_NAME_WIDTH   = 50


def _banner(text: str) -> None:
    width = 72
    print()
    print(_c("─" * width, "dim"))
    print(_c(f"  {text}", "bold"))
    print(_c("─" * width, "dim"))


def _print_result_row(r) -> None:
    badge = _c(f"[{r.status:<4}]", r.status)
    name  = r.name[:_NAME_WIDTH]
    cat   = r.category[:_CAT_WIDTH]
    detail = r.detail[:80]
    print(f"  {badge}  {cat:<{_CAT_WIDTH}}  {name:<{_NAME_WIDTH}}")
    if r.status in {"FAIL", "WARN"} and r.detail:
        print(f"           {_c(detail, 'dim')}")
    if r.extra:
        for k, v in r.extra.items():
            print(f"           {_c(k, 'dim')}: {v}")


def _print_summary(results: list) -> None:
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    total = len(results)
    print()
    print(_c("═" * 72, "bold"))
    print(_c("  Summary", "bold"))
    print(_c("═" * 72, "bold"))
    print(
        f"  {_c(str(counts['PASS']), 'PASS')} passed  "
        f"{_c(str(counts['FAIL']), 'FAIL')} failed  "
        f"{_c(str(counts['WARN']), 'WARN')} warned  "
        f"{_c(str(counts['SKIP']), 'SKIP')} skipped  "
        f"({total} total)"
    )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run data-leakage audit checks for the NFL AI Chatbot.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Path to modeling_table CSV (enables file-based checks).",
    )
    parser.add_argument(
        "--feature-json",
        metavar="PATH",
        default=None,
        help="Path to feature column JSON (enables feature-list checks).",
    )
    parser.add_argument(
        "--category",
        metavar="CAT",
        default=None,
        help="Only run checks in this category (e.g. rolling, h2h, elo).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first FAIL.",
    )
    args = parser.parse_args()

    print(_c("\nNFL AI Chatbot — Data Leakage Audit", "bold"))
    if args.csv:
        print(f"  modeling CSV  : {args.csv}")
    if args.feature_json:
        print(f"  feature JSON  : {args.feature_json}")

    results = run_all_checks(
        modeling_csv=args.csv,
        feature_json=args.feature_json,
    )

    # Optional category filter (post-run for simplicity)
    if args.category:
        results = [r for r in results if r.category == args.category]
        if not results:
            print(f"\nNo checks found for category '{args.category}'.")
            return 0

    # Group by category
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    any_fail = False
    for cat, cat_results in by_cat.items():
        _banner(cat.upper())
        for r in cat_results:
            _print_result_row(r)
            if r.status == "FAIL":
                any_fail = True
                if args.fail_fast:
                    _print_summary(results)
                    print(_c("  Stopped at first FAIL (--fail-fast).\n", "FAIL"))
                    return 1

    _print_summary(results)

    if any_fail:
        print(_c("  AUDIT FAILED — fix leakage issues before training models.\n", "FAIL"))
        return 1

    print(_c("  AUDIT PASSED — no leakage detected.\n", "PASS"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
