"""Command-line verification for the Chapter 5 repeat-generation capsule."""

from __future__ import annotations

import argparse
import json

from .reanalyse import verify_against_frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the recomputed contract as JSON.")
    args = parser.parse_args()
    result = verify_against_frozen()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        counts = result["counts"]
        composite = result["overall"]["composite"]
        print(
            "repeat_generation: "
            f"{counts['completed_signal_pairs']}/720 pairs, "
            f"{counts['completed_component_pairs']} component pairs, "
            f"MAD={composite['mad']}, ICC={composite['icc_3_1']} verified"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
