#!/usr/bin/env python
"""Compare voxel-path and SIREN-path FWI reconstructions.

Loads two HDF5 files produced by ``run_full_usct.py`` (one with
``--parameterization voxel``, one with ``--parameterization siren``)
and prints a regional-RMSE comparison table.

Usage::

    python scripts/validate_siren_vs_voxel.py \\
        --voxel results/voxel.h5 --siren results/siren.h5
"""

import argparse
import json
import sys
from pathlib import Path

from brain_fwi.validation import compare_reconstructions
from brain_fwi.validation.compare import format_comparison


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voxel", type=Path, required=True,
                    help="Path to run_full_usct.py output with parameterization=voxel")
    ap.add_argument("--siren", type=Path, required=True,
                    help="Path to run_full_usct.py output with parameterization=siren")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="If set, also dump comparison as JSON")
    args = ap.parse_args()

    result = compare_reconstructions(voxel_path=args.voxel, siren_path=args.siren)
    print(format_comparison(result))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nJSON dump: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
