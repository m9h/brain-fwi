"""Build the 3D Birnbaum cerebrum-volume dataset for the 3D diffusion prior.

Holds out the last 4 subjects for the 3D FWI target, builds one cerebrum volume
(+ L-R flip) per training head via :func:`brain_fwi.phantoms.birnbaum.build_volume_dataset`,
and saves a flat ``(N, S^3)`` array + normalisation stats. Ship the npz to the
GPU trainer (``scripts/modal_train_unet3d.py``).

    python scripts/build_birnbaum_3d_dataset.py --out /tmp/birnbaum_3d_dataset.npz
"""
import argparse
import numpy as np
from brain_fwi.phantoms import birnbaum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/birnbaum_3d_dataset.npz")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--holdout", type=int, default=4, help="subjects reserved for the FWI target")
    args = ap.parse_args()

    files = birnbaum.label_files()
    train = files[: -args.holdout] if args.holdout else files
    data = birnbaum.build_volume_dataset(train, S=args.S)
    np.savez(args.out, data=data, mean=data.mean(), std=data.std(), S=args.S)
    print(f"3D dataset {data.shape} (S={args.S}), mean {data.mean():.0f} std {data.std():.1f}, "
          f"lesion-bearing {(data.max(1) > 1650).sum()}/{len(data)} -> {args.out}")


if __name__ == "__main__":
    main()
