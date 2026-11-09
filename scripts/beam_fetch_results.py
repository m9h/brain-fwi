"""Fetch result files from the beam Volume `brain-fwi-results` (written by beam_gmwm_ladder.py).

Needed because a multi-hour `.remote()` call can lose its gRPC stream ("Unhandled GRPC
error: StatusCode.INTERNAL") — the REMOTE task keeps running and still writes its outputs
to the Volume, but the bytes never reach the local caller. The beam CLI has no `volume ls/cp`,
so this is a tiny CPU-only @function that mounts the same Volume and returns the files.

    cd scripts && ../.venv-beam/bin/python beam_fetch_results.py --list
    cd scripts && ../.venv-beam/bin/python beam_fetch_results.py --tag s2B      # -> results/gmwm_velocity_3d/
"""
import argparse, os
from beam import function, Image, Volume

VOL = Volume(name="brain-fwi-results", mount_path="/vol")


@function(cpu=1, memory="2Gi", image=Image(python_version="python3.11"), volumes=[VOL], timeout=600)
def fetch(subdir, tag, list_only):
    import os, glob
    d = f"/vol/{subdir}"
    if not os.path.isdir(d):
        return {"files": {}, "listing": []}
    listing = sorted((os.path.basename(f), os.path.getsize(f)) for f in glob.glob(f"{d}/*"))
    if list_only:
        return {"files": {}, "listing": listing}
    files = {}
    for f in glob.glob(f"{d}/*{tag}*"):
        files[os.path.basename(f)] = open(f, "rb").read()
    return {"files": files, "listing": listing}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="gmwm_velocity_3d")
    ap.add_argument("--tag", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    res = fetch.remote(a.subdir, a.tag, a.list or not a.tag)
    for name, size in res["listing"]:
        print(f"  {size:>10d}  {name}")
    if res["files"]:
        OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", a.subdir)
        os.makedirs(OUT, exist_ok=True)
        for name, data in res["files"].items():
            with open(os.path.join(OUT, name), "wb") as fh:
                fh.write(data)
            print(f"saved results/{a.subdir}/{name} ({len(data)} bytes)")
