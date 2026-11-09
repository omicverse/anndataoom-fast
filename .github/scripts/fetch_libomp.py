"""Download conda-forge's llvm-openmp for this macOS arch and unpack it.

    python fetch_libomp.py DEST [SUBDIR]

conda-forge builds libomp for macOS 11 (minos 11.0 on both arches), so wheels
linked against it install on old macOS too (Homebrew's targets the runner's
own macOS).  Only conda-forge's CDN is needed: no conda/micromamba install.
"""
import io
import json
import platform
import sys
import tarfile
import time
import urllib.request
import zipfile

import zstandard

DEST = sys.argv[1]
SUBDIR = (sys.argv[2] if len(sys.argv) > 2
          else "osx-arm64" if platform.machine() == "arm64" else "osx-64")


def get(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"retry {attempt + 1}: {url}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"could not download {url}")


repodata = json.loads(get(f"https://conda.anaconda.org/conda-forge/{SUBDIR}/repodata.json"))
pkgs = {**repodata.get("packages", {}), **repodata.get("packages.conda", {})}
cands = [(fn, m) for fn, m in pkgs.items() if m["name"] == "llvm-openmp"]


def key(item):
    fn, m = item
    v = tuple(int(x) if x.isdigit() else 0 for x in m["version"].split("."))
    return v, m.get("build_number", 0), m.get("timestamp", 0)


fn, meta = max(cands, key=key)
print("llvm-openmp", meta["version"], fn, flush=True)
blob = get(f"https://conda.anaconda.org/conda-forge/{SUBDIR}/{fn}")
if fn.endswith(".conda"):
    z = zipfile.ZipFile(io.BytesIO(blob))
    inner = next(n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
    data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(inner)))
    tf = tarfile.open(fileobj=data, mode="r|")
else:
    tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:bz2")
n = 0
for member in tf:
    if member.name.startswith(("lib/", "include/")):
        tf.extract(member, DEST)
        n += 1
print("extracted", n, "files to", DEST)
