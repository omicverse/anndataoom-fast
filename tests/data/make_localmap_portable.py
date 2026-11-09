"""Regenerate tests/data/localmap_portable.npz: anndataoom_fast.localmap outputs
(default, portable 1/sqrt start) for the cases of make_embed_fixtures.py.
These are machine independent, unlike brisc's LocalMAP (RSQRTSS)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import test_embed as t  # noqa: E402

out = {f"{name}_localmap": t.run(name, "localmap", 3) for name in sorted(t.mk.CASES)}
np.savez_compressed(HERE / "localmap_portable.npz", **out)
print("saved", list(out))
