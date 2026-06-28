# pyright: reportMissingImports=false
# PYTHONSTARTUP shim for the Docker comint REPL. `plt.show()` saves figures and
# prints markers consumed by the Emacs image bridge.
import glob
import os
import time

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as _plt

_PLOT_DIR = "output/.comint-plots"
_plot_count = 0

# Prune old session images.
_cutoff = time.time() - 3 * 86400
for _f in glob.glob(f"{_PLOT_DIR}/fig-*.png"):
    try:
        if os.path.getmtime(_f) < _cutoff:
            os.remove(_f)
    except OSError:
        pass


def _show(*_args, **_kwargs):
    global _plot_count
    os.makedirs(_PLOT_DIR, exist_ok=True)
    # Keep root-created files host-writable.
    os.chmod(_PLOT_DIR, 0o777)
    for num in _plt.get_fignums():
        _plot_count += 1
        path = f"{_PLOT_DIR}/fig-{os.getpid()}-{_plot_count:04d}.png"
        _plt.figure(num).savefig(path, bbox_inches="tight")
        print(f"__OPEN_IMAGE__ {path}", flush=True)
    # Match notebook-cell behavior.
    _plt.close("all")


_plt.show = _show
