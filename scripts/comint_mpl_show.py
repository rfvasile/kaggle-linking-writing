# pyright: reportMissingImports=false
# PYTHONSTARTUP shim for the Docker comint REPL: `plt.show()` saves figures and prints markers
# for the Emacs image bridge, and readline gets a completer python.el can parse.
import glob
import io
import os
import re
import readline
import rlcompleter
import time
from contextlib import redirect_stderr, redirect_stdout

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


# rlcompleter probes `getattr(obj, name)` on every candidate: on a torch Tensor `.volatile`
# warns into python.el's completion stream and `.imag` raises, so `t.<TAB>` yields nothing.
class _QuietCompleter(rlcompleter.Completer):
    def attr_matches(self, text):
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                return super().attr_matches(text)
            except Exception:
                return self._dir_matches(text)

    def _dir_matches(self, text):
        """Match on `dir()` alone, reading no attribute off the object; callables lose their "(" postfix."""
        match = re.match(r"(\w+(\.\w+)*)\.(\w*)", text)
        if not match:
            return []
        expr, attr = match.group(1, 3)
        try:
            obj = eval(expr, self.namespace)
        except Exception:
            return []
        # rlcompleter's rule: hide privates until the prefix asks for them.
        hidden = "_" if attr == "" else "__" if attr == "_" else None
        return sorted(
            f"{expr}.{name}" for name in dir(obj) if name.startswith(attr) and not (hidden and name.startswith(hidden))
        )


# `site` runs its readline hook after PYTHONSTARTUP, but its `import rlcompleter` is a no-op by then.
readline.set_completer(_QuietCompleter().complete)
