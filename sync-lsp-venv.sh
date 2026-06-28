#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Last synced container image and mirror format.
STAMP=".venv/.lsp-synced-image"
MODE_STAMP=".venv/.lsp-sync-mode"
SYNC_MODE="source-mirror-v2"
MIRROR_DIR=".venv/kaggle-source"
MIRROR_TMP=".venv/.kaggle-source.tmp"
# Stdlib mirror is only for traceback navigation.
STDLIB_DIR=".venv/kaggle-stdlib"
STDLIB_TMP=".venv/.kaggle-stdlib.tmp"

running_container() {
    docker ps --filter name=kaggle-notebooks --format '{{.Names}}' 2>/dev/null | head -n1 || true
}

# --- check mode -------------------------------------------------------------
if [[ "${1:-}" == "check" ]]; then
    c=$(running_container)
    # Never block shell startup if nothing is running.
    [[ -z "$c" ]] && exit 0
    if [[ ! -f "$STAMP" || ! -f "$MODE_STAMP" || "$(cat "$MODE_STAMP")" != "$SYNC_MODE" ]]; then
        echo "[lsp] Kaggle LSP source mirror has not been synced yet." >&2
        exit 0
    fi
    current=$(docker inspect --format '{{.Image}}' "$c" 2>/dev/null || true)
    if [[ -n "$current" && "$current" != "$(cat "$STAMP")" ]]; then
        echo "[lsp] Kaggle container image changed since the venv was last synced." >&2
        echo "[lsp] Python references may be out of date — run ./sync-lsp-venv.sh to refresh." >&2
    fi
    exit 0
fi

# --- sync mode --------------------------------------------------------------
container=$(running_container)
if [[ -z "$container" ]]; then
    echo "No running Kaggle container found. Start one first:" >&2
    echo "  docker compose --profile gpu up   # or --profile cpu" >&2
    exit 1
fi
echo "Reading Python paths from container: $container"

mapfile -t container_paths < <(
    docker exec -i "$container" python - <<'PY'
import os
import site
import sys
import sysconfig

py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
print(py_ver)
print(os.path.realpath(sysconfig.get_paths()["stdlib"]))

seen = set()


def add_root(path):
    if not path:
        return
    real = os.path.realpath(path)
    if real in seen or not os.path.isdir(real):
        return
    if real == "/kaggle/working" or real.endswith("/lib-dynload"):
        return
    if real.endswith(".zip"):
        return
    # Mirror stdlib separately; keep only package roots here.
    if real.startswith(("/usr/lib/python3.", "/usr/local/lib/python3.")):
        if "site-packages" not in real and "dist-packages" not in real:
            return
    seen.add(real)
    print(real)


for entry in sys.path:
    add_root(entry)
for entry in site.getsitepackages():
    add_root(entry)
try:
    add_root(site.getusersitepackages())
except AttributeError:
    pass
PY
)

if ((${#container_paths[@]} < 3)); then
    echo "Could not discover any Python import roots in the container." >&2
    exit 1
fi

py_ver=${container_paths[0]}
stdlib_root=${container_paths[1]}
roots=("${container_paths[@]:2}")

echo "Container Python: $py_ver"
echo "  stdlib root: $stdlib_root"
printf '  source root: %s\n' "${roots[@]}"

venv_ver=""
if [[ -x .venv/bin/python ]]; then
    venv_ver=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi
if [[ "$venv_ver" != "$py_ver" || ! -f "$MODE_STAMP" || "$(cat "$MODE_STAMP" 2>/dev/null || true)" != "$SYNC_MODE" ]]; then
    echo "Recreating .venv on Python $py_ver (was: ${venv_ver:-none})"
    rm -rf .venv
    uv venv --python "$py_ver" .venv
fi

copy_source_root() {
    local dest=$1 root=$2
    printf '  mirroring: %s\n' "$root"
    # Prune nested package roots.
    docker exec -i "$container" bash -s -- "$root" <<'BASH' | tar -C "$dest" -xf -
set -euo pipefail

root=$1
cd "$root"

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

find . \( -name 'dist-packages' -o -name 'site-packages' -o -name 'lib-dynload' \) -prune -o \
    \( -type f -o -xtype f \) \( \
    -name '*.py' -o \
    -name '*.pyi' -o \
    -name 'py.typed' \
\) -print0 >"$tmp"

if [[ -s "$tmp" ]]; then
    tar --dereference --null --files-from="$tmp" --create --file=-
else
    tar --create --file=- --files-from=/dev/null
fi
BASH
}

rm -rf "$MIRROR_TMP" "$STDLIB_TMP"
mkdir -p "$MIRROR_TMP" "$STDLIB_TMP"

for ((i = ${#roots[@]} - 1; i >= 0; i--)); do
    copy_source_root "$MIRROR_TMP" "${roots[$i]}"
done
copy_source_root "$STDLIB_TMP" "$stdlib_root"

rm -rf "$MIRROR_DIR" "$STDLIB_DIR"
mv "$MIRROR_TMP" "$MIRROR_DIR"
mv "$STDLIB_TMP" "$STDLIB_DIR"

docker inspect --format '{{.Image}}' "$container" >"$STAMP"
printf '%s\n' "$SYNC_MODE" >"$MODE_STAMP"

echo
echo "Done. Restart Emacs for the changes to take effect."
