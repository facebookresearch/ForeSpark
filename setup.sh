#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#
# Build the venv: pinned vLLM + speculators, our two patches on top, and this
# package installed editable.
#
#   ./setup.sh        # idempotent; re-run after editing patches/
#
# Re-running resets the pinned checkouts and re-applies patches/, so edit the
# patch files, not the checkouts.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="${ROOT}/.venv/src"

# vLLM v0.30.0 (DFlash2, DSpark, adaptive verification, the GLM-5.3 models). Its
# compiled binaries come from the PyPI wheel of the same release, built from
# this commit.
VLLM_URL=https://github.com/vllm-project/vllm.git
VLLM_REV=ced6857afa0ea7b2e3f0846a62e1394e90f15607
VLLM_VERSION=0.30.0
SPECULATORS_URL=https://github.com/vllm-project/speculators.git
SPECULATORS_REV=7a58fc56217632d8d179b665734fa2269e8d9ffa
UV_VERSION=0.11.15
# FlashInfer: library, cubins and JIT cache must agree, and the JIT cache
# must match torch's CUDA build.
FLASHINFER_VERSION=0.6.18.post1
FLASHINFER_CUDA=cu130

pin() { # url rev dest -- force-checkout rev, discarding previously applied patches
    if [[ ! -d "$3/.git" ]]; then
        mkdir -p -- "$(dirname -- "$3")"
        git clone --quiet "$1" "$3"
    fi
    git -C "$3" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$3" fetch --quiet origin "$2"
    git -C "$3" -c advice.detachedHead=false checkout --quiet --force "$2"
    printf '  %-12s %s\n' "$(basename -- "$3")" "$(git -C "$3" rev-parse --short HEAD)"
}

patch_checkout() { # checkout patchfile
    git -C "$1" apply --check "$2" || {
        echo "$(basename -- "$2") does not apply to $(basename -- "$1") -- pins moved?" >&2
        exit 1
    }
    git -C "$1" apply "$2"
    echo "  applied $(basename -- "$2")"
}

echo "creating venv"
if command -v uv >/dev/null; then
    UV=uv
elif [[ -x "${ROOT}/.tools/bin/uv" ]]; then
    UV="${ROOT}/.tools/bin/uv"
else
    mkdir -p -- "${ROOT}/.tools/bin"
    curl --fail --location --silent --show-error \
        "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | UV_INSTALL_DIR="${ROOT}/.tools/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
    UV="${ROOT}/.tools/bin/uv"
fi

# configs/paths.yaml says where the venv lives; pyyaml does not exist yet.
VENV="$(python3 - "$ROOT" <<'PY'
import os, re, sys
root = sys.argv[1]
text = open(os.path.join(root, "configs/paths.yaml")).read()
match = re.search(r"^venv:[ \t]*([^\n#]*)", text, re.M)
value = os.path.expandvars(os.path.expanduser((match.group(1).strip() if match else "") or ".venv"))
print(value if os.path.isabs(value) else os.path.join(root, value))
PY
)"
if [[ "$VENV" != "${ROOT}/.venv" && -e "${ROOT}/.venv" && ! -L "${ROOT}/.venv" ]]; then
    echo "${ROOT}/.venv is a real directory but paths.yaml points at ${VENV}; move it first." >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$VENV")"
"$UV" venv --python 3.12 --allow-existing "$VENV"
if [[ "$VENV" != "${ROOT}/.venv" ]]; then
    ln -sfn -- "$VENV" "${ROOT}/.venv"
fi
# vLLM's precompiled extension needs torch's CUDA symbols loaded globally.
printf '%s\n' 'import os; os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")' \
    > "${VENV}/lib/python3.12/site-packages/forespark.pth"
PY="${ROOT}/.venv/bin/python"

echo "pinning sources"
pin "$VLLM_URL" "$VLLM_REV" "${SRC}/vllm"
pin "$SPECULATORS_URL" "$SPECULATORS_REV" "${SRC}/speculators"

echo "applying patches"
patch_checkout "${SRC}/vllm" "${ROOT}/patches/vllm.patch"
patch_checkout "${SRC}/speculators" "${ROOT}/patches/speculators.patch"

echo "installing vllm (precompiled binaries from the PyPI wheel)"
# Without build isolation, so the build sees the venv's torch instead of
# starting a multi-hour CUDA compile in a throwaway environment. The binaries
# come from the release's PyPI wheel (sha256 checked against PyPI), cached in
# the venv; VLLM_WHEEL=/path/to/vllm-*.whl ./setup.sh uses a local one instead.
"$UV" pip install --python "$PY" -r "${SRC}/vllm/requirements/build/cuda.txt"
WHEEL="${VLLM_WHEEL:-$("$PY" - "$VLLM_VERSION" "${VENV}/wheels" <<'PY'
import hashlib, json, os, sys, urllib.request
version, cache = sys.argv[1], sys.argv[2]
release = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/vllm/{version}/json"))
wheel = next(f for f in release["urls"] if f["filename"].endswith("_x86_64.whl"))
path = os.path.join(cache, wheel["filename"])
if not os.path.exists(path):
    os.makedirs(cache, exist_ok=True)
    urllib.request.urlretrieve(wheel["url"], path + ".part")
    os.replace(path + ".part", path)
with open(path, "rb") as handle:
    if hashlib.sha256(handle.read()).hexdigest() != wheel["digests"]["sha256"]:
        os.remove(path)
        raise SystemExit(f"{wheel['filename']}: sha256 differs from PyPI's")
print(path)
PY
)}"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
    "$UV" pip install --python "$PY" --no-build-isolation -e "${SRC}/vllm"

echo "installing speculators and tools"
"$UV" pip install --python "$PY" \
    -e "${SRC}/speculators" -e "${SRC}/speculators/hs_connectors" \
    datasets transformers tensorboard matplotlib pyyaml pandas numpy pytest

echo "installing flashinfer"
"$PY" - "$FLASHINFER_CUDA" <<'PY'
import sys
import torch
built = "cu" + (torch.version.cuda or "none").replace(".", "")
if built != sys.argv[1]:
    raise SystemExit(f"FLASHINFER_CUDA={sys.argv[1]} but torch is built for {built}")
PY
"$UV" pip install --python "$PY" "flashinfer-python==${FLASHINFER_VERSION}"
"$UV" pip install --python "$PY" --no-deps --link-mode copy \
    --default-index https://flashinfer.ai/whl/ "flashinfer-cubin==${FLASHINFER_VERSION}"
"$UV" pip install --python "$PY" --no-deps --link-mode copy \
    --default-index "https://flashinfer.ai/whl/${FLASHINFER_CUDA}/" \
    "flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CUDA}"

echo "installing forespark"
# Isolated build (a clean setuptools); --no-deps so nothing can move the pins.
"$UV" pip install --python "$PY" --no-deps -e "${ROOT}"

cat <<EOF

done. activate with: source .venv/bin/activate
then see README.md.
EOF
