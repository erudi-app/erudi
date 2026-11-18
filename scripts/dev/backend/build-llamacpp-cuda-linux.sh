#!/usr/bin/env bash
# build-llamacpp-cuda-linux.sh
#
# Goal:
# - Build llama.cpp for Linux with the NVIDIA CUDA backend.
# - Produces the GPU-accelerated llama-server used by CUDA_Engine on Linux.
#   CPU layers stay ON for models larger than VRAM.
#
# Output: backend/artifacts/llama-cpp/cuda/bin/llama-server
#
# Prerequisites:
# - The CUDA toolkit (nvcc) — on PATH, or under $CUDA_HOME / $CUDA_PATH, or the
#   default /usr/local/cuda. Compilation needs the toolkit but NOT a GPU (only
#   runtime does). CI installs it via Jimver/cuda-toolkit.
# - A C/C++ compiler + cmake (venv or PATH; pip-installed on CI).
#
# Usage (from erudi/ or erudi/backend/):
#   bash scripts/dev/backend/build-llamacpp-cuda-linux.sh

set -euo pipefail

# -------- path resolution --------
if [ -d "backend/forks/llama-cpp" ]; then
  BACKEND_ROOT="backend"
elif [ -d "forks/llama-cpp" ]; then
  BACKEND_ROOT="."
else
  echo "ERROR: run from the erudi/ or erudi/backend/ directory" >&2
  exit 1
fi

SRC_DIR="${BACKEND_ROOT}/forks/llama-cpp"
BUILD_DIR="${SRC_DIR}/build-cuda"
INSTALL_DIR="${BACKEND_ROOT}/artifacts/llama-cpp/cuda"
BIN_DIR="${INSTALL_DIR}/bin"

need() { command -v "$1" >/dev/null 2>&1; }
die()  { echo "ERROR: $*" >&2; exit 1; }

[ -f "${SRC_DIR}/CMakeLists.txt" ] || die "llama-cpp source missing at ${SRC_DIR}. Run: git submodule update --init --recursive"
need cc || need gcc || die "a C compiler (gcc) is required"

# -------- locate nvcc --------
NVCC=""
if need nvcc; then
  NVCC="$(command -v nvcc)"
else
  for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda; do
    [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && { NVCC="$d/bin/nvcc"; export PATH="$d/bin:$PATH"; break; }
  done
fi
[ -n "$NVCC" ] || die "nvcc (CUDA toolkit) not found. Install CUDA or set CUDA_HOME."
echo "[cuda] nvcc: $("$NVCC" --version | grep -i release || echo "$NVCC")"

# -------- architectures to compile for --------
# Suffixes matter and a bare number is NOT what we want: bare "86" emits both
# PTX and SASS, so the old bare list produced fourteen code objects where seven
# do the job. This mirrors upstream's own release default -- `-virtual` == PTX
# only, JIT-compiled by the driver on first run; `-real` == native SASS.
# Forward compatibility to any architecture newer than what we list comes from
# the 80-virtual PTX, exactly as upstream intends.
CUDA_ARCHS="50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real"
# sm_120 (Blackwell / RTX 50) only exists from CUDA 12.8: asking an older nvcc
# for it fails the build outright. Add it when the toolkit can emit it, so those
# cards get native code instead of a JIT pass, and stay buildable on 12.x below.
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [ -n "$CUDA_VER" ] && [ "$(printf '%s\n12.8\n' "$CUDA_VER" | sort -V | head -1)" = "12.8" ]; then
  CUDA_ARCHS="${CUDA_ARCHS};120-real"
  echo "[cuda] toolkit ${CUDA_VER}: adding native Blackwell (sm_120) code"
else
  echo "[cuda] toolkit ${CUDA_VER:-unknown} is below 12.8: Blackwell will JIT from PTX instead of running native code."
fi

# -------- toolchain: prefer venv cmake, fall back to PATH --------
CMAKE="${BACKEND_ROOT}/venv/bin/cmake"
if [ ! -x "$CMAKE" ]; then
  if need cmake; then
    CMAKE="$(command -v cmake)"
  else
    "${BACKEND_ROOT}/venv/bin/pip" install --upgrade cmake >/dev/null 2>&1 || pip install --upgrade cmake >/dev/null
    CMAKE="$(command -v cmake)"
  fi
fi
[ -n "$CMAKE" ] || die "cmake not found and could not be installed"
echo "[cmake] $("$CMAKE" --version | head -n1)"

# -------- clean (non-interactive on CI / non-TTY) --------
if [ -d "$BUILD_DIR" ] || [ -d "$INSTALL_DIR" ]; then
  if [ -t 0 ]; then
    read -rp "Existing build output found. Delete before rebuild? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] && rm -rf "$BUILD_DIR" "$INSTALL_DIR"
  else
    rm -rf "$BUILD_DIR" "$INSTALL_DIR"
  fi
fi
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

# -------- configure (CUDA backend + CPU fallback layers) --------
# Build ONLY what we ship (the server, under tools/). The 38 examples/ binaries
# each statically embed the large CUDA lib (BUILD_SHARED_LIBS=OFF), and linking
# the full suite overflows the Linux runner disk ("ld: No space left on device")
# as the submodule grows — so examples + tests are off. The Windows runner has the
# headroom, so only the Linux CUDA leg needs this.
echo "[build] Configuring llama.cpp (Linux CUDA)..."
"$CMAKE" -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DGGML_CPU=ON \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP=OFF \
  -DGGML_SYCL=OFF \
  -DGGML_RPC=OFF \
  -DGGML_WEBGPU=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_BLAS=OFF

# -------- build + install --------
echo "[build] Compiling (this takes a while)..."
"$CMAKE" --build "$BUILD_DIR" --config Release -j"$(nproc 2>/dev/null || echo 4)"
echo "[install] Installing to $INSTALL_DIR..."
"$CMAKE" --install "$BUILD_DIR" --config Release

# -------- verify --------
[ -x "${BIN_DIR}/llama-server" ] || die "llama-server missing after install (check ${BIN_DIR})"
echo "[done] llama.cpp CUDA build complete: ${BIN_DIR}/llama-server"
