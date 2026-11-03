#!/usr/bin/env bash
# build-llamacpp-cpu-linux.sh
#
# Goal:
# - Build llama.cpp for Linux, CPU-only backend (no CUDA).
# - Produces the llama-server that CPU_Engine runs: in the CPU AppImage, and in
#   the CUDA AppImage beside the CUDA build (the "Processor only" setting), where
#   it needs no NVIDIA driver and touches no GPU.
#
# Output: backend/artifacts/llama-cpp/cpu/bin/llama-server
#
# Toolchain: a C/C++ compiler (gcc/clang) + cmake. cmake is taken from the dev
# venv if present, else from PATH (CI installs it via pip). Portable, static
# binary (GGML_NATIVE=OFF + BUILD_SHARED_LIBS=OFF) so it runs across Linux x86-64
# hosts and needs no co-located shared libs once bundled.
#
# Usage (from erudi/ or erudi/backend/):
#   bash scripts/dev/backend/build-llamacpp-cpu-linux.sh

set -euo pipefail

# -------- path resolution (run from erudi/ or erudi/backend/) --------
if [ -d "backend/forks/llama-cpp" ]; then
  BACKEND_ROOT="backend"
elif [ -d "forks/llama-cpp" ]; then
  BACKEND_ROOT="."
else
  echo "ERROR: run from the erudi/ or erudi/backend/ directory" >&2
  exit 1
fi

SRC_DIR="${BACKEND_ROOT}/forks/llama-cpp"
BUILD_DIR="${SRC_DIR}/build-cpu"
INSTALL_DIR="${BACKEND_ROOT}/artifacts/llama-cpp/cpu"
BIN_DIR="${INSTALL_DIR}/bin"

# -------- helpers --------
need() { command -v "$1" >/dev/null 2>&1; }
die()  { echo "ERROR: $*" >&2; exit 1; }

[ -f "${SRC_DIR}/CMakeLists.txt" ] || die "llama-cpp source missing at ${SRC_DIR}. Run: git submodule update --init --recursive"
need cc || need gcc || need clang || die "a C compiler (gcc/clang) is required"

# -------- toolchain: prefer venv cmake, fall back to PATH --------
CMAKE="${BACKEND_ROOT}/venv/bin/cmake"
if [ ! -x "$CMAKE" ]; then
  if need cmake; then
    CMAKE="$(command -v cmake)"
  else
    echo "[cmake] not found; installing via pip"
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

# -------- configure (CPU only, portable static binary) --------
echo "[build] Configuring llama.cpp (Linux CPU)..."
"$CMAKE" -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_CPU=ON \
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
echo "[build] Compiling..."
"$CMAKE" --build "$BUILD_DIR" --config Release -j"$(nproc 2>/dev/null || echo 4)"
echo "[install] Installing to $INSTALL_DIR..."
"$CMAKE" --install "$BUILD_DIR" --config Release

# -------- verify --------
[ -x "${BIN_DIR}/llama-server" ] || die "llama-server missing after install (check ${BIN_DIR})"
echo "[done] llama.cpp CPU build complete: ${BIN_DIR}/llama-server"
