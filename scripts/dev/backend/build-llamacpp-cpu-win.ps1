# build-llamacpp-cpu-win.ps1
#
# Goal:
# - Build llama.cpp for Windows, CPU-only backend (no CUDA).
# - Produces the llama-server.exe that CPU_Engine runs: in the CPU installer,
#   and in the CUDA installer beside the CUDA build (the "Processor only"
#   setting), where it needs no NVIDIA driver and touches no GPU. Building the
#   CUDA installer therefore needs this script AND build-llamacpp-cuda-win.ps1.
#
# Prerequisites:
# - Visual Studio 2019 or 2022 with the "Desktop development with C++" workload
#   (Build Tools are enough), including "C++ CMake tools for Windows", which
#   provides Ninja.
# - Python + cmake. Taken from the dev venv (backend\venv\Scripts) if present,
#   else from PATH (CI installs cmake via pip into the system Python).
# - The llama-cpp submodule populated: git submodule update --init --recursive.
#
# Usage (run from erudi\ or erudi\backend\):
#   .\scripts\dev\backend\build-llamacpp-cpu-win.ps1
#
# Visual Studio is found the same way as by build-llamacpp-cuda-win.ps1:
# through vswhere (under ${env:ProgramFiles(x86)}, else on PATH). Set
# ERUDI_VCVARSALL to the full path of vcvarsall.bat to skip vswhere, e.g. when
# the toolchain lives on another drive:
#   $env:ERUDI_VCVARSALL = "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
#
# Output:
#   backend\artifacts\llama-cpp\cpu\bin\llama-server.exe

$ErrorActionPreference = "Stop"

# -------- helpers --------
function Write-Step  { Write-Host "[build]   $args" -ForegroundColor Cyan }
function Write-OK    { Write-Host "[ok]      $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[warning] $args" -ForegroundColor Yellow }
function Write-Fail  { Write-Host "[error]   $args" -ForegroundColor Red; exit 1 }

# -------- path resolution (run from erudi\ or erudi\backend\) --------
$currentDir = (Get-Item .).Name
if ($currentDir -eq "backend") {
    $BackendRoot = "."
} elseif (Test-Path "backend") {
    $BackendRoot = "backend"
} else {
    Write-Fail "Run this script from the erudi\ or erudi\backend\ directory."
}

$SrcDir     = Join-Path $BackendRoot "forks\llama-cpp"
$BuildDir   = Join-Path $SrcDir      "build-cpu"
$InstallDir = Join-Path $BackendRoot "artifacts\llama-cpp\cpu"
$BinDir     = Join-Path $InstallDir  "bin"

# Toolchain: prefer the dev venv, fall back to PATH (CI installs into system Python).
$VenvPip   = Join-Path $BackendRoot "venv\Scripts\pip.exe"
$VenvCmake = Join-Path $BackendRoot "venv\Scripts\cmake.exe"
if (-not (Test-Path $VenvPip))   { $VenvPip   = (Get-Command pip   -ErrorAction SilentlyContinue).Source }
if (-not (Test-Path $VenvCmake)) { $VenvCmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source }

Write-Step "Paths resolved:"
Write-Host "  Source  : $SrcDir"
Write-Host "  Build   : $BuildDir"
Write-Host "  Install : $InstallDir"

# -------- sanity checks --------
Write-Step "Checking prerequisites..."

if (-not (Test-Path (Join-Path $SrcDir "CMakeLists.txt"))) {
    Write-Fail "llama-cpp source not found at $SrcDir. Run: git submodule update --init --recursive"
}

# cmake (install into the env if missing)
if (-not $VenvCmake) {
    Write-Step "cmake not found. Installing via pip..."
    if (-not $VenvPip) { Write-Fail "Neither cmake nor pip found on PATH." }
    & $VenvPip install --upgrade cmake --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install cmake." }
    $VenvCmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
}
if (-not $VenvCmake) { Write-Fail "cmake still not found after pip install." }
Write-OK "cmake: $((& $VenvCmake --version 2>&1 | Select-Object -First 1))"

# -------- Visual Studio detection + MSVC environment bootstrap --------
# Same bootstrap as build-llamacpp-cuda-win.ps1: vcvarsall.bat's environment
# is loaded into this session and the build runs with Ninja, so a machine whose
# Visual Studio the CMake VS generator cannot locate (a toolchain on another
# drive, or vswhere's instance state lost to a Windows reinstall) still builds
# once ERUDI_VCVARSALL points at vcvarsall.bat.
Write-Step "Detecting Visual Studio (C++ build tools) and bootstrapping MSVC environment..."

$VcVarsAll = $null

if ($env:ERUDI_VCVARSALL) {
    # An explicit vcvarsall.bat wins: a toolchain on another drive, or a machine
    # whose vswhere instance state (kept under ProgramData) is gone.
    if (-not (Test-Path $env:ERUDI_VCVARSALL)) {
        Write-Fail "ERUDI_VCVARSALL points to a file that does not exist: $env:ERUDI_VCVARSALL"
    }
    $VcVarsAll = $env:ERUDI_VCVARSALL
} else {
    # vswhere ships with the Visual Studio Installer under Program Files (x86),
    # wherever that folder lives; else one on PATH.
    $VsWhere = $null
    if (${env:ProgramFiles(x86)}) {
        $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $candidate) { $VsWhere = $candidate }
    }
    if (-not $VsWhere) {
        $VsWhereCmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
        if ($VsWhereCmd) { $VsWhere = $VsWhereCmd.Source }
    }
    if ($VsWhere) {
        $VsInstallPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>&1
        if ($VsInstallPath) {
            $candidate = Join-Path $VsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
            if (Test-Path $candidate) { $VcVarsAll = $candidate }
        }
    }
}

if (-not $VcVarsAll) {
    Write-Fail (
        "vcvarsall.bat not found. Install Visual Studio 2019/2022 with the `n" +
        "'Desktop development with C++' workload, open a VS Developer prompt and re-run, `n" +
        "or set ERUDI_VCVARSALL to the full path of vcvarsall.bat."
    )
}

Write-OK "Found vcvarsall.bat: $VcVarsAll"
Write-Step "Bootstrapping MSVC x64 environment into current session..."

# Capture env vars set by vcvarsall.bat and apply them to this PowerShell session
$tmpEnvFile = [System.IO.Path]::GetTempFileName()
try {
    cmd /c "`"$VcVarsAll`" x64 > nul 2>&1 && set" | Set-Content $tmpEnvFile
    Get-Content $tmpEnvFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
} finally {
    Remove-Item $tmpEnvFile -Force -ErrorAction SilentlyContinue
}

# Verify cl.exe is now in PATH
$ClExe = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $ClExe) {
    Write-Fail "cl.exe not found in PATH after vcvarsall bootstrap. Check VS installation."
}
Write-OK "MSVC cl.exe: $($ClExe.Source)"

# Verify ninja is available (ships with VS Build Tools)
$NinjaExe = Get-Command ninja -ErrorAction SilentlyContinue
if (-not $NinjaExe) {
    Write-Fail (
        "ninja not found in PATH after vcvarsall bootstrap.`n" +
        "Ensure 'C++ CMake tools for Windows' is installed in VS Build Tools."
    )
}
Write-OK "Ninja: $($NinjaExe.Source)"

# -------- clean previous output (non-interactive: CI runners start fresh) --------
if ((Test-Path $BuildDir) -or (Test-Path $InstallDir)) {
    Write-Step "Removing previous build output..."
    if (Test-Path $BuildDir)   { Remove-Item -Path $BuildDir   -Recurse -Force }
    if (Test-Path $InstallDir) { Remove-Item -Path $InstallDir -Recurse -Force }
    Write-OK "Cleaned."
}
New-Item -ItemType Directory -Force -Path $BuildDir   | Out-Null
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# -------- cmake configure (CPU-only) --------
# Ninja with the MSVC environment bootstrapped above (x64), as for the CUDA build.
Write-Host ""
Write-Step "Configuring llama.cpp (CPU-only backend)..."

$CmakeArgs = @(
    "-G", "Ninja",
    "-S", $SrcDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",          # static: a single self-contained exe
    "-DCMAKE_INSTALL_PREFIX=$InstallDir",
    "-DLLAMA_CURL=OFF",                 # no libcurl dependency for server-only use
    # ---- backend flags: CPU only ----
    "-DGGML_CUDA=OFF",
    "-DGGML_CPU=ON",
    "-DGGML_NATIVE=OFF",                # portable: no host-specific ISA baked in (clients vary)
    "-DGGML_METAL=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_HIP=OFF",
    "-DGGML_SYCL=OFF",
    "-DGGML_RPC=OFF",
    "-DGGML_WEBGPU=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_BLAS=OFF"
)

Write-Host "  cmake $($CmakeArgs -join ' ')"
Write-Host ""

& $VenvCmake @CmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Fail "cmake configuration failed. Check output above." }

# -------- cmake build --------
Write-Host ""
Write-Step "Compiling (CPU, this takes a few minutes)..."

$CpuCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
& $VenvCmake --build $BuildDir --config Release --parallel $CpuCount
if ($LASTEXITCODE -ne 0) { Write-Fail "Build failed. Check output above." }

# -------- cmake install --------
Write-Host ""
Write-Step "Installing to $InstallDir..."

& $VenvCmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Fail "Install step failed." }

# -------- bundle the MSVC C++ runtime next to llama-server.exe (#144) --------
# The CPU build links the CRT dynamically (/MD), so llama-server.exe imports
# vcruntime140/vcruntime140_1/msvcp140. On a clean Windows machine without the
# VC++ 2015-2022 redistributable these are absent and the exe fails to load
# (STATUS_DLL_NOT_FOUND) at first chat. The Windows loader searches the exe's own
# directory first, so we ship the runtime DLLs beside it — consistent with the CRT
# the PyInstaller bundle already ships for the Python side. The spec then bundles
# every *.dll in this bin dir (see backend.spec).
Write-Step "Bundling the MSVC C++ runtime next to llama-server.exe..."
$Sys32 = Join-Path $env:SystemRoot "System32"
$RequiredCrt = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
$OptionalCrt = @("msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll")
foreach ($dll in $RequiredCrt) {
    $src = Join-Path $Sys32 $dll
    if (-not (Test-Path $src)) {
        Write-Fail "Required CRT DLL not found on the build machine: $src"
    }
    Copy-Item -Path $src -Destination $BinDir -Force
    Write-OK "Bundled $dll"
}
foreach ($dll in $OptionalCrt) {
    $src = Join-Path $Sys32 $dll
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $BinDir -Force
        Write-OK "Bundled $dll"
    }
}

# -------- verify output --------
Write-Host ""
Write-Step "Verifying output..."

$ServerExe = Join-Path $BinDir "llama-server.exe"

if (Test-Path $ServerExe) {
    Write-OK "llama.cpp CPU build complete: $ServerExe"
} else {
    Write-Warn "llama-server.exe not at $ServerExe. Installed binaries:"
    Get-ChildItem -Path $BinDir -Recurse -Filter "*.exe" | ForEach-Object { Write-Host ('  ' + $_.FullName) }
    Write-Fail "Expected llama-server.exe missing after install."
}
