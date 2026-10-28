# build-llamacpp-cuda-win.ps1
#
# Goal:
# - Build llama.cpp for Windows with the NVIDIA CUDA GPU backend.
# - Enables GPU-accelerated inference via CUDA_Engine.
# - CPU fallback layers remain active for models larger than VRAM.
#
# Prerequisites (must be installed before running this script):
# - Visual Studio 2019 or 2022 with "Desktop development with C++" workload
# - An NVIDIA CUDA Toolkit, any 12.x (sets CUDA_PATH automatically). 12.8+ also
#   emits native code for Blackwell / RTX 50; older toolkits leave those cards
#   to the driver PTX JIT.
# - Python 3.12 and the Erudi venv already created via setup-win-cuda.ps1
# - Git (for the llama-cpp submodule, if not already populated)
#
# Usage (run from erudi\ or erudi\backend\):
#   .\scripts\dev\backend\build-llamacpp-cuda-win.ps1
#
# Visual Studio is found through vswhere (under ${env:ProgramFiles(x86)}, else
# on PATH). Set ERUDI_VCVARSALL to the full path of vcvarsall.bat to skip
# vswhere, e.g. when the toolchain lives on another drive:
#   $env:ERUDI_VCVARSALL = "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
#
# Output:
#   backend\artifacts\llama-cpp\cuda\bin\llama-server.exe

$ErrorActionPreference = "Stop"

# -------- helpers --------
function Write-Step  { Write-Host "[build]   $args" -ForegroundColor Cyan }
function Write-OK    { Write-Host "[ok]      $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[warning] $args" -ForegroundColor Yellow }
function Write-Fail  { Write-Host "[error]   $args" -ForegroundColor Red; exit 1 }

# -------- path resolution --------
# Support running from either erudi\ or erudi\backend\
$currentDir = (Get-Item .).Name
if ($currentDir -eq "backend") {
    $BackendRoot = "."
} elseif (Test-Path "backend") {
    $BackendRoot = "backend"
} else {
    Write-Fail "Run this script from the erudi\ or erudi\backend\ directory."
}

$SrcDir     = Join-Path $BackendRoot "forks\llama-cpp"
$BuildDir   = Join-Path $SrcDir      "build-cuda"
$InstallDir = Join-Path $BackendRoot "artifacts\llama-cpp\cuda"
$BinDir     = Join-Path $InstallDir  "bin"
$VenvPip    = Join-Path $BackendRoot "venv\Scripts\pip.exe"
$VenvPython = Join-Path $BackendRoot "venv\Scripts\python.exe"
$VenvCmake  = Join-Path $BackendRoot "venv\Scripts\cmake.exe"

# Toolchain: prefer the dev venv, fall back to PATH (CI installs into the system
# Python, there is no backend\venv). Each var ends up as an existing path or $null.
if (-not (Test-Path $VenvPip))    { $VenvPip    = (Get-Command pip    -ErrorAction SilentlyContinue).Source }
if (-not (Test-Path $VenvPython)) { $VenvPython = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not (Test-Path $VenvCmake))  { $VenvCmake  = (Get-Command cmake  -ErrorAction SilentlyContinue).Source }

Write-Step "Paths resolved:"
Write-Host "  Source  : $SrcDir"
Write-Host "  Build   : $BuildDir"
Write-Host "  Install : $InstallDir"

# -------- sanity checks --------
Write-Step "Checking prerequisites..."

# llama-cpp submodule present
if (-not (Test-Path (Join-Path $SrcDir "CMakeLists.txt"))) {
    Write-Fail "llama-cpp source not found at $SrcDir. Run: git submodule update --init --recursive"
}

# Python (venv or PATH)
if (-not $VenvPython) {
    Write-Fail "Python not found (no backend\venv and none on PATH). Run setup-win-cuda.ps1 first, or install Python."
}

# -------- CUDA toolkit detection --------
# No version is pinned here on purpose. Since torch-CUDA was dropped (#98) the
# toolkit only compiles llama-server, and any CUDA 12.x can do that. What the
# toolkit version DOES decide is which GPU architectures nvcc can emit native
# code for -- 12.8 is the first that knows Blackwell (sm_120), which is why the
# release installs 12.8 (see .github/workflows/release.yml). Build with an older
# 12.x and you get a working binary that reaches Blackwell only through the
# driver's PTX JIT.
Write-Step "Detecting CUDA toolkit..."

$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$CudaPath = $null
$Candidates = @($env:CUDA_PATH)
if (Test-Path $CudaRoot) {
    # Newest install first, so a machine with several toolkits builds with the
    # one that supports the most architectures.
    $Candidates += (Get-ChildItem $CudaRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | ForEach-Object { $_.FullName })
}
foreach ($candidate in $Candidates) {
    if ($candidate -and (Test-Path (Join-Path $candidate "bin\nvcc.exe"))) {
        $CudaPath = $candidate
        break
    }
}

if (-not $CudaPath) {
    Write-Fail (
        "No CUDA toolkit found.`n" +
        "Install one from https://developer.nvidia.com/cuda-downloads (12.8 or later " +
        "to match the released binaries; any 12.x will build).`n" +
        "Expected nvcc.exe under: $CudaRoot\vXX.Y\bin\nvcc.exe`n" +
        "Or set CUDA_PATH to your toolkit root."
    )
}

$Nvcc = Join-Path $CudaPath "bin\nvcc.exe"
$NvccVersion = (& $Nvcc --version 2>&1 | Select-String "release") -replace ".*release ([0-9.]+).*", 'CUDA $1'
Write-OK "CUDA toolkit : $CudaPath"
Write-OK "nvcc         : $NvccVersion"

# -------- Visual Studio detection + MSVC environment bootstrap --------
# We use -G Ninja instead of the Visual Studio generator because the VS generator
# requires the CUDA VS toolset integration (.props files) which is only present when
# the full VS IDE is installed. Ninja + vcvarsall works with Build Tools too.

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

# -------- cmake detection / install --------
Write-Step "Checking cmake..."

if (-not $VenvCmake) {
    Write-Step "cmake not found. Installing via pip..."
    if (-not $VenvPip) { Write-Fail "Neither cmake nor pip found on PATH." }
    & $VenvPip install --upgrade cmake --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install cmake." }
    $VenvCmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
}

if (-not $VenvCmake) { Write-Fail "cmake still not found after pip install." }

$CmakeVersion = (& $VenvCmake --version 2>&1 | Select-Object -First 1)
Write-OK "cmake: $CmakeVersion"

# -------- optional clean --------
$doClean = $false
if ((Test-Path $BuildDir) -or (Test-Path $InstallDir)) {
    Write-Host ""
    Write-Host "Existing build output found:"
    if (Test-Path $BuildDir)   { Write-Host "  - $BuildDir" }
    if (Test-Path $InstallDir) { Write-Host "  - $InstallDir" }
    $confirm = Read-Host "Delete before rebuild? [Y/n]"
    if ([string]::IsNullOrEmpty($confirm)) { $confirm = "Y" }
    if ($confirm -match "^[Yy]") { $doClean = $true }
} else {
    Write-Step "No previous build found."
}

if ($doClean) {
    Write-Step "Removing old build output..."
    if (Test-Path $BuildDir)   { Remove-Item -Path $BuildDir   -Recurse -Force }
    if (Test-Path $InstallDir) { Remove-Item -Path $InstallDir -Recurse -Force }
    Write-OK "Cleaned."
}

New-Item -ItemType Directory -Force -Path $BuildDir   | Out-Null
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# -------- cmake configure --------
Write-Host ""
Write-Step "Configuring llama.cpp with CUDA..."

# Suffixes matter and a bare number is NOT what we want: bare "86" emits both
# PTX and SASS, so the old bare list produced fourteen code objects where seven
# do the job. This mirrors upstream's own release default -- `-virtual` == PTX
# only, JIT-compiled by the driver on first run; `-real` == native SASS.
# Forward compatibility to any architecture newer than what we list comes from
# the 80-virtual PTX, exactly as upstream intends.
$CudaArchs = "50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real"
# sm_120 (Blackwell / RTX 50) only exists from CUDA 12.8: asking an older nvcc
# for it fails the build outright. Add it when the toolkit can emit it, so those
# cards get native code instead of a JIT pass, and stay buildable on 12.x below.
$CudaVersion = $null
if ($NvccVersion -match '([0-9]+)\.([0-9]+)') {
    $CudaVersion = [version]"$($Matches[1]).$($Matches[2])"
}
if ($CudaVersion -and $CudaVersion -ge [version]"12.8") {
    $CudaArchs += ";120-real"
    Write-OK "CUDA  can emit native Blackwell (sm_120) code - adding it"
} else {
    Write-Host "  CUDA $CudaVersion is below 12.8: Blackwell will JIT from PTX instead of running native code."
}

$CmakeArgs = @(
    "-G", "Ninja",                       # Ninja avoids CUDA VS toolset registration requirement
    "-S", $SrcDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",          # Static linkage avoids DLL placement issues on Windows
    "-DCMAKE_INSTALL_PREFIX=$InstallDir",
    "-DCMAKE_CUDA_COMPILER=$Nvcc",
    "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    "-DCMAKE_CXX_FLAGS=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",   # MSVC STL 19.44 requires CUDA 12.4+; bypass its static_assert
    "-DLLAMA_CURL=OFF",                 # No libcurl dependency needed for server-only use
    # ---- backend flags ----
    "-DGGML_CUDA=ON",                   # Enable NVIDIA CUDA backend
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs",
    "-DGGML_CPU=ON",                    # Keep CPU layers for overflow beyond VRAM
    "-DGGML_NATIVE=OFF",                # OFF for portable binary across same-gen GPUs
    "-DGGML_METAL=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_HIP=OFF",
    "-DGGML_SYCL=OFF",
    "-DGGML_RPC=OFF",
    "-DGGML_WEBGPU=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_ACCELERATE=OFF",
    "-DGGML_BLAS=OFF"
)

Write-Host "  cmake $($CmakeArgs -join ' ')"
Write-Host ""

& $VenvCmake @CmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Fail "cmake configuration failed. Check output above." }

# -------- cmake build --------
Write-Host ""
Write-Step "Compiling (this takes 5-15 minutes)..."

$CpuCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
& $VenvCmake --build $BuildDir --config Release --parallel $CpuCount
if ($LASTEXITCODE -ne 0) { Write-Fail "Build failed. Check output above." }

# -------- cmake install --------
Write-Host ""
Write-Step "Installing to $InstallDir..."

& $VenvCmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Fail "Install step failed." }

# -------- bundle the MSVC C++ runtime + CUDA runtime next to llama-server.exe --------
# Same rationale as the CPU build's #144 fix: the Windows loader searches the exe's
# own directory first, so shipping these here means the packaged app works on a
# clean machine that has the NVIDIA display driver but NOT the multi-GB CUDA
# Toolkit (true of virtually every end user — CUDA_PATH above only exists because
# THIS build machine installed the full Toolkit to compile llama.cpp). Without the
# CUDA DLLs specifically, llama-server.exe fails to launch at all on a real
# end-user install (STATUS_DLL_NOT_FOUND) and GPU inference is silently dead.
Write-Step "Bundling the MSVC C++ runtime and CUDA runtime next to llama-server.exe..."
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

$CudaBinDir = Join-Path $CudaPath "bin"
$RequiredCudaDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
foreach ($dll in $RequiredCudaDlls) {
    $src = Join-Path $CudaBinDir $dll
    if (-not (Test-Path $src)) {
        Write-Fail "Required CUDA runtime DLL not found at $src (expected next to nvcc.exe)."
    }
    Copy-Item -Path $src -Destination $BinDir -Force
    Write-OK "Bundled $dll"
}

# -------- verify output --------
Write-Host ""
Write-Step "Verifying output..."

$ServerExe = Join-Path $BinDir "llama-server.exe"

$allOk = $true
if (Test-Path $ServerExe) {
    Write-OK "Found: $ServerExe"
} else {
    Write-Warn "Missing: $ServerExe"
    $allOk = $false
}

Write-Host ""
if ($allOk) {
    Write-OK "llama.cpp CUDA build complete."
    Write-Host ""
    Write-Host "Artifacts installed to: $InstallDir"
    Write-Host ""
    Write-Host "Test with:"
    Write-Host "  $ServerExe --version"
    Write-Host "  $QuantizeExe --help"
} else {
    Write-Warn "Build completed but some expected binaries are missing."
    Write-Warn "Check $BinDir for what was installed."
    Write-Warn "The server binary may be under a sub-path like $BinDir\Release\ if cmake used the Release config subfolder."
    Write-Host ""
    Write-Host "All installed files:"
    Get-ChildItem -Path $BinDir -Recurse -Filter "*.exe" | ForEach-Object { Write-Host ('  ' + $_.FullName) }
}
