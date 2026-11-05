# build-win-cuda.ps1
#
# Full build pipeline for Erudi on Windows with CUDA.
#
# Steps:
#   1. Verify prerequisites (Python venv, the CUDA and the CPU llama-server
#      with their runtime DLLs, Node/npm)
#   2. Install PyInstaller into the backend venv (if missing)
#   3. Generate icon.ico from icon.png (if missing)
#   4. Run PyInstaller with backend.spec  → backend/dist/backend/
#   5. Copy backend bundle into frontend/  → frontend/backend/
#   6. Run `npm run dist:win` in frontend/ → electron-builder NSIS installer
#
# Usage (from repo root):
#   .\scripts\build\build-win-cuda.ps1
#
# Output:
#   frontend\dist\Erudi Setup <version>.exe (the version in frontend\package.json)

$ErrorActionPreference = "Stop"

function Write-Step { Write-Host "`n[build]   $args" -ForegroundColor Cyan }
function Write-OK   { Write-Host "[ok]      $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[warning] $args" -ForegroundColor Yellow }
function Write-Fail { Write-Host "[error]   $args" -ForegroundColor Red; exit 1 }

# ── Path resolution ────────────────────────────────────────────────────────────
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$BackendRoot  = Join-Path $RepoRoot "backend"
$FrontendRoot = Join-Path $RepoRoot "frontend"
$VenvPython   = Join-Path $BackendRoot "venv\Scripts\python.exe"
$VenvPip      = Join-Path $BackendRoot "venv\Scripts\pip.exe"
$BackendSpec  = Join-Path $BackendRoot "backend.spec"
$BackendDist  = Join-Path $BackendRoot "dist\backend"
$FrontendBackend = Join-Path $FrontendRoot "backend"
$IconPng      = Join-Path $FrontendRoot "assets\icons\icon.png"
$IconIco      = Join-Path $FrontendRoot "assets\icons\icon.ico"
$LlamaServer  = Join-Path $BackendRoot "artifacts\llama-cpp\cuda\bin\llama-server.exe"
# The CUDA installer ships the CPU build too: CPU_Engine ("Processor only")
# runs it, with no NVIDIA library loaded (backend.spec requires both).
$LlamaServerCpu = Join-Path $BackendRoot "artifacts\llama-cpp\cpu\bin\llama-server.exe"

Write-Step "Erudi Windows CUDA build"
Write-Host "  Repo root : $RepoRoot"
Write-Host "  Backend   : $BackendRoot"
Write-Host "  Frontend  : $FrontendRoot"

# ── Prerequisites ──────────────────────────────────────────────────────────────
Write-Step "Checking prerequisites..."

if (-not (Test-Path $VenvPython)) {
    Write-Fail "Backend venv not found at $VenvPython.`nRun: .\scripts\dev\backend\setup-win-cuda.ps1"
}
Write-OK "Backend venv found"

if (-not (Test-Path $LlamaServer)) {
    Write-Fail "llama-server.exe not found at $LlamaServer.`nRun: .\scripts\dev\backend\build-llamacpp-cuda-win.ps1"
}
Write-OK "llama-server.exe found"

if (-not (Test-Path $LlamaServerCpu)) {
    Write-Fail "CPU llama-server.exe not found at $LlamaServerCpu.`nThe CUDA installer ships it for the processor engine.`nRun: .\scripts\dev\backend\build-llamacpp-cpu-win.ps1"
}
Write-OK "CPU llama-server.exe found"

# The runtime DLLs llama-server.exe links against have to sit beside it, because
# the Windows loader searches the exe's own directory first and an end user has
# the NVIDIA display driver but not the multi-GB CUDA Toolkit (#356, and #144
# for the CRT). build-llamacpp-cuda-win.ps1 copies them, but THIS script is the
# one that runs every release while that one runs once in a blue moon -- so an
# artifacts/ directory compiled before #356 landed still passes the existence
# check above and ships a bundle whose GPU path is dead on arrival
# (STATUS_DLL_NOT_FOUND, silently, with no error until the first chat turn).
# Checking here is what turns that into a build failure instead of a bad release.
$LlamaBinDir = Split-Path $LlamaServer -Parent
$RequiredRuntimeDlls = @(
    "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"
)
$MissingRuntimeDlls = @($RequiredRuntimeDlls | Where-Object {
    -not (Test-Path (Join-Path $LlamaBinDir $_))
})
if ($MissingRuntimeDlls.Count -gt 0) {
    Write-Fail ("Runtime DLLs missing next to llama-server.exe: {0}`n" -f ($MissingRuntimeDlls -join ", ") +
                "$LlamaBinDir was built before these were bundled, so GPU inference would be " +
                "dead on any machine without the CUDA Toolkit.`n" +
                "Re-run: .\scripts\dev\backend\build-llamacpp-cuda-win.ps1")
}
Write-OK "llama-server runtime DLLs bundled ($($RequiredRuntimeDlls.Count) checked)"

# The CPU build carries the MSVC C++ runtime beside it (#144) and never the
# CUDA runtime: a cudart in cpu\bin means the CUDA build was copied there.
$LlamaCpuBinDir = Split-Path $LlamaServerCpu -Parent
$RequiredCpuDlls = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
$MissingCpuDlls = @($RequiredCpuDlls | Where-Object {
    -not (Test-Path (Join-Path $LlamaCpuBinDir $_))
})
if ($MissingCpuDlls.Count -gt 0) {
    Write-Fail ("Runtime DLLs missing next to the CPU llama-server.exe: {0}`n" -f ($MissingCpuDlls -join ", ") +
                "It would not start on a machine without the VC++ redistributable.`n" +
                "Re-run: .\scripts\dev\backend\build-llamacpp-cpu-win.ps1")
}
if (Test-Path (Join-Path $LlamaCpuBinDir "cudart64_12.dll")) {
    Write-Fail ("cudart64_12.dll found in $LlamaCpuBinDir`: that folder holds a CUDA build.`n" +
                "Re-run: .\scripts\dev\backend\build-llamacpp-cpu-win.ps1")
}
Write-OK "CPU llama-server runtime DLLs bundled ($($RequiredCpuDlls.Count) checked)"

if (-not (Test-Path $BackendSpec)) {
    Write-Fail "backend.spec not found at $BackendSpec."
}
Write-OK "backend.spec found"

$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd) { Write-Fail "npm not found in PATH. Install Node.js 22 (22.18 or later)." }
Write-OK "npm found: $($npmCmd.Source)"

# ── PyInstaller ────────────────────────────────────────────────────────────────
Write-Step "Checking PyInstaller..."
$pyiVersion = $null
try {
    $pyiVersion = & $VenvPython -m PyInstaller --version 2>$null
} catch { }
if (-not $pyiVersion) {
    Write-Step "PyInstaller not installed. Installing..."
    & $VenvPip install pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install PyInstaller." }
    $pyiVersion = & $VenvPython -m PyInstaller --version 2>$null
}
Write-OK "PyInstaller $pyiVersion"

# ── icon.ico ───────────────────────────────────────────────────────────────────
Write-Step "Checking icon.ico..."
if (-not (Test-Path $IconIco)) {
    if (Test-Path $IconPng) {
        Write-Step "Generating icon.ico from icon.png using Pillow..."
        $iconScript = @"
from PIL import Image
img = Image.open(r'$IconPng').convert('RGBA')
sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]
img.save(r'$IconIco', format='ICO', sizes=sizes)
print('icon.ico created')
"@
        $iconScript | & $VenvPython
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Could not generate icon.ico (Pillow may not be installed). Build will use default icon."
        } else {
            Write-OK "icon.ico created at $IconIco"
        }
    } else {
        Write-Warn "icon.png not found. Build will use default icon."
    }
} else {
    Write-OK "icon.ico already exists"
}

# ── Build backend with PyInstaller ─────────────────────────────────────────────
Write-Step "Building backend with PyInstaller (this takes 5-15 minutes)..."
Push-Location $BackendRoot
try {
    & $VenvPython -m PyInstaller backend.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { Write-Fail "PyInstaller build failed." }
} finally {
    Pop-Location
}

if (-not (Test-Path (Join-Path $BackendDist "backend.exe"))) {
    Write-Fail "backend.exe not found after PyInstaller build. Check output above."
}
Write-OK "PyInstaller build complete: $BackendDist\backend.exe"

# ── Copy backend bundle into frontend ─────────────────────────────────────────
Write-Step "Copying backend bundle to frontend..."
if (Test-Path $FrontendBackend) {
    Write-Step "Removing old frontend/backend/..."
    Remove-Item -Path $FrontendBackend -Recurse -Force
}
Copy-Item -Path $BackendDist -Destination $FrontendBackend -Recurse
Write-OK "Backend copied to $FrontendBackend"

# ── Install frontend dependencies if needed ────────────────────────────────────
Write-Step "Checking frontend node_modules..."
if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    Write-Step "Installing frontend dependencies..."
    Push-Location $FrontendRoot
    try {
        cmd /c "npm install"
        if ($LASTEXITCODE -ne 0) { Write-Fail "npm install failed." }
    } finally {
        Pop-Location
    }
} else {
    Write-OK "node_modules already present"
}

# ── Clean old out/ to prevent stale resources from previous builds ─────────────
$OutDir = Join-Path $FrontendRoot "out"
if (Test-Path $OutDir) {
    Write-Step "Removing stale out/ directory..."
    Remove-Item -Path $OutDir -Recurse -Force
    Write-OK "Cleaned out/"
}

# ── Build Electron app + NSIS installer ───────────────────────────────────────
Write-Step "Building Electron app and NSIS installer (electron-builder)..."
Push-Location $FrontendRoot
try {
    cmd /c "npm run dist:win"
    if ($LASTEXITCODE -ne 0) { Write-Fail "dist:win failed." }
} finally {
    Pop-Location
}

# ── Report output ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-OK "Build complete!"
Write-Host ""
Write-Host "Installer output:"

$installerOut = Join-Path $FrontendRoot "dist"
# dist\ keeps the installers of earlier builds: the newest is this build's.
$installer = $null
if (Test-Path $installerOut) {
    $installer = Get-ChildItem -Path $installerOut -Filter "*.exe" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if ($installer) {
    Write-Host "  $($installer.FullName)" -ForegroundColor White
} else {
    Write-Warn "No installer found in $installerOut. Check build output above."
}
Write-Host "============================================================" -ForegroundColor Green
if ($installer) {
    Write-Host ""
    Write-Host "To install:" -ForegroundColor Cyan
    Write-Host "  Run '$($installer.Name)' - installs to the user's AppData by default."
    Write-Host ""
}
