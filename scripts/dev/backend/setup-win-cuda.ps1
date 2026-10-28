# Erudi Backend Setup Script - Windows CUDA
# Supports both development and production environments
# Compatible with interactive use and CI/CD pipelines
# Requires: Python 3.12 exactly (pgserver ships cp312 wheels only), and a CUDA
#           toolkit if you intend to compile llama-server for an NVIDIA GPU.

param(
    [string]$InstallType = ""
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Status { Write-Host "[STATUS] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Error-Exit { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }

# Determine if running in CI/CD mode
function Is-CIMode {
    return ($env:CI -eq "true") -or ($env:GITHUB_ACTIONS -eq "true") -or ($null -ne $env:JENKINS_HOME)
}

# Determine current directory and set paths
$currentDir = (Get-Item .).Name
if ($currentDir -eq "backend") {
    Write-Status "Running from erudi\backend\"
    $reqDev = ".\requirements\entrypoints\dev\win-cuda.txt"
    $reqProd = ".\requirements\entrypoints\prod\win-cuda-prod.txt"
    $venvPath = ".\venv"
} else {
    Write-Status "Running from erudi\ (don't forget to cd backend\ after setup)"
    $reqDev = ".\backend\requirements\entrypoints\dev\win-cuda.txt"
    $reqProd = ".\backend\requirements\entrypoints\prod\win-cuda-prod.txt"
    $venvPath = ".\backend\venv"
}

# Check Python version (3.12 exactly)
Write-Status "Checking Python version..."

$pythonCandidates = @("python", "python3", "py")
$pythonCmd = ""
$version = ""

foreach ($cmd in $pythonCandidates) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            break
        }
    } catch {
        continue
    }
}

if ([string]::IsNullOrEmpty($pythonCmd)) {
    Write-Error-Exit "No valid Python executable found in PATH (tried: $($pythonCandidates -join ', '))"
}

if ($version -match "(\d+)\.(\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    # 3.12 EXACTLY, not a floor: pgserver ships cp312 wheels only (no cp313,
    # no sdist), so a newer Python gets past this check and then fails to
    # install the embedded PostgreSQL cluster with an opaque pip error.
    if (($major -ne 3) -or ($minor -ne 12)) {
        Write-Error-Exit "Python 3.12 exactly is required (pgserver publishes cp312 wheels only), found: $version"
    } else {
        Write-Status "Using Python: $pythonCmd ($version)"
    }
} else {
    Write-Error-Exit "Could not determine Python version from output: $version"
}

# Determine environment type (production or development)
$installTypeChoice = "dev"

if (Is-CIMode) {
    Write-Status "CI/CD mode detected"
    # Check for INSTALL_TYPE environment variable in CI
    $installTypeChoice = if ($env:INSTALL_TYPE) { $env:INSTALL_TYPE } else { "prod" }
    Write-Status "Installing $installTypeChoice dependencies (set INSTALL_TYPE=dev/prod to change)"
} elseif ($InstallType -ne "") {
    # Parameter provided
    $installTypeChoice = $InstallType.ToLower()
} else {
    # Interactive mode
    Write-Host ""
    Write-Host "Choose installation type:"
    Write-Host "  [1] Development (includes testing, linting, debugging tools)"
    Write-Host "  [2] Production  (minimal dependencies only)"
    Write-Host ""
    $choice = Read-Host "Enter choice [1/2] (default: 1)"
    if ([string]::IsNullOrEmpty($choice)) { $choice = "1" }
    
    switch ($choice) {
        "1" { $installTypeChoice = "dev" }
        "2" { $installTypeChoice = "prod" }
        default { Write-Error-Exit "Invalid choice. Please enter 1 or 2." }
    }
}

if ($installTypeChoice -eq "dev") {
    $reqFile = $reqDev
    Write-Status "Setting up DEVELOPMENT environment"
} else {
    $reqFile = $reqProd
    Write-Status "Setting up PRODUCTION environment"
}

# Handle existing virtual environment
Write-Status "Checking virtual environment in $venvPath..."

if (Test-Path $venvPath) {
    if (Is-CIMode) {
        # In CI, always recreate
        Write-Status "CI mode: Removing existing virtual environment..."
        Remove-Item -Path $venvPath -Recurse -Force
        Write-Status "Creating fresh virtual environment..."
        & $pythonCmd -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to create virtual environment" }
    } else {
        # Interactive mode: ask user
        $venvConfirmation = Read-Host "Virtual environment exists. Recreate it? [Y/n]"
        if ([string]::IsNullOrEmpty($venvConfirmation)) { $venvConfirmation = "Y" }
        if ($venvConfirmation -match "^[Yy]") {
            Write-Status "Removing existing virtual environment..."
            Remove-Item -Path $venvPath -Recurse -Force
            Write-Status "Creating fresh virtual environment..."
            & $pythonCmd -m venv $venvPath
            if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to create virtual environment" }
        } else {
            Write-Status "Using existing virtual environment"
        }
    }
} else {
    Write-Status "Creating virtual environment..."
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to create virtual environment" }
}

# Activate virtual environment
Write-Status "Activating virtual environment..."
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

# Upgrade pip
Write-Status "Upgrading pip..."
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to upgrade pip" }

# Install requirements
if (Is-CIMode) {
    # In CI, always force reinstall for clean state
    Write-Status "Installing requirements from $reqFile (force reinstall)..."
    & $venvPython -m pip install --no-cache-dir --force-reinstall -r $reqFile
    if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to install requirements" }
} else {
    # Interactive mode: ask about force reinstall
    $forceReinstall = Read-Host "Force reinstall all packages? [Y/n]"
    if ([string]::IsNullOrEmpty($forceReinstall)) { $forceReinstall = "Y" }
    
    if ($forceReinstall -match "^[Yy]") {
        Write-Status "Installing requirements from $reqFile (force reinstall)..."
        & $venvPython -m pip install --no-cache-dir --force-reinstall -r $reqFile
        if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to install requirements" }
    } else {
        Write-Status "Installing requirements from $reqFile..."
        & $venvPython -m pip install --no-cache-dir -r $reqFile
        if ($LASTEXITCODE -ne 0) { Write-Error-Exit "Failed to install requirements" }
    }
}

# Success message
Write-Host ""
Write-Success "Python environment setup complete!"
Write-Host ""
Write-Host "Environment: $($installTypeChoice.ToUpper())"
Write-Host "Python: $version"
Write-Host "Virtual env: $venvPath"
Write-Host ""

# -------- CUDA Toolkit Verification --------
Write-Status "Looking for a CUDA toolkit..."

# Any CUDA 12.x compiles llama-server; the version only decides which GPU
# architectures get native code (12.8+ adds Blackwell / RTX 50). So accept
# whatever is installed, newest first, rather than demanding one version.
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$CudaBin = $null
$Candidates = @($env:CUDA_PATH)
if (Test-Path $CudaRoot) {
    $Candidates += (Get-ChildItem $CudaRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | ForEach-Object { $_.FullName })
}
foreach ($candidate in $Candidates) {
    if ($candidate -and (Test-Path (Join-Path $candidate "bin\nvcc.exe"))) {
        $CudaBin = Join-Path $candidate "bin"
        break
    }
}

if (-not $CudaBin) {
    Write-Host ""
    Write-Host "[WARNING] No CUDA toolkit detected." -ForegroundColor Yellow
    Write-Host "  Install from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "  (12.8 or later matches the released binaries; any 12.x will build.)" -ForegroundColor Yellow
    Write-Host "  You only need it to COMPILE llama-server. Running Erudi needs a driver only." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Success "CUDA toolkit found: $CudaBin"

    # Check if CUDA bin is on system PATH
    $pathEntries = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") -split ";"
    $cudaOnPath = $pathEntries | Where-Object { $_ -eq $CudaBin -or $_ -eq $CudaBin.TrimEnd("\") }

    if (-not $cudaOnPath) {
        Write-Host ""
        Write-Host "[NOTE] CUDA bin directory is NOT on your system PATH." -ForegroundColor Yellow
        Write-Host "  The engine handles this at runtime, but adding it to PATH is recommended." -ForegroundColor Yellow
        Write-Host "  To add permanently, run (as Administrator):" -ForegroundColor Yellow
        Write-Host "    [System.Environment]::SetEnvironmentVariable('PATH', `$env:PATH + ';$CudaBin', 'Machine')" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Success "CUDA bin is on system PATH"
    }
}

# -------- llama.cpp Build Step --------
Write-Status "Checking llama.cpp CUDA build..."

if ($currentDir -eq "backend") {
    $llamaServerPath = ".\artifacts\llama-cpp\cuda\bin\llama-server.exe"
    $buildScript = "..\scripts\dev\backend\build-llamacpp-cuda-win.ps1"
} else {
    $llamaServerPath = ".\backend\artifacts\llama-cpp\cuda\bin\llama-server.exe"
    $buildScript = ".\scripts\dev\backend\build-llamacpp-cuda-win.ps1"
}

if (Test-Path $llamaServerPath) {
    Write-Success "llama-server.exe already built at: $llamaServerPath"
    Write-Host "  To rebuild, run: $buildScript"
} else {
    Write-Host ""
    Write-Host "llama-server.exe not found. GPU inference requires a compiled build." -ForegroundColor Yellow

    if (-not (Is-CIMode)) {
        $buildChoice = Read-Host "Run the llama.cpp CUDA build script now? [Y/n]"
        if ([string]::IsNullOrEmpty($buildChoice)) { $buildChoice = "Y" }

        if ($buildChoice -match "^[Yy]") {
            Write-Status "Launching llama.cpp CUDA build..."
            & $buildScript
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARNING] Build failed. You can retry manually: $buildScript" -ForegroundColor Yellow
            } else {
                Write-Success "llama.cpp CUDA build completed successfully."
            }
        } else {
            Write-Host ""
            Write-Host "Skipped. Run later with: $buildScript" -ForegroundColor Yellow
        }
    } else {
        Write-Status "CI mode: Skipping interactive build. Run $buildScript as a separate step."
    }
}

Write-Host ""
Write-Host "------------------------------------------------------"
Write-Host ""
if ($currentDir -ne "backend") {
    Write-Host "Next steps:"
    Write-Host "  1. cd backend"
    Write-Host '  2. .\venv\Scripts\Activate.ps1'
    Write-Host "  3. uvicorn src.main:app --reload"
} else {
    Write-Host "Next steps:"
    Write-Host '  1. .\venv\Scripts\Activate.ps1'
    Write-Host '  2. uvicorn src.main:app --reload'
}
Write-Host ""
