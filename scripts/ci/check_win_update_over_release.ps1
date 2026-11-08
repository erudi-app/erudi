# check_win_update_over_release.ps1
#
# Install a published Windows CPU release, then run a freshly built installer
# over it, both ways a user does, and fail unless each update lands with the old
# uninstaller succeeding at its first attempt.
#
# Usage (from the repo root, on Windows, as an administrator, with GH_TOKEN set):
#   pwsh scripts/ci/check_win_update_over_release.ps1 -DistDir frontend/dist -BaseTag v1.1.2
#
# Installing over an existing Erudi runs the INSTALLED version's uninstaller
# with --updated. It moves every file of the install into
# %TEMP%\nsXXXX.tmp\old-install\, recreating each directory with CreateDirectory
# and moving each file with MoveFileW, both without the \\?\ prefix: a directory
# of 248 characters or more, or a file of 260 or more, makes it give up (exit
# code 2, old install restored). frontend/assets/installers/installer.nsh
# removes the old backend's dist-info license folders, the deepest files of the
# install, before that uninstaller runs (customInit), and retries a failed
# uninstall once without them (customUnInstallCheck).
#
# Two passes over the base release:
# 1. The installer with `/S` alone: an installer started by hand.
# 2. After removing the result with its own uninstaller and installing the base
#    release again, the installer with `--updated /S`, what electron-updater
#    passes when it installs at quit ("Install now" passes `--updated
#    --force-run`, interactive; --force-run would start the app here).
# Both expect the old uninstaller to start exactly once: a second start means
# its first attempt failed. Built without customInit, both passes still update
# (the recovery does it) and the gate fails on that count.
#
# Counting uninstaller starts: electron-builder copies the old uninstaller to
# $PLUGINSDIR\old-uninstaller.exe and runs that copy once per attempt (in place,
# as "Uninstall Erudi.exe", only when the copy cannot start). The script
# subscribes to Win32_ProcessStartTrace (kernel process-start events, which miss
# no process however short-lived; it needs an administrator, which hosted
# runners are) for the duration of each update and counts those starts.
#
# The base release is pinned (-BaseTag, v1.1.2): it carries the torch license
# tree, and the reproduction margin is zero. The uninstaller's destination is
#   <TEMP>\<plugins dir: nsX + 1 to 4 hex digits + .tmp>\old-install
# followed by the file's path relative to the install directory. The script
# points TEMP and TMP at %LOCALAPPDATA%\Temp, the long form of a user's temp
# directory (on hosted runners they can hold the 8.3 form C:\Users\RUNNER~1\...,
# shorter than what the 11-character runneradmin gets on a real machine). There
# the deepest file of v1.1.2 lands at 48 + 11 + 1 + 200 = 260 characters with
# the shortest plugins directory name. The script checks that before each pass
# and fails if it no longer holds: a run that cannot hit the defect proves
# nothing.
#
# Margins of the new build: outside the license folders the include removes,
# the deepest directory of win-unpacked must stay below 248 characters, and its
# deepest file below 260, under
#   C:\Users\<20-character name>\AppData\Local\Temp\nsa0.tmp\old-install
# so a new deep path in the bundle fails here before it fails an update.
#
# The deepest path of the base release's CUDA installer is logged too, without
# failing on it (that edition is not installed here).
#
# Checks after each update: the installer exits 0 within the timeout (a failed
# update shows a message box even in silent mode, so a failure can surface as a
# timeout), the uninstall entry reports the new version, every file of the
# build's win-unpacked tree is installed and the main binaries match it byte for
# byte, and no old-install directory is left under TEMP.

[CmdletBinding()]
param(
  # electron-builder output: holds exactly one installer .exe and win-unpacked\.
  [Parameter(Mandatory = $true)] [string] $DistDir,
  # The published release installed first. Pinned: see above.
  [string] $BaseTag = "v1.1.2",
  [string] $Repo = $env:GITHUB_REPOSITORY,
  [int] $TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$UninstallRoot = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall"
$LicenseFile = '^\\resources\\backend\\_internal\\[^\\]+\.dist-info\\licenses\\'
$LicenseDir = '^\\resources\\backend\\_internal\\[^\\]+\.dist-info\\licenses(\\|$)'
$DirLimit = 248
$FileLimit = 260

function Fail([string] $Message) {
  Write-Host "::error::$Message"
  exit 1
}

function Invoke-Installer([string] $Path, [string[]] $Arguments) {
  Write-Host "Running: `"$Path`" $($Arguments -join ' ')"
  $started = Get-Date
  $proc = Start-Process -FilePath $Path -ArgumentList $Arguments -PassThru
  $null = $proc.Handle # keeps the handle so ExitCode is readable after exit
  # Poll rather than block in WaitForExit, so the process-start events of the
  # watch below keep reaching this session's queue.
  while (-not $proc.HasExited) {
    if (((Get-Date) - $started).TotalSeconds -gt $TimeoutSeconds) {
      & taskkill.exe /F /T /PID $proc.Id | Out-Host
      Fail "`"$Path`" did not exit within $TimeoutSeconds s. A failed uninstall of the old version blocks on a message box, even in silent mode."
    }
    Start-Sleep -Milliseconds 500
  }
  $proc.WaitForExit()
  Write-Host "  exited $($proc.ExitCode) after $([int]((Get-Date) - $started).TotalSeconds) s"
  return $proc.ExitCode
}

# Every Erudi entry in the per-user uninstall key. The key itself does not
# exist until something installs per user: no key means no entry.
function Get-ErudiUninstallEntries {
  if (-not (Test-Path -LiteralPath $UninstallRoot)) { return @() }
  return @(
    Get-ChildItem -LiteralPath $UninstallRoot |
      ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } |
      Where-Object {
        $_.PSObject.Properties["UninstallString"] -and $_.UninstallString -like "*Uninstall Erudi.exe*"
      }
  )
}

function Get-ErudiUninstallEntry {
  $entries = @(Get-ErudiUninstallEntries)
  if ($entries.Count -ne 1) {
    Fail "Expected one Erudi entry under $UninstallRoot, found $($entries.Count)."
  }
  $entry = $entries[0]
  if ($entry.QuietUninstallString -notmatch '^"([^"]+)"\s*(.*)$') {
    Fail "Unexpected QuietUninstallString: $($entry.QuietUninstallString)"
  }
  return [pscustomobject]@{
    Version       = $entry.DisplayVersion
    InstallDir    = Split-Path -Parent $Matches[1]
    Uninstaller   = $Matches[1]
    UninstallArgs = $Matches[2]
  }
}

# Paths below $Dir, each starting with "\".
function Get-RelativePaths([string] $Dir, [switch] $Directories) {
  $full = (Resolve-Path -LiteralPath $Dir).ProviderPath.TrimEnd("\")
  $items = if ($Directories) {
    Get-ChildItem -LiteralPath $full -Recurse -Force -Directory
  } else {
    Get-ChildItem -LiteralPath $full -Recurse -Force -File
  }
  return @($items | ForEach-Object { $_.FullName.Substring($full.Length) })
}

function Get-Longest([string[]] $Paths) {
  if ($Paths.Count -eq 0) { Fail "No path to measure." }
  return @($Paths | Sort-Object Length -Descending)[0]
}

function Get-OldInstallDirs([string] $TempDir) {
  return @(
    Get-ChildItem -LiteralPath $TempDir -Directory -Filter "ns*.tmp" -Force |
      Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "old-install") } |
      ForEach-Object { $_.FullName }
  )
}

function Install-Base {
  $code = Invoke-Installer $baseInstaller @("/S")
  if ($code -ne 0) { Fail "Installing $($asset.name) exited $code." }
  $old = Get-ErudiUninstallEntry
  Write-Host "Installed Erudi $($old.Version) ($BaseTag, $($asset.name)) in $($old.InstallDir)"
  if ("v$($old.Version)" -ne $BaseTag) {
    Fail "The base release is $BaseTag but the installed version is $($old.Version)."
  }
  return $old
}

# Fails unless the old install's deepest file lands at 260 characters or more
# under old-install\ with the shortest plugins directory name.
function Assert-CanReproduce($Old) {
  $deepest = Get-Longest @(Get-RelativePaths $Old.InstallDir)
  $prefix = (Join-Path $gateTemp "nsa0.tmp\old-install").Length
  Write-Host "Deepest file of the old install: $deepest"
  Write-Host "  in place: $($Old.InstallDir.Length + $deepest.Length) characters; under old-install: at least $($prefix + $deepest.Length)"
  if ($prefix + $deepest.Length -lt $FileLimit) {
    Fail "The deepest file would land at $($prefix + $deepest.Length) characters, under the $FileLimit limit: this run cannot reproduce the update failure."
  }
}

# Runs the new installer over $Old with $Arguments, counting the old
# uninstaller's starts, checks the result, and returns the number of starts.
function Update-Over($Old, [string[]] $Arguments) {
  $leftoversBefore = @(Get-OldInstallDirs $gateTemp)
  $source = "erudi-uninstaller-starts"
  try {
    Register-CimIndicationEvent -ClassName Win32_ProcessStartTrace -SourceIdentifier $source | Out-Null
  } catch {
    Fail "Cannot watch process starts (Win32_ProcessStartTrace needs an administrator): $_"
  }
  try {
    $code = Invoke-Installer $newInstaller $Arguments
    Start-Sleep -Seconds 2 # lets the last start events arrive
    # The kernel trace truncates long image names, hence the prefixes.
    # Get-Event errors when no event arrived at all; that is a count of 0.
    $starts = @(
      Get-Event -SourceIdentifier $source -ErrorAction SilentlyContinue |
        Where-Object {
          $name = [string]$_.SourceEventArgs.NewEvent.ProcessName
          $name -like "old-uninst*" -or $name -like "Uninstall Er*"
        }
    ).Count
  } finally {
    Unregister-Event -SourceIdentifier $source
    Get-Event -SourceIdentifier $source -ErrorAction SilentlyContinue | Remove-Event # none left is fine
  }
  Write-Host "  the old uninstaller started $starts time(s)"
  if ($code -ne 0) { Fail "The update over Erudi $($Old.Version) exited $code." }

  $new = Get-ErudiUninstallEntry
  Write-Host "After the update: Erudi $($new.Version) in $($new.InstallDir)"
  if ($new.Version -ne $newVersion) { Fail "The uninstall entry reports $($new.Version), expected $newVersion." }
  if ($new.InstallDir -ne $Old.InstallDir) { Fail "The install moved from $($Old.InstallDir) to $($new.InstallDir)." }

  $installed = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  foreach ($f in @(Get-RelativePaths $new.InstallDir)) { [void]$installed.Add($f) }
  $missing = @($newFiles | Where-Object { -not $installed.Contains($_) })
  if ($missing.Count -gt 0) {
    Fail "$($missing.Count) file(s) of the build are not installed, e.g. $($missing | Select-Object -First 5)"
  }
  foreach ($rel in @("\Erudi.exe", "\resources\app.asar", "\resources\backend\backend.exe")) {
    $a = (Get-FileHash -LiteralPath (Join-Path $unpacked $rel)).Hash
    $b = (Get-FileHash -LiteralPath (Join-Path $new.InstallDir $rel)).Hash
    if ($a -ne $b) { Fail "$rel differs from the build's copy after the update." }
  }
  Write-Host "All $($newFiles.Count) files of the build are installed; Erudi.exe, app.asar and backend.exe match it."

  $leftovers = @(Get-OldInstallDirs $gateTemp | Where-Object { $leftoversBefore -notcontains $_ })
  if ($leftovers.Count -gt 0) { Fail "old-install left under TEMP: $($leftovers -join ', ')" }
  Write-Host "No old-install directory left under $gateTemp."
  return $starts
}

function Assert-SingleStart([int] $Starts, [string] $Pass) {
  if ($Starts -ne 1) {
    Fail "$Pass`: the old uninstaller started $Starts time(s), expected once. Its first attempt failed: the license folders were still in the old install when it ran (the update above completed through the recovery, if it did)."
  }
  Write-Host "$Pass`: the old uninstaller succeeded at its first attempt."
}

# Logs the deepest entry of the CUDA edition's app archive. Informational: any
# failure here is a warning, since that edition is not what this gate installs.
function Write-CudaDepth([string] $Version) {
  $name = "Erudi-Setup-$Version-cuda.exe"
  $cuda = @($release.assets | Where-Object { $_.name -eq $name })
  if ($cuda.Count -ne 1) {
    Write-Host "::warning::$BaseTag has no $name; CUDA depth not logged."
    return
  }
  $sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source # absent: fall back below
  if (-not $sevenZip) { $sevenZip = Join-Path $env:ProgramFiles "7-Zip\7z.exe" }
  if (-not (Test-Path -LiteralPath $sevenZip)) {
    Write-Host "::warning::7z not found; CUDA depth not logged."
    return
  }
  $dir = Join-Path $downloadDir "cuda"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  gh release download $BaseTag --repo $Repo --pattern $name --dir $dir --clobber
  if ($LASTEXITCODE -ne 0) {
    Write-Host "::warning::Could not download $name (exit $LASTEXITCODE); CUDA depth not logged."
    return
  }
  & $sevenZip e -y "-o$dir" (Join-Path $dir $name) "app-64.7z" -r | Out-Null
  $inner = Join-Path $dir "app-64.7z"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $inner)) {
    Write-Host "::warning::Could not extract app-64.7z from $name; CUDA depth not logged."
    return
  }
  $listing = @(& $sevenZip l -slt $inner)
  if ($LASTEXITCODE -ne 0) {
    Write-Host "::warning::Could not list $inner; CUDA depth not logged."
    return
  }
  # Entries follow the "----------" line; before it, "Path =" names the archive.
  $separator = [Array]::IndexOf($listing, "----------")
  $paths = @(
    $listing[($separator + 1)..($listing.Count - 1)] |
      Where-Object { $_ -like "Path = *" } |
      ForEach-Object { "\" + $_.Substring(7).Replace("/", "\") }
  )
  if ($paths.Count -eq 0) {
    Write-Host "::warning::$inner lists no entry; CUDA depth not logged."
    return
  }
  $deepest = Get-Longest $paths
  $prefix = (Join-Path $gateTemp "nsa0.tmp\old-install").Length
  Write-Host "CUDA edition ($name): deepest entry $deepest ($($deepest.Length) characters below the install directory)"
  Write-Host "  under old-install: $($prefix + $deepest.Length) for $env:USERNAME, $($prefix20 + $deepest.Length) for a 20-character user name (shortest plugins directory name)"
  Remove-Item -LiteralPath $dir -Recurse -Force
}

if (-not $Repo) { Fail "No repository: pass -Repo or set GITHUB_REPOSITORY." }
if (-not $env:GH_TOKEN) { Fail "GH_TOKEN is not set; gh needs it to read the releases." }

# --- The freshly built installer ----------------------------------------------
$dist = (Resolve-Path -LiteralPath $DistDir).ProviderPath
$newInstallers = @(Get-ChildItem -LiteralPath $dist -Filter "*.exe" -File)
if ($newInstallers.Count -ne 1) {
  Fail "Expected one installer in $dist, found: $(($newInstallers | ForEach-Object Name) -join ', ')"
}
$newInstaller = $newInstallers[0].FullName
$unpacked = Join-Path $dist "win-unpacked"
if (-not (Test-Path -LiteralPath (Join-Path $unpacked "Erudi.exe"))) {
  Fail "No win-unpacked\Erudi.exe in $dist."
}
$newVersion = (Get-Content -Raw -LiteralPath (Join-Path $dist "..\package.json") | ConvertFrom-Json).version
$newFiles = @(Get-RelativePaths $unpacked)
$newDirs = @(Get-RelativePaths $unpacked -Directories)

# --- Margins of the new build, outside the removed license folders ------------
$prefix20 = ("C:\Users\" + ("u" * 20) + "\AppData\Local\Temp\nsa0.tmp\old-install").Length
$deepestFile = Get-Longest @($newFiles | Where-Object { $_ -notmatch $LicenseFile })
$deepestDir = Get-Longest @($newDirs | Where-Object { $_ -notmatch $LicenseDir })
$fileAt = $prefix20 + $deepestFile.Length
$dirAt = $prefix20 + $deepestDir.Length
Write-Host "Margins of the new build for a 20-character user name, outside the license folders:"
Write-Host "  deepest directory $deepestDir -> $dirAt characters, $($DirLimit - 1 - $dirAt) to spare below $DirLimit"
Write-Host "  deepest file $deepestFile -> $fileAt characters, $($FileLimit - 1 - $fileAt) to spare below $FileLimit"
if ($dirAt -ge $DirLimit) { Fail "The build's deepest directory would reach $dirAt characters under old-install (limit $DirLimit): $deepestDir" }
if ($fileAt -ge $FileLimit) { Fail "The build's deepest file would reach $fileAt characters under old-install (limit $FileLimit): $deepestFile" }

# --- The base release's Windows CPU installer ----------------------------------
# Any gh failure is fatal: a gate that skips on a network error proves nothing.
Write-Host "Base release: $BaseTag ($Repo)"
$releaseJson = gh release view $BaseTag --repo $Repo --json tagName,assets
if ($LASTEXITCODE -ne 0) { Fail "gh release view $BaseTag failed for $Repo (exit $LASTEXITCODE)." }
$release = ($releaseJson -join "`n") | ConvertFrom-Json
$baseVersion = $BaseTag.TrimStart("v")
$assets = @($release.assets | Where-Object { $_.name -eq "Erudi-Setup-$baseVersion.exe" })
if ($assets.Count -ne 1) {
  Fail "Expected Erudi-Setup-$baseVersion.exe in $BaseTag, found: $(($release.assets | ForEach-Object name) -join ', ')"
}
$asset = $assets[0]

$downloadDir = Join-Path ([IO.Path]::GetTempPath()) "erudi-update-gate-$PID"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$baseInstaller = Join-Path $downloadDir $asset.name
for ($attempt = 1; $attempt -le 3; $attempt++) {
  gh release download $BaseTag --repo $Repo --pattern $asset.name --dir $downloadDir --clobber
  if ($LASTEXITCODE -eq 0) { break }
  Write-Host "::warning::gh release download failed (attempt $attempt of 3, exit $LASTEXITCODE)."
  if ($attempt -eq 3) { Fail "Could not download $($asset.name) from $BaseTag." }
  Start-Sleep -Seconds 10
}
$size = (Get-Item -LiteralPath $baseInstaller).Length
if ($size -ne $asset.size) {
  Fail "$($asset.name) is $size bytes, the release lists $($asset.size)."
}
if (@(Get-ErudiUninstallEntries).Count -gt 0) { Fail "Erudi is already installed on this machine; the gate needs a clean one." }
if ($env:GITHUB_STEP_SUMMARY) {
  "Windows update gate: over Erudi $BaseTag ($($asset.name)), to $newVersion." |
    Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
}

$gateTemp = Join-Path $env:LOCALAPPDATA "Temp"
Write-Host "TEMP was $env:TEMP; the installers run with TEMP and TMP = $gateTemp"
$env:TEMP = $gateTemp
$env:TMP = $gateTemp

try {
  Write-CudaDepth $baseVersion
} catch {
  Write-Host "::warning::CUDA depth not logged: $_" # informational only
}

# --- Pass 1: installer started by hand ----------------------------------------
Write-Host "`n== Pass 1: installer started by hand over Erudi $BaseTag (/S) =="
$old = Install-Base
Assert-CanReproduce $old
$starts = Update-Over $old @("/S")
Assert-SingleStart $starts "Pass 1"

# --- Remove the new install with its own uninstaller --------------------------
Write-Host "`n== Removing the new install =="
$current = Get-ErudiUninstallEntry
# An NSIS uninstaller copies itself to TEMP, starts the copy and exits, so the
# wait is on its effects: the uninstall entry and the install directory go.
$code = Invoke-Installer $current.Uninstaller @($current.UninstallArgs -split '\s+' | Where-Object { $_ })
if ($code -ne 0) { Fail "The uninstaller exited $code." }
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while (@(Get-ErudiUninstallEntries).Count -gt 0 -or (Test-Path -LiteralPath $current.InstallDir)) {
  if ((Get-Date) -gt $deadline) {
    Fail "Uninstalling left the uninstall entry or $($current.InstallDir) after $TimeoutSeconds s."
  }
  Start-Sleep -Seconds 1
}
Write-Host "Uninstalled: no uninstall entry, no $($current.InstallDir)."

# --- Pass 2: update the way electron-updater starts it --------------------------
Write-Host "`n== Pass 2: update over Erudi $BaseTag the way electron-updater starts it (--updated /S) =="
$old = Install-Base
Assert-CanReproduce $old
$starts = Update-Over $old @("--updated", "/S")
Assert-SingleStart $starts "Pass 2"
