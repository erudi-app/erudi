# check_win_update_over_release.ps1
#
# Install the latest published Windows CPU release, then run a freshly built
# installer over it, both ways a user does, and fail unless each update lands.
#
# Usage (from the repo root, on Windows, as an administrator, with GH_TOKEN set):
#   pwsh scripts/ci/check_win_update_over_release.ps1 -DistDir frontend/dist
#
# Installing over an existing Erudi runs the INSTALLED version's uninstaller
# with --updated. It moves every file of the install into
# %TEMP%\nsXXXX.tmp\old-install\ with a MoveFileW call that has no \\?\ prefix,
# and gives up (exit code 2, old install restored) on the first destination of
# 260 characters or more. frontend/assets/installers/installer.nsh removes the
# old backend's dist-info license folders, the deepest files of the install, on
# two paths, and each pass below proves one of them:
#
# 1. Manual install over an existing one: the installer runs with `/S` only.
#    The old uninstaller fails, electron-builder retries it, and the include's
#    customUnInstallCheck recovers. Expected: the old uninstaller starts more
#    than once, and the update lands.
# 2. Auto-update: the installer runs with `--updated /S`, what electron-updater
#    passes when it installs at quit ("Install now" passes `--updated
#    --force-run`, interactive; --force-run would start the app here). The
#    include's customInit removes the folders before the first attempt.
#    Expected: the old uninstaller starts exactly once, and the update lands.
#
# Between the two passes the new install is removed with its own uninstaller,
# and the release is installed again.
#
# Counting uninstaller starts: electron-builder copies the old uninstaller to
# $PLUGINSDIR\old-uninstaller.exe and runs that copy once per attempt (in place,
# as "Uninstall Erudi.exe", only when the copy cannot start). The script
# subscribes to Win32_ProcessStartTrace (kernel process-start events, which miss
# no process however short-lived; it needs an administrator, which hosted
# runners are) for the duration of each update and counts those starts. A
# failed first attempt is the defect; a single start is the proof that nothing
# had to be retried.
#
# The reproduction is guaranteed by path arithmetic, checked before each
# update: the uninstaller's destination is
#   <TEMP>\<NSIS plugins dir, at least 8 characters: nsX + hex + .tmp>\old-install
# followed by the file's path relative to the install directory. The script
# points TEMP and TMP at %LOCALAPPDATA%\Temp, the long form of a user's temp
# directory (on hosted runners they can hold the 8.3 form C:\Users\RUNNER~1\...,
# which is shorter than what an 11-character user name such as runneradmin
# gets on a real machine), and fails if the deepest file of the old install
# would still land below 260 characters there: a run that cannot hit the defect
# proves nothing.
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
  [string] $Repo = $env:GITHUB_REPOSITORY,
  [int] $TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$UninstallRoot = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall"

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

function Get-ErudiUninstallEntries {
  return @(
    Get-ChildItem -LiteralPath $UninstallRoot |
      ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } |
      Where-Object {
        $_.PSObject.Properties["UninstallString"] -and $_.UninstallString -like "*Uninstall Erudi.exe*"
      }
  )
}

function Get-ErudiUninstallEntry {
  $entries = Get-ErudiUninstallEntries
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

function Get-RelativeFiles([string] $Dir) {
  $full = (Resolve-Path -LiteralPath $Dir).ProviderPath.TrimEnd("\")
  return @(
    Get-ChildItem -LiteralPath $full -Recurse -Force -File |
      ForEach-Object { $_.FullName.Substring($full.Length) }
  )
}

function Get-OldInstallDirs([string] $TempDir) {
  return @(
    Get-ChildItem -LiteralPath $TempDir -Directory -Filter "ns*.tmp" -Force |
      Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "old-install") } |
      ForEach-Object { $_.FullName }
  )
}

function Install-Release {
  $code = Invoke-Installer $oldInstaller @("/S")
  if ($code -ne 0) { Fail "Installing $($asset.name) exited $code." }
  $old = Get-ErudiUninstallEntry
  Write-Host "Installed Erudi $($old.Version) ($($release.tagName), $($asset.name)) in $($old.InstallDir)"
  if ("v$($old.Version)" -ne $release.tagName) {
    Fail "The release is $($release.tagName) but the installed version is $($old.Version)."
  }
  return $old
}

# Fails unless the old install's deepest file lands at 260 characters or more
# under old-install\ with the shortest plugins directory name.
function Assert-CanReproduce($Old) {
  $oldFiles = Get-RelativeFiles $Old.InstallDir
  $licensePattern = '^\\resources\\backend\\_internal\\[^\\]+\.dist-info\\licenses\\'
  $deepest = $oldFiles | Sort-Object Length -Descending | Select-Object -First 1
  $deepestOther = $oldFiles | Where-Object { $_ -notmatch $licensePattern } | Sort-Object Length -Descending | Select-Object -First 1
  $prefix = (Join-Path $gateTemp "nsa0.tmp\old-install").Length
  Write-Host "Deepest file of the old install: $deepest"
  Write-Host "  in place: $($Old.InstallDir.Length + $deepest.Length) characters; under old-install: at least $($prefix + $deepest.Length)"
  Write-Host "Deepest file outside the dist-info license folders: $deepestOther"
  Write-Host "  under old-install: at least $($prefix + $deepestOther.Length), at most $($prefix + 3 + $deepestOther.Length)"
  if ($prefix + $deepest.Length -lt 260) {
    Fail "The deepest file would land at $($prefix + $deepest.Length) characters, under the 260 limit: this run cannot reproduce the update failure."
  }
}

# Runs the new installer over $Old with $Arguments, counting the old
# uninstaller's starts, checks the result, and returns the number of starts.
function Update-Over($Old, [string[]] $Arguments) {
  $leftoversBefore = Get-OldInstallDirs $gateTemp
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
  foreach ($f in (Get-RelativeFiles $new.InstallDir)) { [void]$installed.Add($f) }
  $expected = Get-RelativeFiles $unpacked
  $missing = @($expected | Where-Object { -not $installed.Contains($_) })
  if ($missing.Count -gt 0) {
    Fail "$($missing.Count) file(s) of the build are not installed, e.g. $($missing | Select-Object -First 5)"
  }
  foreach ($rel in @("\Erudi.exe", "\resources\app.asar", "\resources\backend\backend.exe")) {
    $a = (Get-FileHash -LiteralPath (Join-Path $unpacked $rel)).Hash
    $b = (Get-FileHash -LiteralPath (Join-Path $new.InstallDir $rel)).Hash
    if ($a -ne $b) { Fail "$rel differs from the build's copy after the update." }
  }
  Write-Host "All $($expected.Count) files of the build are installed; Erudi.exe, app.asar and backend.exe match it."

  $leftovers = @(Get-OldInstallDirs $gateTemp | Where-Object { $leftoversBefore -notcontains $_ })
  if ($leftovers.Count -gt 0) { Fail "old-install left under TEMP: $($leftovers -join ', ')" }
  Write-Host "No old-install directory left under $gateTemp."
  return $starts
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

# --- The latest published release's Windows CPU installer ---------------------
# `gh release view` with no tag reads the latest published release (never a
# draft or a prerelease). Any gh failure is fatal: a gate that skips on a
# network error proves nothing.
$releaseJson = gh release view --repo $Repo --json tagName,assets
if ($LASTEXITCODE -ne 0) { Fail "gh release view failed for $Repo (exit $LASTEXITCODE)." }
$release = ($releaseJson -join "`n") | ConvertFrom-Json
$assets = @(
  $release.assets | Where-Object { $_.name -match '^Erudi-Setup-.+\.exe$' -and $_.name -notmatch '-cuda\.exe$' }
)
if ($assets.Count -ne 1) {
  Fail "Expected one Windows CPU installer in $($release.tagName), found: $(($assets | ForEach-Object name) -join ', ')"
}
$asset = $assets[0]

$downloadDir = Join-Path ([IO.Path]::GetTempPath()) "erudi-update-gate-$PID"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$oldInstaller = Join-Path $downloadDir $asset.name
for ($attempt = 1; $attempt -le 3; $attempt++) {
  gh release download $release.tagName --repo $Repo --pattern $asset.name --dir $downloadDir --clobber
  if ($LASTEXITCODE -eq 0) { break }
  Write-Host "::warning::gh release download failed (attempt $attempt of 3, exit $LASTEXITCODE)."
  if ($attempt -eq 3) { Fail "Could not download $($asset.name) from $($release.tagName)." }
  Start-Sleep -Seconds 10
}
$size = (Get-Item -LiteralPath $oldInstaller).Length
if ($size -ne $asset.size) {
  Fail "$($asset.name) is $size bytes, the release lists $($asset.size)."
}
if (Get-ErudiUninstallEntries) { Fail "Erudi is already installed on this machine; the gate needs a clean one." }

$gateTemp = Join-Path $env:LOCALAPPDATA "Temp"
Write-Host "TEMP was $env:TEMP; the installers run with TEMP and TMP = $gateTemp"
$env:TEMP = $gateTemp
$env:TMP = $gateTemp

# --- Pass 1: manual install over an existing one (recovery) -------------------
Write-Host "`n== Pass 1: installer started by hand over Erudi $($release.tagName) (/S) =="
if ($env:GITHUB_STEP_SUMMARY) {
  "Windows update gate: over Erudi $($release.tagName) ($($asset.name)), to $newVersion." |
    Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
}
$old = Install-Release
Assert-CanReproduce $old
$starts = Update-Over $old @("/S")
if ($starts -lt 2) {
  Fail "The old uninstaller started $starts time(s): its first attempt did not fail, so this pass did not exercise the recovery."
}
Write-Host "Pass 1: the first uninstall attempt failed and the recovery completed the update."

# --- Remove the new install with its own uninstaller --------------------------
Write-Host "`n== Removing the new install =="
$current = Get-ErudiUninstallEntry
# An NSIS uninstaller copies itself to TEMP, starts the copy and exits, so the
# wait is on its effects: the uninstall entry and the install directory go.
$code = Invoke-Installer $current.Uninstaller @($current.UninstallArgs -split '\s+' | Where-Object { $_ })
if ($code -ne 0) { Fail "The uninstaller exited $code." }
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-ErudiUninstallEntries) -or (Test-Path -LiteralPath $current.InstallDir)) {
  if ((Get-Date) -gt $deadline) {
    Fail "Uninstalling left the uninstall entry or $($current.InstallDir) after $TimeoutSeconds s."
  }
  Start-Sleep -Seconds 1
}
Write-Host "Uninstalled: no uninstall entry, no $($current.InstallDir)."

# --- Pass 2: auto-update (preventive cleanup) ---------------------------------
Write-Host "`n== Pass 2: update the way electron-updater starts it (--updated /S) =="
$old = Install-Release
Assert-CanReproduce $old
$starts = Update-Over $old @("--updated", "/S")
if ($starts -ne 1) {
  Fail "The old uninstaller started $starts time(s), expected once: the license folders were not removed before the first attempt."
}
Write-Host "Pass 2: the first uninstall attempt succeeded."
