# check_win_update_over_release.ps1
#
# Install the latest published Windows CPU release, then run a freshly built
# installer over it the way electron-updater does, and fail unless the update
# lands.
#
# Usage (from the repo root, on Windows, with GH_TOKEN set):
#   pwsh scripts/ci/check_win_update_over_release.ps1 -DistDir frontend/dist
#
# Installing over an existing Erudi runs the INSTALLED version's uninstaller
# with --updated. It moves every file of the install into
# %TEMP%\nsXXXX.tmp\old-install\ with a MoveFileW call that has no \\?\ prefix,
# and gives up (exit code 2, old install restored) on the first destination of
# 260 characters or more. frontend/assets/installers/installer.nsh retries that
# uninstall once without the old backend's dist-info license folders, the
# deepest files of the install. This script proves the retry works against a
# real published release.
#
# The reproduction is guaranteed by path arithmetic, checked below before the
# update runs: the uninstaller's destination is
#   <TEMP>\<NSIS plugins dir, at least 8 characters: nsX + hex + .tmp>\old-install
# followed by the file's path relative to the install directory. The script
# points TEMP and TMP at %LOCALAPPDATA%\Temp, the long form of a user's temp
# directory (on hosted runners they can hold the 8.3 form C:\Users\RUNNER~1\...,
# which is shorter than what an 11-character user name such as runneradmin
# gets on a real machine), and fails if the deepest file of the old install
# would still land below 260 characters there: a run that cannot hit the defect
# proves nothing.
#
# electron-updater runs the installer with `--updated /S` when it installs at
# quit, and `--updated --force-run` (interactive) for "Install now"; both take
# the same uninstall path. This script uses `--updated /S`: --force-run would
# start the app after the install.
#
# Checks: both installers exit 0 within the timeout (a failed update shows a
# message box even in silent mode, so a failure can surface as a timeout), the
# uninstall entry reports the new version, every file of the build's
# win-unpacked tree is installed and the main binaries match it byte for byte,
# and no old-install directory is left under TEMP.

[CmdletBinding()]
param(
  # electron-builder output: holds exactly one installer .exe and win-unpacked\.
  [Parameter(Mandatory = $true)] [string] $DistDir,
  [string] $Repo = $env:GITHUB_REPOSITORY,
  [int] $TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Fail([string] $Message) {
  Write-Host "::error::$Message"
  exit 1
}

function Invoke-Installer([string] $Path, [string[]] $Arguments) {
  Write-Host "Running: `"$Path`" $($Arguments -join ' ')"
  $proc = Start-Process -FilePath $Path -ArgumentList $Arguments -PassThru
  $null = $proc.Handle # keeps the handle so ExitCode is readable after exit
  if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
    & taskkill.exe /F /T /PID $proc.Id | Out-Host
    Fail "`"$Path`" did not exit within $TimeoutSeconds s. A failed uninstall of the old version blocks on a message box, even in silent mode."
  }
  return $proc.ExitCode
}

function Get-ErudiUninstallEntry {
  $root = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall"
  $entries = @(
    Get-ChildItem -LiteralPath $root |
      ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } |
      Where-Object {
        $_.PSObject.Properties["UninstallString"] -and $_.UninstallString -like "*Uninstall Erudi.exe*"
      }
  )
  if ($entries.Count -ne 1) {
    Fail "Expected one Erudi entry under $root, found $($entries.Count)."
  }
  $entry = $entries[0]
  if ($entry.UninstallString -notmatch '^"([^"]+)"') {
    Fail "Unexpected UninstallString: $($entry.UninstallString)"
  }
  return [pscustomobject]@{
    Version    = $entry.DisplayVersion
    InstallDir = Split-Path -Parent $Matches[1]
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

# --- First install: the published release -------------------------------------
$code = Invoke-Installer $oldInstaller @("/S")
if ($code -ne 0) { Fail "Installing $($asset.name) exited $code." }
$old = Get-ErudiUninstallEntry
Write-Host "First install: Erudi $($old.Version) ($($release.tagName), $($asset.name)) in $($old.InstallDir)"
if ("v$($old.Version)" -ne $release.tagName) {
  Fail "The release is $($release.tagName) but the installed version is $($old.Version)."
}
if ($env:GITHUB_STEP_SUMMARY) {
  "Windows update gate: installed over Erudi $($old.Version) ($($release.tagName), $($asset.name)), updated to $newVersion." |
    Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
}

# --- Path arithmetic: the update must be able to hit the defect ---------------
$gateTemp = Join-Path $env:LOCALAPPDATA "Temp"
Write-Host "TEMP was $env:TEMP; the update runs with TEMP and TMP = $gateTemp"
$env:TEMP = $gateTemp
$env:TMP = $gateTemp

$oldFiles = Get-RelativeFiles $old.InstallDir
$licensePattern = '^\\resources\\backend\\_internal\\[^\\]+\.dist-info\\licenses\\'
$deepest = $oldFiles | Sort-Object Length -Descending | Select-Object -First 1
$deepestOther = $oldFiles | Where-Object { $_ -notmatch $licensePattern } | Sort-Object Length -Descending | Select-Object -First 1
$prefix = (Join-Path $gateTemp "nsa0.tmp\old-install").Length # shortest plugins dir name
Write-Host "Deepest file of the old install: $deepest"
Write-Host "  in place: $($old.InstallDir.Length + $deepest.Length) characters; under old-install: at least $($prefix + $deepest.Length)"
Write-Host "Deepest file outside the dist-info license folders: $deepestOther"
Write-Host "  under old-install: at least $($prefix + $deepestOther.Length), at most $($prefix + 3 + $deepestOther.Length)"
if ($prefix + $deepest.Length -lt 260) {
  Fail "The deepest file would land at $($prefix + $deepest.Length) characters, under the 260 limit: this run cannot reproduce the update failure."
}

$leftoversBefore = Get-OldInstallDirs $gateTemp

# --- The update ----------------------------------------------------------------
$code = Invoke-Installer $newInstaller @("--updated", "/S")
if ($code -ne 0) { Fail "The update over Erudi $($old.Version) exited $code." }

# --- The new version is in place ---------------------------------------------
$new = Get-ErudiUninstallEntry
Write-Host "After the update: Erudi $($new.Version) in $($new.InstallDir)"
if ($new.Version -ne $newVersion) { Fail "The uninstall entry reports $($new.Version), expected $newVersion." }
if ($new.InstallDir -ne $old.InstallDir) { Fail "The install moved from $($old.InstallDir) to $($new.InstallDir)." }

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
