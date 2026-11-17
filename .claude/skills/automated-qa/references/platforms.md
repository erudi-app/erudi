# Platforms

Everything that differs between macOS, Windows and Linux. The pass itself does
not. Paths come from `backend/src/launcher/runtime_paths.py` (data and backend
logs) and `frontend/electron-builder.yml` (artifacts, install layout); if either
changes, this file is what goes stale first.

## Where things live

| | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| Artifact on the draft | `Erudi-<v>-arm64.dmg` | `Erudi-Setup-<v>.exe` (CPU) / `Erudi-Setup-<v>-cuda.exe` (CUDA) | `Erudi-<v>.AppImage` (CPU) / `Erudi-<v>-cuda.AppImage` |
| Installed app | `/Applications/Erudi.app` | `%LOCALAPPDATA%\Programs\Erudi\Erudi.exe` (per-user NSIS) | wherever the AppImage was placed |
| Backend data (models, DB, KB) | `~/Library/Application Support/erudi/backend/prod/` | `%LOCALAPPDATA%\erudi\backend\prod\` | `$XDG_DATA_HOME/erudi/backend/prod/` (default `~/.local/share/erudi/...`) |
| Backend's own log dir | `~/Library/Logs/erudi/` | `%LOCALAPPDATA%\erudi\logs\` | `$XDG_STATE_HOME/erudi/logs/` (default `~/.local/state/erudi/logs/`) |
| Electron logs (`main.log`, `backend.log`) | `~/Library/Logs/Erudi/` | `%APPDATA%\Erudi\logs\` | `~/.config/Erudi/logs/` |
| Backend stdout capture | `$TMPDIR/erudi-backend.log` | `%TEMP%\erudi-backend.log` | `/tmp/erudi-backend.log` |
| Electron profile (localStorage, caches) | `~/Library/Application Support/Erudi/` (same dir as the data on a case-insensitive FS) | `%APPDATA%\Erudi\` | `~/.config/Erudi/` |
| Electron profile of a run from source | `~/Library/Application Support/erudi/` (the same folder as the installed app's on a case-insensitive FS) | `%APPDATA%\erudi\` (the same folder as the installed app's) | `~/.config/erudi/` (a separate folder) |
| Inference child | `mlx_vlm.server` (an `mp.Process` of the backend) | `llama-server.exe` | `llama-server` |
| Renderer sandbox | seatbelt, `--enable-sandbox` expected | expected sandboxed | `--no-sandbox` expected (user-namespace workaround) |

On macOS `erudi/` and `Erudi/` are the same directory. The Electron logs are
**not** inside the data directory on any platform, so they survive the clean
step: always filter them by today's date before reading them as this run's.

Help → Clear All Data, by contrast, deletes the backend data folder, both log
folders and the stdout capture, and empties the Electron profile's storage
(the profile folder itself stays): the full list is in `docs/privacy.md`,
"What Clear All Data removes". Pasted images live in the backend data folder
(`.../prod/data/pasted-images/`). A folder is deleted only when it carries
`.erudi-owned`; a file still held is retried at the next start of the same app
(the marker is `clear-all-data-pending.json` in the profile folder, at most
three starts).

## Confirm nothing is running

```bash
# macOS / Linux
pgrep -fl "Erudi|erudi|pginstall|llama-server|mlx_vlm" | grep -v "$0"
lsof -ti :27182 :9222
```

```powershell
# Windows
Get-Process Erudi, backend, postgres, llama-server -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 27182 -ErrorAction SilentlyContinue
```

`pgrep -f` matches its own shell when the pattern appears in the command line
— a pipeline that greps for `no-sandbox` finds itself. Exclude the shell, or
match on the executable path (`Erudi.app/Contents/MacOS`, `pginstall/bin/postgres`).

## Clean state

Inventory first, then remove. Say what is going.

```bash
# macOS
du -sh ~/Library/Application\ Support/erudi
rm -rf /Applications/Erudi.app ~/Library/Application\ Support/erudi
```

```powershell
# Windows — run the uninstaller so the NSIS registry entries go too
& "$env:LOCALAPPDATA\Programs\Erudi\Uninstall Erudi.exe" /S
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\erudi", "$env:APPDATA\Erudi"
```

```bash
# Linux
rm -f ~/Applications/Erudi-*.AppImage
rm -rf ~/.local/share/erudi ~/.local/state/erudi ~/.config/Erudi
```

## Install from the draft

```bash
gh release download v<version> --pattern "<pattern>" --dir <scratchpad>
```

**macOS** — mount, copy, *verify*, unmount, and only then delete the image.
The volume name is not the app name: read it from `hdiutil`'s output.

```bash
MOUNT=$(hdiutil attach <scratchpad>/Erudi-<v>-arm64.dmg -nobrowse | tail -1 | grep -o '/Volumes/.*')
cp -R "$MOUNT/Erudi.app" /Applications/ && ls /Applications/Erudi.app/Contents/MacOS/Erudi
hdiutil detach "$MOUNT" -quiet
rm -f <scratchpad>/Erudi-<v>-arm64.dmg
spctl -a -vv /Applications/Erudi.app        # expect "Notarized Developer ID"
/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" /Applications/Erudi.app/Contents/Info.plist
```

**Windows** — silent per-user install, then confirm the exe exists.

```powershell
Start-Process -Wait "<scratchpad>\Erudi-Setup-<v>.exe" -ArgumentList "/S"
Test-Path "$env:LOCALAPPDATA\Programs\Erudi\Erudi.exe"
Remove-Item "<scratchpad>\Erudi-Setup-<v>.exe"
(Get-Item "$env:LOCALAPPDATA\Programs\Erudi\Erudi.exe").VersionInfo.ProductVersion
```

The CUDA installer is the same flow with the `-cuda` artifact; it is the one
that needs an NVIDIA machine and it is the one that verifies the bundled
runtime DLLs (`cudart64_12`, `cublas64_12`, `cublasLt64_12`) actually ship.

**Linux** — make the AppImage executable and run it in place.

```bash
chmod +x <scratchpad>/Erudi-<v>.AppImage
mv <scratchpad>/Erudi-<v>.AppImage ~/Applications/
```

## Launch with the debugging port

```bash
open -a Erudi --args --remote-debugging-port=9222              # macOS
~/Applications/Erudi-<v>.AppImage --remote-debugging-port=9222 & # Linux
```

```powershell
Start-Process "$env:LOCALAPPDATA\Programs\Erudi\Erudi.exe" -ArgumentList "--remote-debugging-port=9222"
```

Then wait on conditions:

```bash
until curl -s http://127.0.0.1:9222/json/version >/dev/null; do sleep 1; done
until [ "$(curl -s -o /dev/null -w '%{http_code}' -L http://127.0.0.1:27182/erudi/health)" = 200 ]; do sleep 2; done
```

## Quit, close, hard-kill

| | macOS | Windows | Linux |
|---|---|---|---|
| Clean quit | `osascript -e 'quit app "Erudi"'` | `Stop-Process -Name Erudi` is a *hard* kill; quit through the tray/menu or `taskkill /IM Erudi.exe` (graceful) | `pkill -TERM -f Erudi` |
| Hard kill (scenario) | `kill -9 <main pid>` | `Stop-Process -Name Erudi -Force` / Task Manager "End task" | `kill -9 <main pid>` |
| Close last window | app keeps running (expected) | app quits and backend stops (expected) | app quits and backend stops (expected) |

After a hard kill, what must be gone within a few seconds: the backend, the
embedded `postgres` (path contains `pginstall`), and any inference child. Any
system-wide PostgreSQL the machine already runs (Homebrew, a service) is not
ours — check the path before calling it an orphan.

## Reading the logs

The backend's `app.log` carries request ids: `fe-…` for calls made by the
renderer, `be-…` for anything else (your own `curl`, the updater, the poll).
That prefix is how you tell "the app did this" from "I did this".

```bash
grep "$(date -u +%Y-%m-%d)" <electron-logs>/backend.log | grep -E "Tool invoked|Turn mode|Query completed"
```
