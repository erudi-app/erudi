# Building Erudi for Distribution

Erudi ships as an Electron app with a PyInstaller-frozen Python backend bundled
inside it. Every distributable is produced the same way:

1. Freeze the backend with PyInstaller → `backend/dist/backend/`
2. Stage that bundle where electron-builder expects it → `frontend/backend/`
3. Package the Electron app with **electron-builder** → `frontend/dist/`

There is no electron-forge in this repository: `frontend/electron-builder.yml`
is the single packaging configuration, and it picks the frozen backend up from
`../backend/dist/backend` via `extraResources`.

PyInstaller cannot cross-compile, so **each platform is built on that platform**
(natively or on a matching CI runner).

---

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | **exactly 3.12** | `pgserver` only ships cp312 wheels; CI pins `3.12` |
| Node.js | **22.18+, 24.11+ or 26+** | the CI runners use Node 22; Babel 8 and Vitest 5 together accept `^22.18.0 || ^24.11.0 || >=26.0.0` |
| npm | bundled with Node 22 | |
| PyInstaller | latest | installed into the backend venv |
| CUDA toolkit | any **12.x**; releases use **12.8** | only for the Windows/Linux CUDA legs, to compile `llama-server`. 12.8 is the first that emits native code for Blackwell (RTX 50); on an older 12.x the build scripts drop `120-real` and those cards JIT from PTX instead. Not needed for a CPU or macOS build — and never needed to *run* Erudi. |

Backend dependencies are not a single `requirements.txt`. Composed entrypoints
live under `backend/requirements/entrypoints/`:

```
backend/requirements/entrypoints/dev/{mac-silicon,win-cpu,win-cuda,linux-cpu,linux-cuda}.txt
backend/requirements/entrypoints/prod/{mac-silicon,win-cpu,win-cuda,linux-cpu,linux-cuda}-prod.txt
```

Read `backend/requirements/README.md` before touching any of them — common deps
live in `meta/base.txt` and platform/hardware specifics in `meta/*-specs.txt`;
the entrypoints only compose those.

On Windows and Linux the inference binary (`llama-server`) is compiled from the
`backend/forks/llama-cpp` submodule (upstream `ggml-org/llama.cpp`, pinned to a
release tag) before PyInstaller runs, so clone with `--recurse-submodules` (or run
`git submodule update --init --recursive`; a clone predating the move to upstream
needs `git submodule sync --recursive` first).
macOS does not need it: inference there goes through MLX, which is a pip
dependency.

### PyInstaller specs

| Spec | Target |
|---|---|
| `backend/backend-mac-silicon.spec` | macOS Apple Silicon |
| `backend/backend.spec` | Windows CUDA |
| `backend/backend-cpu.spec` | Windows CPU **and** Linux CPU |

The Linux CUDA leg reuses `backend/backend.spec`.

Which `llama-server` builds a spec bundles is decided by
`backend/pyi_llama_artifacts.py`, unit-tested in `tests/test_pyi_llama_artifacts.py`:
`backend-cpu.spec` ships `artifacts/llama-cpp/cpu/bin`, and `backend.spec` ships **both**
`cuda/bin` and `cpu/bin`. The CUDA engine runs the CUDA build; "Processor only" in
Settings runs the CPU build, which loads no NVIDIA library and starts without a driver.
A missing build fails the freeze, naming the script that compiles it.

---

## Local builds

**The version.** `frontend/package.json` keeps the last version committed to it;
only `release.yml` stamps the tag (`npm version <tag> --no-git-tag-version
--allow-same-version`) before electron-builder runs. Every local build therefore
carries the committed version everywhere the app shows one (the installer name,
Settings, the Diagnostics page), whatever the platform. To build a candidate
under a release's version, stamp it first and leave the stamp uncommitted:

```bash
cd frontend
npm version 1.1.3 --no-git-tag-version --allow-same-version
```

### macOS (Apple Silicon)

```bash
bash scripts/build/build-mac-silicon.sh
```

The script verifies the backend venv and npm, installs PyInstaller if missing,
runs `backend-mac-silicon.spec` into `backend/dist/backend/`, copies the bundle
to `frontend/backend/`, then runs `npm run dist:mac`. The DMG lands in
`frontend/dist/`.

To sign and notarize, three things are needed, all from an Apple Developer
Program membership:

- A **Developer ID Application** certificate installed in the login keychain.
  electron-builder finds it on its own; `security find-identity -v -p codesigning`
  lists what is installed, and `CSC_NAME` picks one when there are several.
- The **Team ID**, shown under Membership in the Apple Developer account.
- An **app-specific password** for the Apple ID, generated at
  [appleid.apple.com](https://appleid.apple.com) under Sign-In and Security.

Export them before running the script. These are the names electron-builder
reads for notarization, and the ones the release workflow sets from repository
secrets:

```bash
# .env.notarize (never committed)
export APPLE_ID=...
export APPLE_APP_SPECIFIC_PASSWORD=...
export APPLE_TEAM_ID=...
export APPLE_SIGNING_IDENTITY="Developer ID Application: <name> (<team id>)"

source .env.notarize
bash scripts/build/build-mac-silicon.sh
```

`APPLE_SIGNING_IDENTITY` is read by the script alone: without it the build still
runs, and the script warns that the app is unsigned and macOS may block it on
first launch.

### Windows (CUDA)

```powershell
.\scripts\build\build-win-cuda.ps1
```

Same shape: prerequisites, PyInstaller with `backend.spec`, copy to
`frontend\backend\`, then `npm run dist:win`. The NSIS installer lands in
`frontend\dist\`.

The installer carries two builds of `llama-server.exe`, and the script expects
both already compiled, each with its runtime DLLs beside it:

```powershell
.\scripts\dev\backend\build-llamacpp-cuda-win.ps1   # artifacts\llama-cpp\cuda\bin
.\scripts\dev\backend\build-llamacpp-cpu-win.ps1    # artifacts\llama-cpp\cpu\bin
```

Both scripts find Visual Studio through `vswhere` — under
`${env:ProgramFiles(x86)}`, else on `PATH` — and build with Ninja in the MSVC
environment of `vcvarsall.bat`. When the toolchain lives elsewhere, or `vswhere`
lost its instance state (kept under `ProgramData`, emptied by a Windows
reinstall), point both builds at `vcvarsall.bat` directly; this is checked first:

```powershell
$env:ERUDI_VCVARSALL = "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
```

A development setup does not need the CPU build: `setup-win-cuda.ps1` compiles
the CUDA one only, and "Processor only" then runs the CUDA build with every GPU
hidden from it (`CUDA_VISIBLE_DEVICES=-1`), logging a WARNING that it fell back
to the other flavour.

**The CUDA update channel.** The release's `win-cuda` leg passes
`-c.publish.channel=cuda` and a `-cuda` artifact name to electron-builder (see
[below](#ci-release-pipeline)); CI runs it in bash. From PowerShell, quote every
`-c.` argument: unquoted, PowerShell splits `-c.publish.channel=cuda` at the dot
and electron-builder reads `-c` as a config file path. Single quotes also keep
`${productName}`, `${version}` and `${ext}` literal for electron-builder.

```powershell
cd frontend
npm run build
npx electron-builder --win --config electron-builder.yml '-c.publish.channel=cuda' '-c.nsis.artifactName=${productName}-Setup-${version}-cuda.${ext}'
```

### Linux, and Windows CPU

There is no local build script for these two. Build them through CI (a `v*` tag,
see below) or run the manual sequence.

### Manual sequence (any platform)

```bash
# 1. Backend
cd backend
source venv/bin/activate                       # venv\Scripts\activate on Windows
pip install -r requirements/entrypoints/prod/<platform>-prod.txt
pip install pyinstaller
pyinstaller <spec> --distpath dist             # -> backend/dist/backend/

# 2. Frontend
cd ../frontend
npm ci
npm run dist:mac     # or dist:win / dist:linux
```

`electron-builder.yml` pulls the backend straight from `../backend/dist/backend`,
so the copy into `frontend/backend/` that the platform scripts do is a
convenience, not a requirement of the manual path.

---

## Build outputs

Everything electron-builder produces goes to `frontend/dist/` (`directories.output`):

| Platform | Artifacts |
|---|---|
| macOS | `.dmg` (first install) + `.zip` (electron-updater deltas), arm64 |
| Windows | NSIS `.exe`, x64 |
| Linux | `AppImage`, x64 |

macOS builds set `hardenedRuntime`, apply `assets/entitlements.mac.plist`,
`notarize: true`, and list the bundled backend under `mac.binaries`
(`Contents/Resources/backend/backend`) as belt-and-braces: electron-builder's
own signing walk already recurses through the whole `Contents/` tree,
including `Resources/`, and signs every binary file it finds there.

### Windows update over an existing install

Installing over an existing Erudi, by hand or through the auto-updater, makes
the new installer run the uninstaller of the installed version with `--updated`.
That uninstaller (electron-builder's `un.atomicRMDir`) moves every file of the
install into `%TEMP%\nsXXXX.tmp\old-install\`: it recreates each directory there
with `CreateDirectory` and moves each file with NSIS `Rename` (`MoveFileW`), both
without the `\\?\` prefix, so it cannot write a directory of 248 characters or
more, nor a file of 260 or more. On the first file it cannot move, it moves the
others back and exits with code 2, and electron-builder's installer stops with
"Failed to uninstall old application files".

The deepest files of the install are the license notices that wheels ship in
`resources\backend\_internal\<name>.dist-info\licenses\`: torch's vendored
`third_party` tree puts a `LICENSE.txt` 200 characters below the install
directory, in a directory 188 characters below it. The destination prefix
`C:\Users\<user>\AppData\Local\Temp\nsXXXX.tmp\old-install` is 48 characters
plus the user name plus the 1 to 4 hex digits of the plugins directory name
(`nsX`, hex, `.tmp`, picked at random), and both limits are reached at the same
length: the failure starts at a user name of 8 to 11 characters, possible from
8, certain from 11. The notices ship on purpose (torch's BSD license requires
them).

`nsis.include` in `electron-builder.yml` adds
`frontend/assets/installers/installer.nsh` to the installer. It removes the
`*.dist-info\licenses` folders under the old install's
`resources\backend\_internal\`, so the old uninstaller never has to move them.
The old install's directory comes from the registry (`InstallLocation`, as
electron-builder reads it), must be an absolute drive path holding `Erudi.exe`
and the frozen backend, and is addressed through a `\\?\` path so the deletion
itself has no length limit. Nothing else is deleted, and the new install ships
the same notices again. The removal runs at two points:

- **Before the old uninstaller, on every install over an existing one** (an
  update, which electron-updater starts with `--updated`, silently at quit or
  interactively from "Install now", or an installer started by hand): in
  `customInit`, at the end of the installer's `.onInit`. The old uninstaller's
  first attempt then succeeds, so electron-builder never reaches its retry loop
  (five attempts, one second apart) or its "Erudi cannot be closed. Please close
  it manually and click Retry to continue." dialog, whose Retry repeats the
  failing uninstall. `.onInit` runs before the installer's pages: a user who
  cancels an interactive install at that point keeps the old install, which
  works, without its license notices until the next install.
- **After a failed uninstall, as a safety net** (an old install that
  `customInit` did not clean): in `customUnInstallCheck` and
  `customUnInstallCheckCurrentUser`, which electron-builder's
  `handleUninstallResult` runs instead of its own check of the old
  uninstaller's exit code. They keep that check (an uninstaller that cannot
  start is logged and the install goes on; a non-zero exit shows the failure
  message, sets exit code 2 and quits) and add one recovery before giving up:
  remove the folders, run the old uninstaller once more. When the old
  uninstaller succeeds the first time, none of this runs.

`app-build-smoke.yml` exercises the path on its Windows leg with
`scripts/ci/check_win_update_over_release.ps1 -BaseTag v1.1.2`. The base release
is pinned: v1.1.2 carries the torch license tree, and on the runner's user,
`runneradmin`, its deepest file lands at exactly 260 characters under
old-install (48 + 11 + 1 + 200) with the shortest plugins directory name, so a
later release without that tree could not reproduce the failure. The script
points `TEMP` at its long form and fails when the old install's deepest file
would land below 260 characters, since such a run would prove nothing.

It installs the base release's CPU installer silently and runs the freshly
built installer over it with `/S` alone (an installer started by hand), removes
the result with its own uninstaller, installs the base release again and runs
the new installer with `--updated /S` (what electron-updater passes when it
installs at quit). After each update it checks the exit code, the installed
version and files, and that no `old-install` folder is left behind. It also
counts the starts of the old uninstaller through `Win32_ProcessStartTrace`, and
requires exactly one on both passes: a second start means the first attempt
failed.

Before installing anything, the script also checks the margins of the new
build outside the license folders it removes: under
`C:\Users\<20-character name>\AppData\Local\Temp\nsa0.tmp\old-install\`, its
deepest directory must stay below 248 characters and its deepest file below
260, so a new deep path in the bundle fails the gate before it reaches users.
It logs the deepest path of the base release's CUDA installer too, without
failing on it.

---

## Development mode

Development does **not** need a frozen backend. Run the two processes side by
side; the Electron main process expects the backend to be up already and simply
health-checks it.

```bash
# Terminal 1 — backend
cd backend && source venv/bin/activate && python run.py --port 27182

# Terminal 2 — frontend
cd frontend && npm start
```

`bash scripts/dev/dev-start.sh` does both in two Terminal windows (macOS only)
after killing whatever holds `BACKEND_PORT`.

Keep `run.py` as the entrypoint rather than a bare `uvicorn` call: it supervises
uvicorn, scans ports `27182-27199`, and emits the newline-delimited JSON
lifecycle events (`starting`, `ready`, `shutdown`, `startup_error`) that the
Electron main process parses.

In production the main process spawns the packaged backend with only two extra
environment variables — `PYTHONUTF8=1` and `ERUDI_WATCH_STDIN=1`
(`frontend/src/main.js`). Everything else (data directories, database URL) is
resolved by the backend itself in `src/launcher/runtime_paths.py`.

---

## CI release pipeline

`.github/workflows/release.yml` runs on any `v*` tag with a five-leg matrix:

| Leg | Runner | Requirements | Spec | llama-server |
|---|---|---|---|---|
| `mac-arm` | `macos-15` | `mac-silicon-prod.txt` | `backend-mac-silicon.spec` | — (MLX) |
| `win-cpu` | `windows-2022` | `win-cpu-prod.txt` | `backend-cpu.spec` | cpu |
| `win-cuda` | `windows-2022` | `win-cuda-prod.txt` | `backend.spec` | cuda + cpu |
| `linux-cpu` | `ubuntu-22.04` | `linux-cpu-prod.txt` | `backend-cpu.spec` | cpu |
| `linux-cuda` | `ubuntu-22.04` | `linux-cuda-prod.txt` | `backend.spec` | cuda + cpu |

Each leg: checks out with submodules, sets up Python 3.12 and Node 22, compiles
`llama-server` from the submodule (Windows/Linux only; the CUDA legs compile the
CPU build too, in the same job), runs
`pyinstaller backend/<spec> --distpath backend/dist`, verifies the frozen tree,
then `npm ci`, `npm version <tag>` and `npm run release:<platform>`.

The verification step locates each bundled `llama-server` inside `backend/dist` at
its own path (`artifacts/llama-cpp/cpu/bin`, plus `cuda/bin` on the CUDA legs) and
starts the CPU copy, on every leg, with `--version` — which prints llama.cpp's build
header and exits 0 from the argument parser, without loading a model. A filename
alone would match a truncated file or a wrong-architecture build; starting it
proves the copy PyInstaller made is a loadable executable whose libraries resolve.
On Windows the CPU copy must also have the MSVC runtime beside it and no CUDA
runtime (a `cudart64_12.dll` there means the CUDA build landed in the CPU folder),
and a CPU leg must carry no `cuda/bin` at all. The CUDA copies are not started:
ggml links the NVIDIA driver library (`libcuda.so.1` /
`nvcuda.dll`, shipped with the driver rather than the toolkit), so the loader
fails before `main()` on a driverless runner. Linux CUDA is checked with `ldd`
instead — every dependency of the bundled binary must resolve except that driver
library, which is the only thing that would catch a freeze dropping
`libcublas.so.12` — and Windows CUDA is checked for the bundled CUDA runtime
DLLs. Real hardware in the QA pass is the first thing that runs either.

The `mac-arm` leg runs on a newer macOS than the app's floor
(`mac.minimumSystemVersion` in `electron-builder.yml`, 14.0), and pip prefers
the wheels built for the runner's own macOS (mlx, mlx-metal and orjson publish
wheels built for recent macOS releases). So the leg installs through
`scripts/ci/install_backend_requirements.sh`, which downloads the requirements
with `--only-binary=:all: --platform macosx_<floor>_arm64` into a wheelhouse and
installs from it alone. Restricting pip to the floor's wheels could make it fall
back to an older release of an unpinned dependency, so the script also resolves
the file natively (`pip install --dry-run --report`) and fails when an installed
version differs from it. Then `scripts/ci/check_macos_floor.sh backend/dist <floor>`
reads the minimum macOS of every Mach-O file in the frozen backend (`otool -l`)
and fails the leg on any file above the floor. `app-build-smoke.yml` calls the
same two scripts on its macOS leg, so the smoke gate installs what a release
installs.

The CUDA legs inject `-c.publish.channel=cuda` (plus a distinct artifact name)
so the GPU builds feed a separate `cuda` auto-update channel while the CPU
builds feed the default `latest` channel. The electron-builder step runs in bash
(git-bash on Windows): `-c.publish.channel=cuda` goes through as is, and the
artifact name's value is single-quoted so bash keeps `${productName}` literal for
electron-builder. From PowerShell every `-c.` argument needs quotes (see
[Windows (CUDA)](#windows-cuda)). macOS signing and notarization happen
on the `mac-arm` leg only.

### Draft, QA, promote

A tag publishes a **draft** GitHub Release (`releaseType: draft` in
`electron-builder.yml`). electron-updater's feed 404s while a release is
drafted, so no installed client is touched.

Promotion is a separate manual step: `.github/workflows/release-promote.yml`
(`workflow_dispatch`, takes the tag as input). It refuses to publish unless all
five platform legs of the tag's `release.yml` run concluded green, which
prevents a partial release with a missing `latest-*.yml` feed.

Between the two, run the manual QA pass on the real signed artifact:
`docs/dev/release-qa-checklist.md`.
