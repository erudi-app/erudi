"""Where the installed app, its data and its logs live on each OS (docs/APP_REFERENCE.md)."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()


@dataclass
class AppLayout:
    os_name: str  # darwin | windows | linux
    app_path: Path | None  # Erudi.app / install dir / AppImage file
    main_exe: Path | None  # binary to exec; None for an unknown layout
    resources_dir: Path | None
    data_root: Path
    backend_log_dir: Path
    capture_logs: list[Path]
    electron_log_dir: Path
    extra_install_dirs: list[Path] = field(default_factory=list)  # e.g. AppImage mount, learnt at runtime

    @property
    def backend_lib(self) -> Path | None:
        return self.resources_dir / "backend" / "_internal" if self.resources_dir else None

    @property
    def install_dirs(self) -> tuple[str, ...]:
        dirs = [p for p in [self.app_path if self.app_path and self.app_path.is_dir() else None, self.resources_dir, *self.extra_install_dirs] if p]
        if self.app_path and self.app_path.is_file():
            dirs.append(self.app_path)  # an AppImage file
        return tuple(str(d) for d in dirs)

    def learn_main_exe(self, exe: str) -> None:
        """AppImage: resources live in the runtime mount, only known once the app runs."""
        if not exe:
            return
        exe_dir = Path(exe).parent
        if self.os_name == "linux" and exe_dir not in self.extra_install_dirs and (exe_dir / "resources").is_dir():
            self.extra_install_dirs.append(exe_dir)
            self.resources_dir = exe_dir / "resources"


def current_os() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def resolve_layout(app_path: str | None = None, os_name: str | None = None, env: dict | None = None) -> AppLayout:
    os_name = os_name or current_os()
    env = os.environ if env is None else env
    if os_name == "darwin":
        app = Path(app_path) if app_path else Path("/Applications/Erudi.app")
        main_exe = app / "Contents" / "MacOS" / "Erudi"
        resources = app / "Contents" / "Resources"
        if not app.suffix == ".app" and app.is_dir():
            main_exe, resources = _generic_dir(app)
        return AppLayout(
            os_name, app, main_exe, resources,
            data_root=HOME / "Library" / "Application Support" / "erudi" / "backend" / "prod",
            backend_log_dir=HOME / "Library" / "Logs" / "erudi",
            capture_logs=_capture_logs(Path(env.get("TMPDIR") or tempfile.gettempdir())),
            electron_log_dir=HOME / "Library" / "Logs" / "Erudi",
        )
    if os_name == "windows":
        local = Path(env.get("LOCALAPPDATA") or HOME / "AppData" / "Local")
        roaming = Path(env.get("APPDATA") or HOME / "AppData" / "Roaming")
        app = Path(app_path) if app_path else local / "Programs" / "Erudi"
        if app.suffix.lower() == ".exe":
            app = app.parent
        return AppLayout(
            os_name, app, app / "Erudi.exe", app / "resources",
            data_root=local / "erudi" / "backend" / "prod",
            backend_log_dir=local / "erudi" / "logs",
            capture_logs=_capture_logs(Path(env.get("TEMP") or tempfile.gettempdir())),
            electron_log_dir=roaming / "Erudi" / "logs",
        )
    # linux
    data_home = Path(env.get("XDG_DATA_HOME") or HOME / ".local" / "share")
    state_home = Path(env.get("XDG_STATE_HOME") or HOME / ".local" / "state")
    app = Path(app_path) if app_path else _find_appimage()
    main_exe, resources = (app, None) if app and app.is_file() else _generic_dir(app) if app else (None, None)
    return AppLayout(
        os_name, app, main_exe, resources,
        data_root=data_home / "erudi" / "backend" / "prod",
        backend_log_dir=state_home / "erudi" / "logs",
        capture_logs=_capture_logs(Path("/tmp")),
        electron_log_dir=HOME / ".config" / "Erudi" / "logs",
    )


def _capture_logs(tmp: Path) -> list[Path]:
    return [tmp / "erudi-backend.log", tmp / "erudi-backend.old.log"]


def _generic_dir(app: Path) -> tuple[Path | None, Path | None]:
    """An unpacked install dir (linux-unpacked, a test fixture): <dir>/erudi[.exe|Erudi] + <dir>/resources."""
    for name in ("erudi", "Erudi", "Erudi.exe", "erudi.exe"):
        if (app / name).is_file():
            return app / name, app / "resources"
    return None, app / "resources"


def _find_appimage() -> Path | None:
    candidates = sorted((HOME / "Applications").glob("Erudi-*.AppImage")) if (HOME / "Applications").is_dir() else []
    return candidates[-1] if candidates else None


def app_version(layout: AppLayout) -> dict[str, str | None]:
    """Version string and build flavour (mlx / cpu / cuda / unknown) from the install, without running it."""
    version, flavour = None, "unknown"
    app = layout.app_path
    if layout.os_name == "darwin" and app and (app / "Contents" / "Info.plist").exists():
        import plistlib

        with open(app / "Contents" / "Info.plist", "rb") as fh:
            version = plistlib.load(fh).get("CFBundleShortVersionString")
        flavour = "mlx"
    elif layout.os_name == "windows" and layout.main_exe and layout.main_exe.exists():
        import subprocess

        try:
            # -LiteralPath with a doubled-quote escape: --app-path is user
            # input, and a stray apostrophe must not break out of the string.
            exe_quoted = str(layout.main_exe).replace("'", "''")
            version = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Item -LiteralPath '{exe_quoted}').VersionInfo.ProductVersion"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            version = None
    elif app and app.is_file():
        m = re.search(r"Erudi-(\d[\w.\-]*?)(-cuda)?\.AppImage$", app.name)
        if m:
            version = m.group(1)
            flavour = "cuda" if m.group(2) else "cpu"
    if layout.os_name != "darwin" and layout.backend_lib and (layout.backend_lib / "artifacts" / "llama-cpp").is_dir():
        flavour = "cuda" if (layout.backend_lib / "artifacts" / "llama-cpp" / "cuda").is_dir() else "cpu"
    return {"version": version, "flavour": flavour}
