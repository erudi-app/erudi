"""Find the Erudi process tree and attribute every process to a category.

Everything below `take_snapshot()` is pure: it works on a list of `ProcInfo`
records, so the rules are unit-tested over recorded/modelled command lines for
macOS, Windows and Linux (tests/fixtures/processes_*.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES = (
    "electron_main",
    "electron_renderer",
    "electron_gpu",
    "electron_utility",
    "backend",
    "database",
    "inference",
    "embedding",
    "transient",
)

MAIN_EXE_STEMS = ("erudi",)
BACKEND_STEM = "backend"
PG_BIN_MARKER = "/pginstall/bin/"
PG_TOOL_STEMS = ("pg_ctl", "initdb", "pg_dump", "pg_restore", "psql", "pg_isready")
CRASHPAD_STEMS = ("chrome_crashpad_handler", "crashpad_handler")
MP_HELPER_MARKERS = ("resource_tracker", "semaphore_tracker", "forkserver")


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    name: str = ""
    exe: str = ""
    cmdline: tuple[str, ...] = ()
    create_time: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ProcInfo":
        return cls(
            pid=int(d["pid"]),
            ppid=int(d.get("ppid") or 0),
            name=d.get("name") or "",
            exe=d.get("exe") or "",
            cmdline=tuple(d.get("cmdline") or ()),
            create_time=float(d.get("create_time") or 0.0),
        )


@dataclass(frozen=True)
class DiscoveryContext:
    """What the harness knows about the installation it is observing."""

    main_pid: int | None = None
    install_dirs: tuple[str, ...] = ()  # app bundle / install dir / resources dir
    data_root: str | None = None  # .../erudi/backend/prod
    api_port: int = 27182
    exclude_pids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Classified:
    pid: int
    category: str
    role: str
    proc: ProcInfo


def norm_path(path: str) -> str:
    """Separator- and case-normalised path for matching (APFS/NTFS are case-insensitive)."""
    return (path or "").replace("\\", "/").casefold()


def path_stem(path: str) -> str:
    base = norm_path(path).rstrip("/").rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".exe") else base


def identity_paths(p: ProcInfo) -> list[str]:
    """Executable path plus argv[0] (they differ when a binary is started via a link)."""
    paths = [p.exe] if p.exe else []
    if p.cmdline and p.cmdline[0] and p.cmdline[0] != p.exe:
        paths.append(p.cmdline[0])
    return paths


def identity_stems(p: ProcInfo) -> set[str]:
    stems = {path_stem(x) for x in identity_paths(p)}
    if p.name:
        stems.add(path_stem(p.name))
    stems.discard("")
    return stems


def chromium_type(p: ProcInfo) -> str | None:
    # Chromium on Linux may rewrite argv into one space-joined string: scan tokens.
    for token in " ".join(p.cmdline).split():
        if token.startswith("--type="):
            return token.split("=", 1)[1]
    return None


def _cmd_option(p: ProcInfo, flag: str) -> str | None:
    args = list(p.cmdline)
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def is_under(path: str, dirs: tuple[str, ...]) -> bool:
    np = norm_path(path)
    for d in dirs:
        nd = norm_path(d).rstrip("/")
        if nd and (np == nd or np.startswith(nd + "/")):
            return True
    return False


def _under_install(p: ProcInfo, ctx: DiscoveryContext) -> bool:
    return any(is_under(x, ctx.install_dirs) for x in identity_paths(p))


def _is_pg_binary(p: ProcInfo) -> bool:
    return any(PG_BIN_MARKER in norm_path(x) for x in identity_paths(p))


def children_map(procs: list[ProcInfo]) -> dict[int, list[ProcInfo]]:
    by_pid = {p.pid: p for p in procs}
    out: dict[int, list[ProcInfo]] = {}
    for p in procs:
        if p.ppid == p.pid:
            continue
        parent = by_pid.get(p.ppid)
        # A PID can be reused (notably on Windows): a "child" older than its parent is not one.
        if parent and p.create_time and parent.create_time and p.create_time + 1 < parent.create_time:
            continue
        out.setdefault(p.ppid, []).append(p)
    return out


def descendants(root_pid: int, kids: dict[int, list[ProcInfo]]) -> set[int]:
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in kids.get(pid, []):
            if child.pid not in seen:
                seen.add(child.pid)
                stack.append(child.pid)
    return seen


def find_main_pid(procs: list[ProcInfo], ctx: DiscoveryContext) -> int | None:
    """The Electron main process: explicit PID, else the top-most Erudi binary without --type."""
    by_pid = {p.pid: p for p in procs}

    def is_main_binary(p: ProcInfo) -> bool:
        return (
            p.pid not in ctx.exclude_pids
            and chromium_type(p) is None
            and bool(identity_stems(p) & set(MAIN_EXE_STEMS))
            and (not ctx.install_dirs or _under_install(p, ctx))
        )

    # An explicit PID is what `quit` will signal: honour it only if it is this install's main binary.
    if ctx.main_pid is not None and ctx.main_pid in by_pid and is_main_binary(by_pid[ctx.main_pid]):
        return ctx.main_pid
    candidates = [p for p in procs if is_main_binary(p)]
    cand_pids = {p.pid for p in candidates}
    tops = [p for p in candidates if p.ppid not in cand_pids]
    if tops:
        return sorted(tops, key=lambda p: (p.create_time, p.pid))[0].pid
    # Fallback: parent of a bundled backend executable. The parent must itself
    # be a binary of this install: when the Electron main dies before the
    # backend (every quit traverses that window), the backend is re-parented to
    # launchd/init, whose pid would otherwise be returned as "the main" --
    # adopting every user process into membership and aiming quit's signal
    # outside the app tree.
    for p in procs:
        if BACKEND_STEM in identity_stems(p) and _under_install(p, ctx) and p.ppid in by_pid:
            parent = by_pid[p.ppid]
            if (
                parent.pid > 1
                and BACKEND_STEM not in identity_stems(parent)
                and (not ctx.install_dirs or _under_install(parent, ctx))
            ):
                return parent.pid
    return None


def select_members(procs: list[ProcInfo], ctx: DiscoveryContext, main_pid: int | None) -> set[int]:
    """PIDs that belong to this Erudi installation."""
    kids = children_map(procs)
    members: set[int] = set()
    if main_pid is not None:
        members.add(main_pid)
        members |= descendants(main_pid, kids)
    data_root = norm_path(ctx.data_root) if ctx.data_root else None
    for p in procs:
        if p.pid in ctx.exclude_pids or p.pid in members:
            continue
        adopt = False
        if _is_pg_binary(p):
            # Re-parented postmaster: only ours if it is the bundled binary or serves our data root.
            # A developer's pgserver cluster (site-packages/pgserver/pginstall) must not match.
            cmd = norm_path(" ".join(p.cmdline))
            adopt = _under_install(p, ctx) or bool(data_root and data_root in cmd) or p.ppid in members
        elif ctx.install_dirs and _under_install(p, ctx):
            adopt = True  # safety scan: anything executing from inside the install
        if adopt:
            members.add(p.pid)
            members |= descendants(p.pid, kids)
    return members - set(ctx.exclude_pids)


def find_backend_pids(procs: list[ProcInfo], members: set[int], main_pid: int | None, api_port: int) -> set[int]:
    by_pid = {p.pid: p for p in procs}
    out = set()
    for pid in members:
        p = by_pid[pid]
        if pid == main_pid or chromium_type(p) is not None or _is_pg_binary(p):
            continue
        parent = by_pid.get(p.ppid)
        stems = identity_stems(p)
        if BACKEND_STEM in stems and not (parent and BACKEND_STEM in identity_stems(parent)):
            out.add(pid)
        elif main_pid is not None and p.ppid == main_pid and _cmd_option(p, "--port") == str(api_port):
            out.add(pid)
    return out


def classify(
    p: ProcInfo,
    by_pid: dict[int, ProcInfo],
    main_pid: int | None,
    backend_pids: set[int],
) -> tuple[str, str]:
    """Category and role of one member process. Pure; see module docstring."""
    if p.pid == main_pid:
        return "electron_main", "main"
    ctype = chromium_type(p)
    if ctype is not None:
        if ctype == "renderer":
            return "electron_renderer", "renderer"
        if ctype == "gpu-process":
            return "electron_gpu", "gpu"
        sub = _cmd_option(p, "--utility-sub-type")
        return "electron_utility", ctype + (f":{sub.rsplit('.', 1)[-1]}" if sub else "")
    stems = identity_stems(p)
    if stems & set(CRASHPAD_STEMS):
        return "electron_utility", "crashpad"
    if _is_pg_binary(p):
        if "postgres" in stems:
            parent = by_pid.get(p.ppid)
            if parent is not None and _is_pg_binary(parent) and "postgres" in identity_stems(parent):
                return "database", "pg_child"
            return "database", "postmaster"
        tool = next((s for s in stems if s in PG_TOOL_STEMS), "pg_tool")
        return "transient", tool
    if "llama-server" in stems:
        return "inference", "llama_server"
    if p.pid in backend_pids:
        return "backend", "backend"
    if p.ppid in backend_pids:
        parent = by_pid[p.ppid]
        same_exe = bool(p.exe) and norm_path(p.exe) == norm_path(parent.exe)
        if same_exe:
            cmd = " ".join(p.cmdline)
            if any(marker in cmd for marker in MP_HELPER_MARKERS):
                return "backend", "mp_helper"
            # macOS MLX: a multiprocessing spawn of the frozen backend executable.
            return "inference", "mlx_child"
        return "transient", "backend_child"
    return "transient", "other"


def classify_snapshot(procs: list[ProcInfo], ctx: DiscoveryContext) -> tuple[int | None, dict[int, Classified]]:
    """Select this installation's processes and classify each. Returns (main_pid, {pid: Classified})."""
    by_pid = {p.pid: p for p in procs}
    main_pid = find_main_pid(procs, ctx)
    members = select_members(procs, ctx, main_pid)
    backend_pids = find_backend_pids(procs, members, main_pid, ctx.api_port)
    out = {}
    for pid in sorted(members):
        category, role = classify(by_pid[pid], by_pid, main_pid, backend_pids)
        out[pid] = Classified(pid, category, role, by_pid[pid])
    return main_pid, out


def take_snapshot() -> tuple[list[ProcInfo], dict[int, str]]:
    """All visible processes (psutil). Returns (records, {pid: error}) for unreadable fields."""
    import psutil

    procs: list[ProcInfo] = []
    errors: dict[int, str] = {}
    for proc in psutil.process_iter(["pid", "ppid", "name", "create_time", "status"]):
        info = proc.info
        if info.get("status") == psutil.STATUS_ZOMBIE:
            continue  # exited, only waiting to be reaped: it holds no memory and is not a survivor
        exe, cmdline = "", ()
        try:
            exe = proc.exe() or ""
        except (psutil.Error, OSError) as e:
            errors[info["pid"]] = f"exe: {type(e).__name__}"
        try:
            cmdline = tuple(proc.cmdline())
        except (psutil.Error, OSError) as e:
            errors[info["pid"]] = f"cmdline: {type(e).__name__}"
        procs.append(
            ProcInfo(
                pid=info["pid"],
                ppid=info.get("ppid") or 0,
                name=info.get("name") or "",
                exe=exe,
                cmdline=cmdline,
                create_time=info.get("create_time") or 0.0,
            )
        )
    return procs, errors
