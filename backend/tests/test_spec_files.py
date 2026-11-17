"""Guard the PyInstaller specs against an mlx-vlm/mlx_lm regression.

The MLX bundle must collect `mlx_vlm` and hidden-import the mlx-vlm runner +
server, with no `mlx_lm.server` / `_mlx_server_runner` left behind. These
specs are only exercised at build time, so this text guard is the only
automated check that the swap reached them.
"""

import pathlib
import re

import pytest

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    path = _BACKEND / name
    if not path.exists():
        # Some specs (e.g. the CPU variant) are gitignored / build-time only.
        pytest.skip(f"{name} not present in this checkout")
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_mac_silicon_spec_collects_mlx_vlm():
    spec = _read("backend-mac-silicon.spec")
    assert 'collect_all("mlx_vlm")' in spec
    # mlx_lm IS collected — not as a server, but because mlx-vlm's text-only
    # path imports the per-architecture model module (mlx_lm.models.<arch>) by
    # name at runtime (bug 5). The swap guard is that the SERVER stays mlx-vlm.
    assert 'collect_all("mlx_lm")' in spec
    assert "src.engines._mlx_vlm_server_runner" in spec
    assert '"mlx_vlm.server"' in spec
    assert "_mlx_server_runner" not in spec
    assert '"mlx_lm.server"' not in spec


@pytest.mark.unit
def test_mac_silicon_spec_excludes_unused_heavy_deps():
    # Lazy mlx-vlm deps that the inference / vision-input path never imports.
    spec = _read("backend-mac-silicon.spec")
    assert '"cv2"' in spec
    assert '"mlx_audio"' in spec


@pytest.mark.unit
def test_mac_silicon_spec_bundles_py3langid_data():
    # The langid model (model.plzma) is package data loaded at runtime by
    # src/agents/language.py; without collecting it the systematic KB path
    # dies with FileNotFoundError (packaging bug 6).
    spec = _read("backend-mac-silicon.spec")
    assert 'collect_data_files("py3langid")' in spec


@pytest.mark.unit
def test_backend_spec_targets_mlx_vlm_not_mlx_lm():
    spec = _read("backend.spec")
    assert "mlx_vlm" in spec
    assert "mlx_lm" not in spec


@pytest.mark.unit
def test_all_specs_bundle_py3langid_data():
    # The systematic KB path (detect_language) is shared across platforms, so
    # every spec must collect py3langid's model data (packaging bug 6).
    for name in ("backend-mac-silicon.spec", "backend.spec"):
        spec = _read(name)
        assert 'collect_data_files("py3langid")' in spec, name


@pytest.mark.unit
def test_all_specs_collect_pgserver_binaries():
    # pgserver's embedded postgres binaries must be collected on every platform,
    # else the frozen backend dies at startup (packaging bug 1).
    for name in ("backend-mac-silicon.spec", "backend.spec"):
        spec = _read(name)
        assert 'collect_all("pgserver")' in spec, name


@pytest.mark.unit
def test_all_specs_bundle_alembic_migrations():
    # The startup migration reads alembic.ini + alembic/versions from the
    # filesystem (ROOT_DIR == bundle root when frozen), so every shipping spec must
    # bundle the alembic tree and collect alembic's dynamically-loaded submodules.
    for name in ("backend-mac-silicon.spec", "backend.spec"):
        spec = _read(name)
        assert 'spec_root / "alembic"' in spec, name
        assert "alembic.ini" in spec, name
        assert 'collect_submodules("alembic")' in spec, name


@pytest.mark.unit
def test_all_specs_pin_lazy_langchain_stacks():
    # The agent LangChain stack (deferred to the first turn) and the ingestion
    # splitter stack (deferred to KB ingestion time) are imported at FUNCTION
    # scope only (#160 lazy imports). PyInstaller's bytecode scan still sees
    # function-level imports today, but these hiddenimports pin the deferred
    # dependencies of a frozen build against any analysis change.
    needed = (
        '"langchain.agents"',
        '"langchain.agents.middleware"',
        '"langchain.tools"',
        '"langchain_core.messages"',
        '"langchain_core.tools"',
        '"langchain_openai"',
        '"langchain_text_splitters"',
        '"src.agents.middleware"',
        '"src.agents.kb_mode"',
        '"src.agents.tools"',
    )
    for name in ("backend-mac-silicon.spec", "backend.spec"):
        spec = _read(name)
        for entry in needed:
            assert entry in spec, f"{name} must hidden-import {entry} (#160)"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["backend.spec", "backend-mac-silicon.spec", "backend-cpu.spec"])
def test_no_spec_hidden_imports_pkg_resources(name):
    # Nothing in the backend or its runtime dependencies imports it, and a
    # build environment whose setuptools no longer ships it turned the entry
    # into `ERROR: Hidden import 'pkg_resources' not found` in the build log.
    assert not re.search(r"[\"']pkg_resources[\"']", _read(name))


@pytest.mark.unit
def test_cpu_spec_excludes_mlx_vlm():
    spec = _read("backend-cpu.spec")
    assert "mlx_vlm" in spec
    assert "mlx_lm" not in spec


@pytest.mark.unit
def test_backend_spec_fails_the_build_when_the_inference_binary_is_missing():
    """A release without llama-server is an app that cannot infer at all.

    The spec used to only `warnings.warn` when the artifact directory was
    absent, so a silently failed llama build would still produce a bundle and
    still publish. Nothing downstream checked, so the defect would surface as
    "the app does nothing" on a user's machine. The collection helper raises
    (tests/test_pyi_llama_artifacts.py); the spec must hand it the opt-out
    rather than swallow the error.
    """
    spec = _read("backend.spec")
    assert "warnings.warn" not in spec, "a missing inference binary must fail the build, not warn"
    assert "collect_llama_artifacts(" in spec
    # The boot-only smoke build legitimately has no llama-server, so there is
    # one explicit, named opt-out - and it has to be spelled out in the
    # workflow that uses it rather than being the default.
    assert "ERUDI_ALLOW_MISSING_INFERENCE_BINARY" in spec
    assert "except RuntimeError" not in spec


@pytest.mark.unit
def test_smoke_workflow_is_the_only_place_that_opts_out():
    """The opt-out belongs to the boot-only gate, never to a release."""
    root = _BACKEND.parent
    smoke = (root / ".github/workflows/app-build-smoke.yml").read_text(encoding="utf-8")
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "ERUDI_ALLOW_MISSING_INFERENCE_BINARY" in smoke
    assert "ERUDI_ALLOW_MISSING_INFERENCE_BINARY" not in release


@pytest.mark.unit
def test_release_asserts_the_bundle_actually_contains_the_binary():
    """Belt and braces: the spec can only see what it was handed.

    PyInstaller could succeed and still drop the file, so the release verifies
    the frozen bundle on disk before it is packaged and published.
    """
    root = _BACKEND.parent
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "Verify the bundled inference binary" in release


@pytest.mark.unit
def test_backend_spec_bundles_the_llama_artifacts_through_the_helper():
    # What ships is decided by pyi_llama_artifacts.collect_llama_artifacts (unit
    # tested on every OS): the spec passes it the build variant and appends its
    # output -- datas before Analysis, the runtime libraries verbatim after the
    # torch-CUDA strip. No flavour path is hardcoded in the spec.
    spec = _read("backend.spec")
    assert "from pyi_llama_artifacts import collect_llama_artifacts" in spec
    assert "ERUDI_BUILD_VARIANT" in spec
    assert ".runtime_files" in spec
    assert '"llama-cpp" / "cuda" / "bin"' not in spec
    assert '"llama-cpp" / "cpu" / "bin"' not in spec
    # The runtime libraries are appended after the strip, never ridden in datas.
    strip = spec.index("_cuda_fragments")
    append = spec.index('a.datas.append((_name, _src, "DATA"))')
    assert append > strip


def _release_steps():
    import yaml  # a transformers dependency, installed on every leg

    root = _BACKEND.parent
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    return workflow["jobs"]["build"]["steps"]


def _step(steps, name):
    (step,) = [s for s in steps if s.get("name") == name]
    return step


@pytest.mark.unit
def test_release_compiles_the_cpu_flavour_on_the_cuda_legs():
    """The CUDA installers ship the CPU llama-server for the processor engine.

    Its compile is a separate step gated on the CUDA legs, after the CUDA
    compile and before PyInstaller, one per OS.
    """
    steps = _release_steps()
    names = [s.get("name") for s in steps]
    win = _step(steps, "Build the CPU llama-server too (Windows CUDA leg)")
    linux = _step(steps, "Build the CPU llama-server too (Linux CUDA leg)")
    assert win["if"] == "matrix.llama_build == 'cuda' && runner.os == 'Windows'"
    assert win["run"].strip() == "./scripts/dev/backend/build-llamacpp-cpu-win.ps1"
    assert linux["if"] == "matrix.llama_build == 'cuda' && runner.os == 'Linux'"
    assert linux["run"].strip() == "bash scripts/dev/backend/build-llamacpp-cpu-linux.sh"
    bundle = names.index("Build backend bundle")
    for step, primary in (
        (win, "Build llama-server (Windows inference binary)"),
        (linux, "Build llama-server (Linux inference binary)"),
    ):
        assert names.index(primary) < names.index(step["name"]) < bundle


@pytest.mark.unit
def test_release_verifies_both_flavours_in_the_cuda_bundle():
    """The Verify step checks each flavour at its explicit path.

    A single `find -print -quit` would be satisfied by either copy; the CPU
    copy is started with --version on every leg (it needs no driver), and must
    not carry the CUDA runtime (a CPU binary that loads cudart is the CUDA
    build in the wrong folder).
    """
    run = _step(_release_steps(), "Verify the bundled inference binary")["run"]
    assert "-print -quit" not in run
    assert "llama-cpp/cpu/bin" in run
    assert "llama-cpp/cuda/bin" in run
    assert '"$cpu_exe" --version' in run
    assert "cudart64_12.dll" in run
    assert 'if [ -f "$cpu_dir/cudart64_12.dll" ]' in run


@pytest.mark.unit
def test_specs_collect_docx_as_pyz_and_source_files():
    # python-docx loads default-styles.xml (and the four other default parts)
    # through os.path.join(<docx/parts dir>, "..", "templates", ...). A `.docx`
    # saved without a styles part (TextEdit) reaches that path. In the frozen
    # app docx/parts/ is not a directory (its modules live in the PYZ), so the
    # OS cannot resolve the `..` on macOS or Linux (only Windows normalises it
    # lexically) and the read fails with ENOENT (#705).
    # "pyz+py" also writes the sources, which makes docx/parts/ exist on disk.
    # backend-cpu.spec executes backend.spec in its own namespace, so the three
    # specs carry the one option as long as the two Analysis calls do.
    option = 'module_collection_mode={"docx": "pyz+py"}'
    assert option in _read("backend.spec")
    assert option in _read("backend-mac-silicon.spec")
    assert "backend.spec" in _read("backend-cpu.spec")


@pytest.mark.unit
def test_every_spec_ships_the_frozen_e5_tokenizer():
    # KB chunking and retrieval count tokens with the e5 tokenizer frozen in
    # the repository; src/ingestion/chunking.py reads it at
    # ROOT_DIR/src/ingestion/e5_tokenizer.json.gz, so a bundle without it at
    # that path cannot ingest a document. backend-cpu.spec executes
    # backend.spec, so the two Analysis inputs cover the three specs.
    entry = (
        'datas.append((str(spec_root / "src" / "ingestion" / "e5_tokenizer.json.gz"),'
        ' "src/ingestion"))'
    )
    assert entry in _read("backend.spec")
    assert entry in _read("backend-mac-silicon.spec")
    assert "backend.spec" in _read("backend-cpu.spec")
    assert (_BACKEND / "src" / "ingestion" / "e5_tokenizer.json.gz").is_file()


@pytest.mark.unit
def test_release_and_smoke_check_the_frozen_bundle_ships_the_docx_parts():
    # The spec option is only text; what proves it is the frozen tree on disk.
    # Both the release build and the boot gate run the check right after
    # PyInstaller, on every leg (it only walks the directory).
    import yaml

    root = _BACKEND.parent
    for workflow, job in (("release.yml", "build"), ("app-build-smoke.yml", None)):
        data = yaml.safe_load((root / ".github/workflows" / workflow).read_text(encoding="utf-8"))
        jobs = data["jobs"]
        steps = jobs[job]["steps"] if job else next(iter(jobs.values()))["steps"]
        names = [s.get("name") for s in steps]
        step = _step(steps, "Check the frozen backend ships python-docx's default parts")
        assert "scripts/ci/check_docx_default_parts.py backend/dist" in step["run"], workflow
        assert "if" not in step, workflow
        freeze = next(
            i for i, s in enumerate(steps) if "pyinstaller backend/" in str(s.get("run", ""))
        )
        assert freeze < names.index(step["name"]), workflow


@pytest.mark.unit
def test_mac_app_signature_is_verified_in_smoke_and_before_the_release_publishes():
    # Without a certificate electron-builder signs the main executable only, which
    # `codesign --verify --deep --strict` rejects and macOS 26 refuses to launch
    # (#690). The smoke leg has it sign every file ad hoc (`-c.mac.identity=-`,
    # which electron-builder skips on a pull request without
    # CSC_FOR_PULL_REQUEST) and verifies; the release builds with no publish,
    # verifies the signed and notarized app, and only then publishes it.
    import yaml

    root = _BACKEND.parent
    script = "scripts/ci/check_mac_app_signature.sh"
    smoke = yaml.safe_load(
        (root / ".github/workflows/app-build-smoke.yml").read_text(encoding="utf-8")
    )
    job = next(iter(smoke["jobs"].values()))
    mac = next(
        leg for leg in job["strategy"]["matrix"]["include"] if leg["dist_script"] == "dist:mac"
    )
    assert mac["eb_sign_args"] == "-c.mac.identity=-"
    steps = job["steps"]
    names = [s.get("name") for s in steps]
    build = _step(steps, "Build packaged app (unsigned, no publish)")
    assert "matrix.eb_sign_args" in build["run"]
    assert build["env"]["CSC_FOR_PULL_REQUEST"] == "true"
    check = _step(steps, "Check the macOS app signature")
    assert check["if"] == "runner.os == 'macOS'"
    assert check["run"].strip() == f"bash {script} frontend/dist/mac-arm64/Erudi.app"
    assert names.index(build["name"]) < names.index(check["name"])

    release = yaml.safe_load((root / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    steps = release["jobs"]["build"]["steps"]
    names = [s.get("name") for s in steps]
    generic = _step(steps, "Build, sign and publish (electron-builder)")
    assert generic["if"] == "matrix.variant != 'mac-arm'"
    build = _step(steps, "Build and sign (electron-builder, macOS)")
    assert "--publish never" in build["run"]
    assert "--publish always" not in build["run"]
    verify = _step(steps, "Verify the macOS app signature")
    assert verify["run"].strip() == f"bash {script} --release frontend/dist/mac-arm64/Erudi.app"
    publish = _step(steps, "Publish the macOS artifacts (electron-builder)")
    assert "--prepackaged dist/mac-arm64/Erudi.app" in publish["run"]
    assert "--publish always" in publish["run"]
    for step in (build, verify, publish):
        assert step["if"] == "matrix.variant == 'mac-arm'"
    assert names.index(build["name"]) < names.index(verify["name"]) < names.index(publish["name"])


@pytest.mark.unit
def test_mac_signature_script_only_verifies_and_the_local_build_signs_ad_hoc():
    root = _BACKEND.parent
    text = (root / "scripts/ci/check_mac_app_signature.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "codesign --verify --deep --strict" in code
    assert "codesign --force" not in code
    assert "--sign" not in code
    assert "Authority=Developer ID Application:" in code
    assert "TeamIdentifier=$RELEASE_TEAM_ID" in code
    assert 'RELEASE_TEAM_ID="FAF8JCAT86"' in code
    assert "xcrun stapler validate" in code
    assert "runtime" in code

    build = (root / "scripts/build/build-mac-silicon.sh").read_text(encoding="utf-8")
    assert '"-c.mac.identity=-"' in build
    assert 'npm run dist:mac -- ${EB_SIGN_ARGS[@]+"${EB_SIGN_ARGS[@]}"}' in build
    assert "codesign --force" not in build
    assert "--prepackaged" not in build
    assert "dist/mac-arm64/Erudi.app" in build
    assert build.index("npm run dist:mac") < build.index("check_mac_app_signature.sh")
