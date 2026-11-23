"""A failed download stores a stable ``error_code`` beside its English message (#780).

The download widget shows the user a sentence in the interface language, keyed
on the code; the English ``error_message`` stays for the log and Diagnostics.
The code is derived from the exception where the failure is recorded
(``download_error_code``) and persisted on the job row.

The Hub is faked; the download task, the job row and Postgres are real.
"""

from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from src.core import config
from src.domains.llms import endpoints as llm_endpoints
from src.domains.llms import services as llm_services
from src.domains.llms.schemas import DownloadJobResponse
from src.domains.llms.download_errors import DOWNLOAD_ERROR_CODES, download_error_code
from tests.test_download_cancel import _RealDownload

URL = "https://huggingface.co/api/models/org/model"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", URL))


def _api(files, *, tags=("mlx",), repo_info_error=None):
    api = MagicMock()
    if repo_info_error is not None:
        api.repo_info.side_effect = repo_info_error
    else:
        api.repo_info.return_value = SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=f, size=16) for f in files],
            tags=list(tags),
            library_name=None,
        )
    api.list_repo_files.return_value = list(files)
    return api


class _Fs:
    """A transfer that writes the file, or raises ``error`` on the first one."""

    def __init__(self, error=None):
        self.error = error

    def get_file(self, remote, dest, callback):
        if self.error is not None:
            raise self.error
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"\x00" * 16)
        callback.relative_update(16)


@pytest.fixture
def real_download(monkeypatch, test_db_engine, tmp_path):
    harness = _RealDownload(monkeypatch, test_db_engine, tmp_path)
    yield harness
    harness.drop()


def _hub(monkeypatch, api, fs):
    monkeypatch.setattr(llm_services, "HfApi", lambda token=None: api)
    monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)


FILES = ["config.json", "model.safetensors"]

FAILURES = {
    # No network: huggingface_hub's resolver error, renamed by download_llm.
    "offline": lambda: (_api(FILES), _Fs(OSError("[Errno 11001] getaddrinfo failed"))),
    "disk_full": lambda: (
        _api(FILES),
        _Fs(OSError(errno.ENOSPC, "No space left on device")),
    ),
    "gated": lambda: (
        _api(FILES, repo_info_error=GatedRepoError("Access restricted", response=_response(401))),
        _Fs(),
    ),
    "not_found": lambda: (
        _api(
            FILES,
            repo_info_error=RepositoryNotFoundError(
                "Repository not found", response=_response(404)
            ),
        ),
        _Fs(),
    ),
    "rate_limited": lambda: (
        _api(FILES, repo_info_error=HfHubHTTPError("Too many requests", response=_response(429))),
        _Fs(),
    ),
    "no_engine_artifact": lambda: (_api(FILES, tags=("transformers",)), _Fs()),
    "unexpected": lambda: (_api(FILES), _Fs(RuntimeError("something nobody planned for"))),
}


@pytest.mark.integration
class TestAFailedDownloadPersistsItsCode:
    @pytest.mark.parametrize("code", sorted(FAILURES))
    def test_the_job_row_carries_the_code_and_the_english_message(
        self, real_download, monkeypatch, code
    ):
        api, fs = FAILURES[code]()
        _hub(monkeypatch, api, fs)

        real_download.run()

        job = real_download.job()
        assert job.status == "failed"
        assert job.error_code == code
        # The English text stays, for the log and Diagnostics.
        assert job.error_message
        # And the status the widget polls carries it.
        assert DownloadJobResponse.model_validate(job).error_code == code

    def test_a_plain_401_is_gated(self, real_download, monkeypatch):
        error = HfHubHTTPError("Unauthorized", response=_response(401))
        _hub(monkeypatch, _api(FILES, repo_info_error=error), _Fs())

        real_download.run()

        assert real_download.job().error_code == "gated"

    def test_an_artifact_that_fails_the_integrity_gate_is_integrity_failed(
        self, real_download, monkeypatch
    ):
        _hub(monkeypatch, _api(FILES), _Fs())

        real_download.run()

        job = real_download.job()
        assert job.status == "failed"
        assert job.error_code == "integrity_failed"
        assert "incomplete or invalid" in job.error_message


@pytest.mark.unit
class TestDownloadErrorCode:
    """The refusals raised before any transfer, classified one by one."""

    def test_every_code_is_declared(self):
        assert set(FAILURES) | {"integrity_failed", "interrupted"} <= set(DOWNLOAD_ERROR_CODES)

    def test_raw_gguf_chunks(self, monkeypatch):
        monkeypatch.setattr(
            config, "LLM_Engine", SimpleNamespace(USES_GGUF=True, FORMAT_TAG="gguf")
        )
        with pytest.raises(Exception) as caught:
            llm_services._assert_repo_has_engine_artifact(
                "org/model", None, ["model-Q4_K_M.gguf.part1of2", "model-Q4_K_M.gguf.part2of2"]
            )
        assert download_error_code(caught.value) == "raw_gguf_chunks"

    def test_no_engine_artifact(self, monkeypatch):
        monkeypatch.setattr(
            config, "LLM_Engine", SimpleNamespace(USES_GGUF=True, FORMAT_TAG="gguf")
        )
        with pytest.raises(Exception) as caught:
            llm_services._assert_repo_has_engine_artifact("org/model", None, ["model.safetensors"])
        assert download_error_code(caught.value) == "no_engine_artifact"

    def test_incomplete_split(self):
        with pytest.raises(Exception) as caught:
            llm_services._assert_split_is_complete(
                "model-Q4_K_M-00001-of-00003.gguf", ["model-Q4_K_M-00001-of-00003.gguf"]
            )
        assert download_error_code(caught.value) == "incomplete_split"

    def test_unsafe_file_name(self, tmp_path):
        with pytest.raises(Exception) as caught:
            llm_services.resolve_member_path(tmp_path, "../outside.gguf")
        assert download_error_code(caught.value) == "unsafe_file_name"

    def test_not_runnable(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: False, __name__="CPU_Engine"),
        )
        with pytest.raises(Exception) as caught:
            llm_services._assert_runnable("org/model")
        assert download_error_code(caught.value) == "not_runnable"

    def test_a_disk_full_error_wrapped_in_another_is_still_disk_full(self):
        try:
            try:
                raise OSError(errno.ENOSPC, "No space left on device")
            except OSError as inner:
                raise RuntimeError("the transfer failed") from inner
        except RuntimeError as outer:
            assert download_error_code(outer) == "disk_full"

    def test_anything_else_is_unexpected(self):
        assert download_error_code(ValueError("nope")) == "unexpected"


@pytest.mark.unit
def test_a_failed_progress_write_records_a_code(monkeypatch):
    """The progress thread's own failure path writes a code too."""
    from src.domains.llms import repository as llm_repo_mod

    dbj = SimpleNamespace(
        status="running",
        error_message=None,
        error_code=None,
        temp_local_model_link="",
        final_local_model_link="",
    )
    session = MagicMock()
    session.query.return_value.get.return_value = dbj
    session.commit.side_effect = [RuntimeError("database went away"), None]
    monkeypatch.setattr(llm_repo_mod, "SessionLocal", lambda: session)
    tracker = SimpleNamespace(
        percent=10.0, total_bytes=16, eta_seconds=1, should_continue=lambda: True
    )
    monkeypatch.setattr(llm_repo_mod.time, "sleep", lambda *_: None)

    llm_repo_mod.update_db_with_progress(tracker, 1, 1)

    assert dbj.status == "failed"
    assert dbj.error_code == "unexpected"


@pytest.mark.unit
def test_the_download_task_writes_the_code_with_the_failure(monkeypatch):
    job = SimpleNamespace(status="pending", error_message=None, updated_at=None)
    session = MagicMock()
    session.query.return_value.get.return_value = job
    monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: session)

    async def disk_full(**kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(llm_endpoints, "download_llm", disk_full)

    llm_endpoints._run_download_task("org/model", 1, "/tmp/t", "/tmp/f", job_id=9)

    values = session.query.return_value.filter.return_value.update.call_args_list[1].args[0]
    assert values["status"] == "failed"
    assert values["error_code"] == "disk_full"
    assert "No space left" in values["error_message"]
