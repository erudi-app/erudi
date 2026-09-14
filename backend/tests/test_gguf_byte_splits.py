"""GGUF weights published as raw byte chunks are never selected, sized or downloaded.

llama.cpp's own split convention is ``<name>-NNNNN-of-MMMMM.gguf``: every part is a
valid GGUF file and the loader stitches them together. Some uploaders instead cut
one big ``.gguf`` into byte-level pieces (``-chunk-001-of-018.gguf``,
``.gguf.part1of2``, ``.gguf.a``, ``.gguf-split-a``). Only the first piece carries a
GGUF header, so nothing can load them until they are concatenated, which Erudi
does not do.

``stevescot1979/DeepSeek-V3.2-MXFP4-GGUF`` is the real case (#524): the resolver
picked it for DeepSeek V3.2, the catalog sized one 21.5 GB chunk of a 366 GB model,
and a download would have fetched that one unloadable chunk. The listing below is
that repo's, as the Hub reports it.

All HuggingFace API calls are faked; no network access occurs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub import ModelInfo

from src.core import config
from src.core.exceptions import InvalidInputException
from src.database import seed as seed_mod
from src.database.seed import Model_Seeder, Search_Config
from src.domains.llms import services as llm_services
from src.domains.llms.services import (
    _select_download_files,
    has_downloadable_gguf,
    is_byte_split_gguf,
    pick_best_gguf,
)
from src.engines.cpu_engine import CPU_Engine
from src.utils.hf_model_metadata import _chosen_artifact_bytes

pytestmark = pytest.mark.unit

CHUNK_BYTES = 21_474_836_480

# The Hub listing of stevescot1979/DeepSeek-V3.2-MXFP4-GGUF (name, size in bytes).
DEEPSEEK_V32_BYTE_SPLIT_LISTING = (
    [(".gitattributes", 2_923)]
    + [(f"DeepSeek-V3.2-MXFP4-chunk-{n:03d}-of-018.gguf", CHUNK_BYTES) for n in range(1, 18)]
    + [("DeepSeek-V3.2-MXFP4-chunk-018-of-018.gguf", 1_186_389_088), ("README.md", 3_118)]
)
DEEPSEEK_V32_BYTE_SPLIT_FILES = [name for name, _size in DEEPSEEK_V32_BYTE_SPLIT_LISTING]


class Fake_GGUF_Engine:
    """llama.cpp stand-in (CPU/CUDA): consumes .gguf files."""

    USES_GGUF = True
    FORMAT_TAG = "gguf"

    @classmethod
    def is_runnable(cls, model_link: str) -> bool:
        return True


class TestIsByteSplitGguf:
    @pytest.mark.parametrize(
        "filename",
        [
            "DeepSeek-V3.2-MXFP4-chunk-001-of-018.gguf",
            "DeepSeek-V3.2-MXFP4-chunk-018-of-018.gguf",
            "model-chunk-1-of-2.gguf",
            "sub/dir/model-CHUNK-0001-OF-0003.GGUF",
            "Llama-2-70b-chat-hf.i1-Q6_K.gguf.part1of2",
            "Llama-2-70b-chat-hf.i1-Q6_K.gguf.part2of2",
            "Llama-3.1-70B-Instruct-Q5_K_M.gguf.a",
            "Llama-3.1-70B-Instruct-Q5_K_M.gguf.b",
            "falcon-40b-f16.gguf-split-a",
            "falcon-40b-f16.gguf-split-b",
            "FALCON-40B-F16.GGUF-SPLIT-A",
        ],
    )
    def test_byte_split_conventions_match(self, filename):
        assert is_byte_split_gguf(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
            "zai-org.GLM-5.3-Flash.Q4_K_M.gguf-00001-of-00015.gguf",
            "Q4_K_M/DeepSeek-V3.2-Q4_K_M-00001-of-00009.gguf",
            "model-Q4_K_M.gguf",
            "gemma-3-4b-it-q4_0.gguf",
            "mmproj-model-f16.gguf",
            "imatrix_unsloth.gguf_file",
            "model.gguf.json",
            "README.md",
        ],
    )
    def test_loadable_files_do_not_match(self, filename):
        assert is_byte_split_gguf(filename) is False


class TestSelectionSkipsByteSplits:
    def test_repo_with_a_byte_split_and_a_normal_quant_selects_the_normal_one(self):
        file_sizes = {
            "model-Q4_K_M-chunk-001-of-002.gguf": 4_000_000_000,
            "model-Q4_K_M-chunk-002-of-002.gguf": 1_000_000_000,
            "model-Q8_0.gguf": 8_000_000_000,
            "config.json": 2_000,
        }
        files = list(file_sizes)

        assert pick_best_gguf(files) == "model-Q8_0.gguf"
        selection = _select_download_files(files, file_sizes, uses_gguf=True)
        assert selection.best_gguf == "model-Q8_0.gguf"
        assert set(selection.files) == {"model-Q8_0.gguf", "config.json"}

    def test_repo_with_only_byte_splits_selects_nothing(self):
        file_sizes = dict(DEEPSEEK_V32_BYTE_SPLIT_LISTING)

        assert pick_best_gguf(DEEPSEEK_V32_BYTE_SPLIT_FILES) is None
        selection = _select_download_files(
            DEEPSEEK_V32_BYTE_SPLIT_FILES, file_sizes, uses_gguf=True
        )
        assert selection.best_gguf is None
        assert selection.files == []

    def test_byte_split_projector_is_never_selected(self):
        file_sizes = {
            "model-Q4_K_M.gguf": 4_000_000_000,
            "mmproj-model-f16.gguf": 600_000_000,
            "mmproj-model-f32-chunk-001-of-002.gguf": 700_000_000,
            "mmproj-model-f32-chunk-002-of-002.gguf": 500_000_000,
        }
        selection = _select_download_files(list(file_sizes), file_sizes, uses_gguf=True)
        assert selection.mmproj_files == ["mmproj-model-f16.gguf"]

    def test_small_byte_split_piece_is_not_taken_as_an_auxiliary_file(self):
        file_sizes = {
            "model-Q4_K_M.gguf": 400_000_000,
            "tiny-f16.gguf.a": 4_000_000,
            "tiny-f16.gguf.b": 3_000_000,
            "config.json": 2_000,
        }
        selection = _select_download_files(list(file_sizes), file_sizes, uses_gguf=True)
        assert set(selection.files) == {"model-Q4_K_M.gguf", "config.json"}


class TestHasDownloadableGguf:
    def _api(self, files):
        api = MagicMock()
        api.model_info.return_value = SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=name) for name in files]
        )
        return api

    def test_true_for_a_complete_loadable_quant(self):
        api = self._api(["model-Q4_K_M.gguf", "README.md"])
        assert has_downloadable_gguf("org/model-GGUF", api) is True

    def test_false_for_byte_splits_only(self):
        api = self._api(DEEPSEEK_V32_BYTE_SPLIT_FILES)
        assert has_downloadable_gguf("stevescot1979/DeepSeek-V3.2-MXFP4-GGUF", api) is False

    def test_false_for_a_standard_split_with_a_missing_part(self):
        # The downloader refuses this listing too (_assert_split_is_complete).
        api = self._api(["model-q4_k_m-00001-of-00002.gguf"])
        assert has_downloadable_gguf("org/model-GGUF", api) is False

    def test_reads_the_listing_of_the_named_repo(self):
        api = self._api(["model-Q4_K_M.gguf"])
        has_downloadable_gguf("org/model-GGUF", api)
        args, kwargs = api.model_info.call_args
        assert args == ("org/model-GGUF",)
        assert kwargs == {"expand": ["siblings"]}

    def test_listing_failure_propagates_to_the_caller(self):
        api = MagicMock()
        api.model_info.side_effect = RuntimeError("hub down")
        with pytest.raises(RuntimeError, match="hub down"):
            has_downloadable_gguf("org/model-GGUF", api)


class TestSizingOfTheDeepSeekByteSplitRepo:
    def test_catalog_size_is_never_one_chunk(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", Fake_GGUF_Engine)
        repo_info = SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in DEEPSEEK_V32_BYTE_SPLIT_LISTING
            ]
        )

        total = _chosen_artifact_bytes(repo_info)

        # Nothing is selectable, so the size is the whole repository -- the real
        # 366 GB -- and never the first 21.5 GB chunk.
        assert total == sum(size for _name, size in DEEPSEEK_V32_BYTE_SPLIT_LISTING)
        assert total > 360_000_000_000


class _RecordingFs:
    """Fake HfFileSystem that records every transfer request."""

    def __init__(self):
        self.requested = []

    def get_file(self, remote, dest, callback):
        self.requested.append(remote)


class TestDownloadRefusesByteSplits:
    def test_guard_names_the_raw_chunks(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", Fake_GGUF_Engine)
        info = SimpleNamespace(tags=["gguf"], library_name=None)
        with pytest.raises(InvalidInputException) as exc_info:
            llm_services._assert_repo_has_engine_artifact(
                "stevescot1979/DeepSeek-V3.2-MXFP4-GGUF", info, DEEPSEEK_V32_BYTE_SPLIT_FILES
            )
        message = str(exc_info.value.message)
        assert "stevescot1979/DeepSeek-V3.2-MXFP4-GGUF" in message
        assert "raw chunks" in message
        assert "does not join them" in message
        assert message.isascii()

    def test_repo_without_any_gguf_keeps_the_generic_message(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", Fake_GGUF_Engine)
        info = SimpleNamespace(tags=["transformers"], library_name="transformers")
        with pytest.raises(InvalidInputException) as exc_info:
            llm_services._assert_repo_has_engine_artifact(
                "org/model", info, ["config.json", "model.safetensors"]
            )
        message = str(exc_info.value.message)
        assert "no gguf artefact" in message
        assert "raw chunks" not in message

    async def test_download_llm_fetches_nothing(self, monkeypatch, tmp_path):
        async def _instant(self, interval=20.0):
            return None

        monkeypatch.setattr(llm_services.DownloadTracker, "monitor_eta", _instant)
        monkeypatch.setattr(config, "LLM_Engine", Fake_GGUF_Engine)
        api = MagicMock()
        api.repo_info.return_value = SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in DEEPSEEK_V32_BYTE_SPLIT_LISTING
            ],
            tags=["gguf"],
            library_name=None,
        )
        api.list_repo_files.return_value = list(DEEPSEEK_V32_BYTE_SPLIT_FILES)
        fs = _RecordingFs()
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: api)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        with pytest.raises(InvalidInputException, match="raw chunks"):
            await llm_services.download_llm(
                model_link="stevescot1979/DeepSeek-V3.2-MXFP4-GGUF",
                model_id=1,
                temp_save_dir=str(tmp_path / "temp_1"),
                final_save_dir=str(tmp_path / "1"),
                job_id=None,
            )

        assert fs.requested == []
        assert not any((tmp_path / "temp_1").iterdir())


def _hit(model_id: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        pipeline_tag="text-generation",
        tags=["gguf", "text-generation", "conversational"],
        downloads=123456,
        likes=789,
        gated=False,
    )


class TestDerivedCatalogSkipsByteSplitRepos:
    def _seeder(self, monkeypatch, hits, listings):
        monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
        listed = []

        def model_info(repo_id, **kwargs):
            listed.append(repo_id)
            listing = listings[repo_id]
            if isinstance(listing, Exception):
                raise listing
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=f) for f in listing])

        api = SimpleNamespace(list_models=lambda **kw: list(hits), model_info=model_info)
        return Model_Seeder(db=None, hf_api=api), listed

    def _build(self, seeder):
        return seeder.build_derived_models(
            [Search_Config(search_term="", model_type="community", default_param_size=7.0)]
        )

    def test_byte_split_only_repo_is_skipped_and_its_twin_still_gets_in(self, monkeypatch):
        seeder, _listed = self._seeder(
            monkeypatch,
            [
                _hit("stevescot1979/DeepSeek-V3.2-MXFP4-GGUF"),
                _hit("unsloth/DeepSeek-V3.2-GGUF"),
                _hit("bartowski/Qwen3-8B-GGUF"),
            ],
            {
                "stevescot1979/DeepSeek-V3.2-MXFP4-GGUF": DEEPSEEK_V32_BYTE_SPLIT_FILES,
                "unsloth/DeepSeek-V3.2-GGUF": [
                    f"Q4_K_M/DeepSeek-V3.2-Q4_K_M-0000{n}-of-00009.gguf" for n in range(1, 10)
                ],
                "bartowski/Qwen3-8B-GGUF": ["Qwen3-8B-Q4_K_M.gguf"],
            },
        )

        rows = self._build(seeder)

        # Skipped BEFORE the normalized-key dedup, so the loadable twin of the
        # same model is not shadowed by a row that was never built.
        assert [r.link for r in rows] == ["unsloth/DeepSeek-V3.2-GGUF", "bartowski/Qwen3-8B-GGUF"]

    def test_repo_whose_listing_fails_is_skipped(self, monkeypatch):
        seeder, _listed = self._seeder(
            monkeypatch,
            [_hit("bartowski/Broken-8B-GGUF"), _hit("bartowski/Qwen3-8B-GGUF")],
            {
                "bartowski/Broken-8B-GGUF": RuntimeError("hub down"),
                "bartowski/Qwen3-8B-GGUF": ["Qwen3-8B-Q4_K_M.gguf"],
            },
        )

        assert [r.link for r in self._build(seeder)] == ["bartowski/Qwen3-8B-GGUF"]
