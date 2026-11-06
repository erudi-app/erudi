"""Embedding-model device selection: prefer MPS, survive an unusable one (#335).

Apple Silicon is the primary desktop platform, so the embedding model must keep
loading on MPS whenever Metal actually works. But ``torch.backends.mps.is_available()``
answers "this build/OS supports MPS", not "an allocation will succeed": a
virtualised macOS host can report available and then refuse the 366 MiB
the model needs, and a real Mac under memory pressure fails
the same way. That failure is recoverable, so it must degrade to CPU with a
warning instead of taking the Knowledge Base down.

torch is mocked throughout, so these tests are meaningful on Linux CI too.
"""

import pytest

from src.ingestion import embedding_model as em

pytestmark = pytest.mark.unit

MPS_OOM_MESSAGE = (
    "MPS backend out of memory (MPS allocated: 0 bytes, other allocations: 16.00 KiB, "
    "max allowed: 7.93 GiB). Tried to allocate 366.27 MiB on shared pool."
)


@pytest.fixture(autouse=True)
def _fresh_device_verdict(monkeypatch):
    """Each test starts before any MPS verdict has been cached."""
    monkeypatch.setattr(em, "_MPS_UNUSABLE", False)


def _fake_mps(monkeypatch, available: bool):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


class _SpyModel:
    """Stands in for SentenceTransformer; optionally fails on a given device."""

    def __init__(self, fails_on: str | None = None):
        self.calls: list[dict] = []
        self._fails_on = fails_on
        self.error_message = MPS_OOM_MESSAGE

    def __call__(self, model_id, **kwargs):
        self.calls.append({"model_id": model_id, **kwargs})
        if self._fails_on is not None and kwargs.get("device") == self._fails_on:
            raise RuntimeError(self.error_message)
        return self

    @property
    def devices(self) -> list:
        return [call.get("device") for call in self.calls]


def _install(monkeypatch, spy: _SpyModel):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", spy)


class TestDeviceResolution:
    def test_mps_is_preferred_when_available(self, monkeypatch):
        _fake_mps(monkeypatch, True)

        assert em.resolve_embedding_device() == "mps"

    def test_without_mps_the_device_choice_is_left_to_sentence_transformers(self, monkeypatch):
        # None, not "cpu": a CUDA box must keep auto-selecting its GPU.
        _fake_mps(monkeypatch, False)

        assert em.resolve_embedding_device() is None

    def test_verdict_is_cached_after_a_failed_allocation(self, monkeypatch):
        _fake_mps(monkeypatch, True)
        monkeypatch.setattr(em, "_MPS_UNUSABLE", True)

        assert em.resolve_embedding_device() == "cpu"

    def test_missing_torch_backend_is_not_fatal(self, monkeypatch):
        import torch

        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert em.resolve_embedding_device() is None


class TestLoadFallback:
    def test_working_mps_keeps_mps(self, monkeypatch):
        _fake_mps(monkeypatch, True)
        spy = _SpyModel()
        _install(monkeypatch, spy)

        model = em.load_sentence_transformer(
            "intfloat/multilingual-e5-small", cache_folder="/cache", local_files_only=True
        )

        assert model is spy
        assert spy.devices == ["mps"]
        assert em._MPS_UNUSABLE is False

    def test_unusable_mps_falls_back_to_cpu(self, monkeypatch, caplog):
        _fake_mps(monkeypatch, True)
        spy = _SpyModel(fails_on="mps")
        _install(monkeypatch, spy)

        with caplog.at_level("WARNING"):
            model = em.load_sentence_transformer(
                "intfloat/multilingual-e5-small", cache_folder="/cache", local_files_only=True
            )

        assert model is spy  # no exception propagated
        assert spy.devices == ["mps", "cpu"]
        assert any("falling back to CPU" in record.message for record in caplog.records)

    def test_fallback_is_decided_once_not_per_load(self, monkeypatch):
        _fake_mps(monkeypatch, True)
        spy = _SpyModel(fails_on="mps")
        _install(monkeypatch, spy)

        em.load_sentence_transformer("m", cache_folder="/cache", local_files_only=True)
        em.load_sentence_transformer("m", cache_folder="/cache", local_files_only=True)

        # The second load goes straight to CPU: MPS is retried zero times.
        assert spy.devices == ["mps", "cpu", "cpu"]

    def test_no_mps_platform_is_unaffected(self, monkeypatch):
        _fake_mps(monkeypatch, False)
        spy = _SpyModel()
        _install(monkeypatch, spy)

        em.load_sentence_transformer("m", cache_folder="/cache", local_files_only=False)

        assert spy.devices == [None]
        assert em._MPS_UNUSABLE is False

    def test_an_unrelated_load_failure_still_raises(self, monkeypatch):
        _fake_mps(monkeypatch, True)
        spy = _SpyModel(fails_on="mps")
        spy.error_message = "We couldn't connect to 'https://huggingface.co'"
        _install(monkeypatch, spy)

        with pytest.raises(RuntimeError, match="huggingface"):
            em.load_sentence_transformer("m", cache_folder="/cache", local_files_only=False)

        assert em._MPS_UNUSABLE is False


class TestConsumersUseTheResolvedDevice:
    def test_download_path_goes_through_the_fallback_loader(self, monkeypatch):
        _fake_mps(monkeypatch, True)
        spy = _SpyModel(fails_on="mps")
        _install(monkeypatch, spy)
        monkeypatch.setattr(em, "embedding_model_available", lambda: False)

        em._load_model()

        assert spy.devices == ["mps", "cpu"]
        assert spy.calls[-1]["local_files_only"] is False

    def test_e5_embeddings_go_through_the_fallback_loader(self, monkeypatch):
        from src.ingestion import embeddings as embeddings_module
        from src.ingestion.embeddings import E5Embeddings

        _fake_mps(monkeypatch, True)
        spy = _SpyModel(fails_on="mps")
        _install(monkeypatch, spy)
        monkeypatch.setattr(embeddings_module, "embedding_model_available", lambda: True)
        monkeypatch.setattr(E5Embeddings, "_model", None)  # reset the resident singleton

        model = E5Embeddings._get_model()

        assert model is spy
        assert spy.devices == ["mps", "cpu"]
        assert spy.calls[-1]["local_files_only"] is True
