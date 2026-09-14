"""Build-time catalog snapshot (#112): serialize/load round-trip + first-boot
loading. The snapshot is the instant, zero-HF first-boot catalog; offline JSON is
the fallback. Offline unit tests — no network, no real DB."""

import json
import types
from unittest.mock import MagicMock

import pytest

from src.core import config
from src.database import catalog_snapshot as snap
from src.database import generation_hints as gh
from src.database import seed as seed_mod
from src.database.seed import Model_Seeder
from src.entities.Llm import Llm

_HINTS = {
    "base_repo": "Qwen/Qwen3-8B",
    "generation_config": {"temperature": 0.6, "top_k": 20},
    "supports_thinking": True,
    "context_length": 40960,
    "captured_at": "2026-08-28",
}


@pytest.fixture(autouse=True)
def _no_capture_cache():
    gh.reset_capture_cache()
    yield
    gh.reset_capture_cache()


def test_llm_dict_roundtrip():
    llm = Llm(
        local=0,
        name="Qwen3 8B",
        link="lmstudio/Qwen3-8B-MLX-4bit",
        type="qwen",
        quantized=True,
        model_metadata="meta",
        param_size=8.0,
        supports_tools=None,
        is_base=True,
        conversational=True,
        category="general",
        generation_hints=_HINTS,
        artifact_size_bytes=4_620_000_000,
    )
    d = snap.llm_to_dict(llm)
    assert d == {
        "name": "Qwen3 8B",
        "link": "lmstudio/Qwen3-8B-MLX-4bit",
        "type": "qwen",
        "quantized": True,
        "model_metadata": "meta",
        "param_size": 8.0,
        "supports_tools": None,
        "is_base": True,
        "conversational": True,
        "category": "general",
        "generation_hints": _HINTS,
        "artifact_size_bytes": 4_620_000_000,
    }
    back = snap.dict_to_llm(d)
    assert (back.local, back.name, back.link, back.type, back.param_size) == (
        0,
        "Qwen3 8B",
        "lmstudio/Qwen3-8B-MLX-4bit",
        "qwen",
        8.0,
    )
    assert back.is_base is True
    assert back.category == "general"
    assert back.generation_hints == _HINTS
    assert back.artifact_size_bytes == 4_620_000_000


def test_dict_to_llm_tolerates_absent_artifact_size_bytes():
    # The bundled snapshots predate the column: absent key or null = size unknown.
    assert (
        snap.dict_to_llm({"name": "X", "link": "y/z", "type": "qwen"}).artifact_size_bytes is None
    )
    assert (
        snap.dict_to_llm(
            {"name": "X", "link": "y/z", "type": "qwen", "artifact_size_bytes": None}
        ).artifact_size_bytes
        is None
    )


def test_dict_to_llm_tolerates_absent_generation_hints():
    # Pre-#388 snapshots carry no hints: None = "no hints" = the fallback sampling.
    assert snap.dict_to_llm({"name": "X", "link": "y/z", "type": "qwen"}).generation_hints is None
    assert (
        snap.dict_to_llm(
            {"name": "X", "link": "y/z", "type": "qwen", "generation_hints": None}
        ).generation_hints
        is None
    )


def test_dict_to_llm_defaults_is_base_false_when_missing():
    # Old snapshots predate the flag; loading one must not crash and defaults to derived.
    back = snap.dict_to_llm({"name": "X", "link": "y/z", "type": "qwen"})
    assert back.is_base is False


def test_dict_to_llm_keeps_unclassified_category_none():
    """#192: pre-#122 snapshots carry no category (or an explicit null). The loader
    must NOT coalesce to "general" — None is the "unclassified" sentinel that lets
    the boot reconcile keep the existing row's classification. Plain inserts still
    land on the Llm column default ("general")."""
    assert snap.dict_to_llm({"name": "X", "link": "y/z", "type": "qwen"}).category is None
    assert (
        snap.dict_to_llm({"name": "X", "link": "y/z", "type": "qwen", "category": None}).category
        is None
    )


def test_build_path_emits_classified_snapshot_entries(monkeypatch):
    """#192 root cause pin: the bundled snapshots were generated BEFORE #122 wired
    the classifier, so no entry carried a category and every boot reconciled the
    catalog down to "general". The generation path (discovery → creators →
    llm_to_dict, the exact chain generate_snapshot dumps) must consult the
    classifier and emit a real category on every entry."""
    calls = {"n": 0}
    real_categorize = seed_mod.categorize

    def spy(*a, **k):
        calls["n"] += 1
        return real_categorize(*a, **k)

    monkeypatch.setattr(seed_mod, "categorize", spy)

    class _Size:
        size_bytes = None

        def to_string(self):
            return "1 GB"

    monkeypatch.setattr(seed_mod, "format_model_info_metadata", lambda *a, **k: "meta")
    monkeypatch.setattr(seed_mod, "get_disk_size_after_quant", lambda *a, **k: _Size())

    api = MagicMock()
    api.list_models.side_effect = lambda **kw: (
        [types.SimpleNamespace(id="Qwen/Qwen3-Coder-8B", downloads=100000, tags=["conversational"])]
        if kw.get("pipeline_tag") == "text-generation"
        else []
    )
    api.model_info.return_value = object()
    seeder = Model_Seeder(db=None, hf_api=api)

    # Base path: discovery classifies; _create_base_llm carries it onto the row.
    (cfg,) = seeder.discover_instruct_models("Qwen", "qwen", min_downloads=1)
    base_entry = snap.llm_to_dict(seeder._create_base_llm(cfg, "quanter/Qwen3-Coder-8B-GGUF"))
    assert base_entry["category"] == "code"

    # Derived path: _create_derived_llm classifies from the community slug/tags.
    model_info = types.SimpleNamespace(
        id="community/foo-r1-GGUF", tags=[], pipeline_tag="text-generation"
    )
    search_config = types.SimpleNamespace(model_type="x", default_param_size=7.0)
    derived_entry = snap.llm_to_dict(seeder._create_derived_llm(model_info, search_config))
    assert derived_entry["category"] == "reasoning"

    assert calls["n"] >= 2  # the classifier was actually consulted


def _seeder_with_capture_spy(monkeypatch):
    """Model_Seeder over a stub HF api, with the hints capture spied (#388).

    The active engine is pinned to a GGUF one so the format tag the seeder
    threads into the capture (the engine side the row is built for) is the same
    on every platform."""
    captured = []

    def spy(base_repo, hf_api, quant_repo=None, quant_format=None):
        captured.append((base_repo, quant_repo, quant_format))
        return dict(_HINTS, base_repo=base_repo)

    class _GgufEngine:
        FORMAT_TAG = "gguf"

    monkeypatch.setattr(seed_mod.config, "LLM_Engine", _GgufEngine)
    monkeypatch.setattr(seed_mod, "capture_generation_hints", spy)

    class _Size:
        size_bytes = None

        def to_string(self):
            return "1 GB"

    monkeypatch.setattr(seed_mod, "format_model_info_metadata", lambda *a, **k: "meta")
    monkeypatch.setattr(seed_mod, "get_disk_size_after_quant", lambda *a, **k: _Size())
    api = MagicMock()
    api.model_info.return_value = object()
    return Model_Seeder(db=None, hf_api=api), captured


def test_base_creators_capture_hints_from_the_base_repo_with_the_quant_as_second_stage(monkeypatch):
    """#388: the snapshot generation path captures the BASE repo's generation
    facts (Model_Config.link) with the resolved quant link as the cascade's
    second stage, and carries them onto the row that llm_to_dict dumps."""
    seeder, captured = _seeder_with_capture_spy(monkeypatch)
    cfg = seed_mod.Model_Config(name="Qwen3-8B", link="Qwen/Qwen3-8B", model_type="qwen")

    entry = snap.llm_to_dict(seeder._create_base_llm(cfg, "lmstudio/Qwen3-8B-MLX-4bit"))
    assert entry["generation_hints"]["base_repo"] == "Qwen/Qwen3-8B"
    assert entry["generation_hints"]["generation_config"] == {"temperature": 0.6, "top_k": 20}

    fallback = snap.llm_to_dict(seeder._create_base_llm_fallback(cfg, "lmstudio/Qwen3-8B-MLX-4bit"))
    assert fallback["generation_hints"]["base_repo"] == "Qwen/Qwen3-8B"
    # The engine side travels with the pair: a GGUF row's window comes from the
    # quant's GGUF metadata, not the base repo's config.json.
    assert captured == [("Qwen/Qwen3-8B", "lmstudio/Qwen3-8B-MLX-4bit", "gguf")] * 2


def test_derived_creator_captures_from_the_base_model_tag_else_itself(monkeypatch):
    seeder, captured = _seeder_with_capture_spy(monkeypatch)
    search_config = types.SimpleNamespace(model_type="qwen", default_param_size=7.0)

    tagged = types.SimpleNamespace(
        id="mlx-community/Qwen3-8B-4bit",
        pipeline_tag="text-generation",
        tags=["mlx", "base_model:quantized:Qwen/Qwen3-8B"],
    )
    entry = snap.llm_to_dict(seeder._create_derived_llm(tagged, search_config))
    assert entry["generation_hints"]["base_repo"] == "Qwen/Qwen3-8B"

    untagged = types.SimpleNamespace(
        id="community/foo-GGUF", pipeline_tag="text-generation", tags=[]
    )
    entry = snap.llm_to_dict(seeder._create_derived_llm(untagged, search_config))
    assert entry["generation_hints"]["base_repo"] == "community/foo-GGUF"
    assert captured == [
        ("Qwen/Qwen3-8B", "mlx-community/Qwen3-8B-4bit", "gguf"),
        ("community/foo-GGUF", "community/foo-GGUF", "gguf"),
    ]


def test_capture_failure_leaves_hints_none(monkeypatch):
    seeder, _ = _seeder_with_capture_spy(monkeypatch)
    monkeypatch.setattr(seed_mod, "capture_generation_hints", lambda *a, **k: None)
    cfg = seed_mod.Model_Config(name="X", link="org/X", model_type="x")
    assert snap.llm_to_dict(seeder._create_base_llm(cfg, "q/X-GGUF"))["generation_hints"] is None


# ---------------------------------------------------------------------------
# Real artifact size at snapshot time. The frontend's per-parameter estimate
# misses a VLM's vision tower by 25-35 % (Qwen2.5-VL-3B: "~2.3 GB" shown, 3.09 GB
# downloaded), so the catalog carries the bytes the downloader would fetch.
# ---------------------------------------------------------------------------


class _ApiSize:
    """What get_disk_size_after_quant returns on an API hit."""

    size_bytes = 3_090_000_000

    def to_string(self):
        return "~3.1 GB"


class _EstimatedSize:
    size_bytes = None

    def to_string(self):
        return "~2.3 GB"


def _seeder_with_size_spy(monkeypatch, size):
    monkeypatch.setattr(seed_mod, "capture_generation_hints", lambda *a, **k: None)
    monkeypatch.setattr(seed_mod, "format_model_info_metadata", lambda *a, **k: "meta")
    sized = []

    def spy(link, hf_api=None):
        sized.append((link, hf_api))
        return size

    monkeypatch.setattr(seed_mod, "get_disk_size_after_quant", spy)
    api = MagicMock()
    api.model_info.return_value = object()
    return Model_Seeder(db=None, hf_api=api), sized, api


def test_base_creators_record_the_real_artifact_size_of_the_resolved_quant(monkeypatch):
    """The base row's size comes from the quant repo the resolver picked (the repo
    actually downloaded), through the seeder's own HF client: for MLX the whole
    single-artifact repo, for GGUF the chosen quant file (+ mmproj + aux) -- the
    same selection the downloader applies. Zero extra requests: this is the
    repo_info(files_metadata=True) call the Size line already paid for."""
    seeder, sized, api = _seeder_with_size_spy(monkeypatch, _ApiSize())
    cfg = seed_mod.Model_Config(
        name="Qwen2.5-VL-3B", link="Qwen/Qwen2.5-VL-3B-Instruct", model_type="qwen"
    )

    entry = snap.llm_to_dict(
        seeder._create_base_llm(cfg, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    )
    assert entry["artifact_size_bytes"] == 3_090_000_000

    fallback = snap.llm_to_dict(
        seeder._create_base_llm_fallback(cfg, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    )
    assert fallback["artifact_size_bytes"] == 3_090_000_000
    assert sized == [("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", api)] * 2


def test_derived_creator_records_the_real_artifact_size_of_the_community_repo(monkeypatch):
    """Community rows used to carry a full-precision family guess; they now pay
    the same one repo_info call as a base row and record the exact bytes."""
    seeder, sized, api = _seeder_with_size_spy(monkeypatch, _ApiSize())
    search_config = types.SimpleNamespace(model_type="qwen", default_param_size=7.0)
    hit = types.SimpleNamespace(
        id="bartowski/Qwen3-8B-GGUF", pipeline_tag="text-generation", tags=["gguf"]
    )

    entry = snap.llm_to_dict(seeder._create_derived_llm(hit, search_config))
    assert entry["artifact_size_bytes"] == 3_090_000_000
    assert sized == [("bartowski/Qwen3-8B-GGUF", api)]


def test_estimated_size_leaves_artifact_size_unknown(monkeypatch):
    """When HF cannot be asked, the Size line falls back to an estimate but the
    bytes column stays None: an estimate must never be stored as a measurement
    (the frontend keeps its own estimate as fallback)."""
    seeder, _, _ = _seeder_with_size_spy(monkeypatch, _EstimatedSize())
    cfg = seed_mod.Model_Config(name="X", link="org/X", model_type="x")
    assert snap.llm_to_dict(seeder._create_base_llm(cfg, "q/X-GGUF"))["artifact_size_bytes"] is None
    hit = types.SimpleNamespace(id="community/foo-GGUF", pipeline_tag="text-generation", tags=[])
    search_config = types.SimpleNamespace(model_type="x", default_param_size=7.0)
    assert (
        snap.llm_to_dict(seeder._create_derived_llm(hit, search_config))["artifact_size_bytes"]
        is None
    )


def test_load_missing_snapshot_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(snap.config, "ROOT_DIR", tmp_path)
    assert snap.load_catalog_snapshot("mlx") == []


def test_load_existing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(snap.config, "ROOT_DIR", tmp_path)
    db_dir = tmp_path / "src" / "database"
    db_dir.mkdir(parents=True)
    (db_dir / "catalog_snapshot_gguf.json").write_text(json.dumps([{"name": "x", "link": "y/z"}]))
    assert snap.load_catalog_snapshot("gguf") == [{"name": "x", "link": "y/z"}]


def test_seed_from_snapshot_inserts_local0_rows(monkeypatch):
    class _Eng:
        FORMAT_TAG = "mlx"

    monkeypatch.setattr(config, "LLM_Engine", _Eng)
    monkeypatch.setattr(
        snap,
        "load_catalog_snapshot",
        lambda tag: [
            {"name": "A", "link": "a/A", "type": "qwen"},
            {"name": "B", "link": "b/B", "type": "llama"},
        ],
    )
    db = MagicMock()
    n = Model_Seeder(db=db, hf_api=None).seed_from_snapshot()
    assert n == 2
    rows = db.add_all.call_args[0][0]
    assert [r.name for r in rows] == ["A", "B"]
    assert all(r.local == 0 for r in rows)


def test_seed_from_snapshot_no_tag_returns_zero(monkeypatch):
    class _Eng:
        FORMAT_TAG = None

    monkeypatch.setattr(config, "LLM_Engine", _Eng)
    assert Model_Seeder(db=MagicMock(), hf_api=None).seed_from_snapshot() == 0


def test_seed_initial_catalog_prefers_snapshot(monkeypatch):
    seeder = Model_Seeder(db=MagicMock(), hf_api=None)
    monkeypatch.setattr(seeder, "seed_from_snapshot", lambda: 42)
    tried_offline = {"v": False}

    def _offline():
        tried_offline["v"] = True
        return 7

    monkeypatch.setattr(seeder, "seed_base_models_offline", _offline)
    assert seeder.seed_initial_catalog() == 42
    assert tried_offline["v"] is False  # snapshot won → offline never tried


def test_seed_initial_catalog_falls_back_to_offline(monkeypatch):
    seeder = Model_Seeder(db=MagicMock(), hf_api=None)
    monkeypatch.setattr(seeder, "seed_from_snapshot", lambda: 0)
    monkeypatch.setattr(seeder, "seed_base_models_offline", lambda: 7)
    assert seeder.seed_initial_catalog() == 7


def test_seed_initial_catalog_never_raises_when_both_fail(monkeypatch):
    # Boot must not crash if neither a snapshot nor the fallback JSON is available.
    def _boom():
        raise RuntimeError("missing")

    seeder = Model_Seeder(db=MagicMock(), hf_api=None)
    monkeypatch.setattr(seeder, "seed_from_snapshot", _boom)
    monkeypatch.setattr(seeder, "seed_base_models_offline", _boom)
    assert seeder.seed_initial_catalog() == 0
