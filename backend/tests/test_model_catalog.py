"""Catalog ↔ engine invariants (#95), post auto-catalog refactor.

The catalog is built ONLY from repos carrying the engine's format tag (filter="mlx"
/ filter="gguf"), and base ids are resolved to their public quant by the resolver
(see test_model_resolver). So there is no hand-maintained MODEL_MAPPING and no
name-based allowlist: everything in the catalog is runnable by construction, and
the lone exclusion is KNOWN_BROKEN (quants that load-crash). These are pure
class-attribute / schema checks — no network, CI-safe.
"""

import pytest

from src.core import config
from src.core.exceptions import UnsupportedPlatformException
from src.database.seed import Database_Seeder
from src.domains.llms import services
from src.utils.hf_model_metadata import humanize_model_name
from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.mlx_engine import MLX_Engine

# Representative base ids (the live catalog is org-discovered, not a static list).
CATALOG_LINKS = [
    "google/gemma-3-270m-it",
    "google/gemma-2-2b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
]


class TestEngineFormatTag:
    """Each engine declares its HF format tag; the community/base search filters on
    it (across all of HF, any author) instead of a hand-maintained mapping/org."""

    def test_format_tag_per_engine(self):
        assert MLX_Engine.FORMAT_TAG == "mlx"
        assert BaseLlamaCppEngine.FORMAT_TAG == "gguf"
        assert CPU_Engine.FORMAT_TAG == "gguf"
        assert CUDA_Engine.FORMAT_TAG == "gguf"

    def test_uses_gguf_is_inherited_true(self):
        assert BaseLlamaCppEngine.USES_GGUF is True
        assert CPU_Engine.USES_GGUF is True
        assert CUDA_Engine.USES_GGUF is True

    def test_community_search_filters_on_format_tag(self):
        kw = MLX_Engine.community_search_kwargs("gemma 1b")
        assert kw["filter"] == "mlx" and kw["search"] == "gemma 1b"
        for engine in (CPU_Engine, CUDA_Engine):
            kw = engine.community_search_kwargs("gemma 1b")
            assert kw["filter"] == "gguf" and kw["search"] == "gemma 1b"


class TestOrgDiscovery:
    """discover_instruct_models is permissive but drops quant/adapter/non-final
    noise, applies a downloads floor, dedups by normalized slug, and caps the count."""

    def _seeder(self, ids):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.database.seed import Model_Seeder

        api = MagicMock()
        # Real chat releases carry HF's `conversational` tag; discovery requires it
        # for the Base catalog (#182). Fixtures model chat models, so attach it (the
        # non-chat noise here — quants, OCR — is dropped by earlier filters anyway).
        models = [SimpleNamespace(id=i, downloads=d, tags=["conversational"]) for i, d in ids]

        # Discovery now queries per pipeline_tag (text + vision passes) and per sort
        # (downloads + freshness, #305). These fixtures are plain text models ranked
        # by downloads, so only the text-generation downloads pass returns them; the
        # vision and freshness passes return nothing (freshness has dedicated tests).
        def _list(**kwargs):
            if (
                kwargs.get("pipeline_tag") == "text-generation"
                and kwargs.get("sort") == "downloads"
            ):
                return models
            return []

        api.list_models.side_effect = _list
        return Model_Seeder(db=None, hf_api=api)

    def test_filters_quants_and_floor(self):
        seeder = self._seeder(
            [
                ("Qwen/Qwen3-8B", 100000),  # keep
                ("Qwen/Qwen3-8B-GGUF", 50000),  # skip: quant
                ("Qwen/Qwen3-4B", 30000),  # keep
                ("Qwen/Qwen2.5-7B-Instruct-AWQ", 9000),  # skip: quant
                ("google/gemma-4-12B-it-qat-w4a16-ct", 80000),  # skip: QAT quant, not the base
                (
                    "google/diffusiongemma-26B-A4B-it",
                    70000,
                ),  # skip: diffusion (image-gen), not a chat LLM
                ("Qwen/tiny-thing", 10),  # skip: below floor
            ]
        )
        links = [
            c.link for c in seeder.discover_instruct_models("Qwen", "qwen", min_downloads=1000)
        ]
        assert "Qwen/Qwen3-8B" in links and "Qwen/Qwen3-4B" in links
        assert all(x not in " ".join(links) for x in ("GGUF", "AWQ", "qat", "diffusion"))
        assert "Qwen/tiny-thing" not in links

    def test_dedups_by_normalized_slug(self):
        seeder = self._seeder(
            [
                ("Qwen/Qwen3-8B", 100000),
                ("Qwen/Qwen3-8B-bf16", 90000),  # same normalized slug → dropped
            ]
        )
        configs = seeder.discover_instruct_models("Qwen", "qwen", min_downloads=1)
        assert len(configs) == 1

    def test_caps_top_n(self):
        seeder = self._seeder([(f"Org/Model-{i}-Instruct", 100000 - i) for i in range(20)])
        assert len(seeder.discover_instruct_models("Org", "x", top_n=5, min_downloads=1)) == 5

    def test_search_failure_returns_empty(self):
        from unittest.mock import MagicMock
        from src.database.seed import Model_Seeder

        api = MagicMock()
        api.list_models.side_effect = RuntimeError("boom")
        assert Model_Seeder(db=None, hf_api=api).discover_instruct_models("Org", "x") == []

    def _seeder_by_pipeline(self, mapping):
        """Mock list_models to return a different list per pipeline_tag (text/vision)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.database.seed import Model_Seeder

        api = MagicMock()

        def fake_list(**kw):
            # Chat releases carry the `conversational` tag; discovery requires it (#182).
            # Downloads pass only — the freshness pass (#305) has dedicated tests below.
            if kw.get("sort") != "downloads":
                return []
            return [
                SimpleNamespace(id=i, downloads=d, tags=["conversational"])
                for i, d in mapping.get(kw.get("pipeline_tag"), [])
            ]

        api.list_models.side_effect = fake_list
        return Model_Seeder(db=None, hf_api=api)

    def test_skips_assistant_distillates(self):
        # #122: the real foundation model is the multimodal -it VLM (discovered under a
        # vision pipeline); the text-only -assistant distillate must NOT be a base model.
        seeder = self._seeder_by_pipeline(
            {
                "text-generation": [("google/gemma-4-E4B-it-assistant", 352943)],
                "image-text-to-text": [("google/gemma-4-E4B-it", 5677536)],
                "any-to-any": [],
            }
        )
        links = [
            c.link for c in seeder.discover_instruct_models("google", "gemma", min_downloads=1000)
        ]
        assert "google/gemma-4-E4B-it" in links  # real foundation VLM surfaces
        assert "google/gemma-4-E4B-it-assistant" not in links  # text-only distillate skipped

    def test_discovers_across_pipelines_deduped(self):
        # #122: foundation models span modalities — discover across pipelines, count a
        # repo once (highest downloads kept) even when it lists under several tags.
        seeder = self._seeder_by_pipeline(
            {
                "text-generation": [("Qwen/Qwen3-8B", 100000)],
                "image-text-to-text": [("Qwen/Qwen3-VL-8B", 80000), ("Qwen/Qwen3-8B", 100000)],
                "any-to-any": [],
            }
        )
        links = [
            c.link for c in seeder.discover_instruct_models("Qwen", "qwen", min_downloads=1000)
        ]
        assert links.count("Qwen/Qwen3-8B") == 1  # same repo across pipelines → once
        assert "Qwen/Qwen3-VL-8B" in links  # multimodal foundation included

    def test_filters_ocr_and_florence_but_keeps_chat_models(self):
        # #203: OCR / Florence-2 are vision-TASK models that list under image-text-to-text
        # (like real chat VLMs) yet cannot be chatted with. They must be dropped while a
        # plain chat model and a real chat VLM in the same passes survive. The 'ocr' token
        # is a substring so it also catches the fused 'olmOCR' slug (no dash before OCR).
        seeder = self._seeder_by_pipeline(
            {
                "text-generation": [("Qwen/Qwen3-8B", 900000)],  # keep: plain chat
                "image-text-to-text": [
                    ("deepseek-ai/DeepSeek-OCR-2", 500000),  # drop: OCR
                    ("allenai/olmOCR-2-7B-1025", 400000),  # drop: OCR (fused token)
                    ("microsoft/Florence-2-large-ft", 300000),  # drop: vision-task foundation
                    ("Qwen/Qwen3-VL-8B-Instruct", 200000),  # keep: real chat VLM
                ],
                "any-to-any": [],
            }
        )
        links = [c.link for c in seeder.discover_instruct_models("org", "x", min_downloads=1000)]
        assert "Qwen/Qwen3-8B" in links and "Qwen/Qwen3-VL-8B-Instruct" in links
        assert not any("ocr" in link.lower() for link in links)  # no OCR leaks (#203)
        assert not any("florence" in link.lower() for link in links)  # no vision-task leak

    def test_drops_raw_pretrain_keeps_conversational(self):
        # #182: the Base catalog is chat-only. A raw pretrain (no `conversational`
        # tag, no instruct suffix) is dropped; its instruct sibling and a suffix-less
        # chat model (tagged conversational) are kept.
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.database.seed import Model_Seeder

        api = MagicMock()
        api.list_models.side_effect = lambda **kw: (
            [
                SimpleNamespace(id="meta-llama/Llama-3.2-1B", downloads=500000, tags=[]),
                SimpleNamespace(
                    id="meta-llama/Llama-3.2-1B-Instruct", downloads=900000, tags=["conversational"]
                ),
                SimpleNamespace(
                    id="deepseek-ai/DeepSeek-V3", downloads=800000, tags=["conversational"]
                ),
            ]
            if kw.get("pipeline_tag") == "text-generation"
            else []
        )
        links = [
            c.link
            for c in Model_Seeder(db=None, hf_api=api).discover_instruct_models(
                "x", "y", min_downloads=1
            )
        ]
        assert "meta-llama/Llama-3.2-1B" not in links  # raw pretrain dropped
        assert "meta-llama/Llama-3.2-1B-Instruct" in links  # instruct sibling kept
        assert "deepseek-ai/DeepSeek-V3" in links  # suffix-less chat kept via tag


class TestFreshnessPass:
    """#305: a downloads-only ranking structurally hides recent releases (established
    checkpoints accumulate CI downloads a new model can't match for months). Discovery
    therefore runs a second, creation-date-sorted pass per org/pipeline with its own
    small quota and a higher downloads floor. Fresh candidates go through the SAME
    rejection rules and dedup as the downloads pass, and ADD to (never displace) the
    downloads-ranked quota."""

    def _seeder(self, by_downloads, by_created):
        """Fake HF api keyed on the `sort` kwarg: `by_downloads` feeds the classic
        pass, `by_created` feeds the freshness pass (both text-generation only)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.database.seed import Model_Seeder

        def mk(rows):
            return [SimpleNamespace(id=i, downloads=d, tags=list(t)) for i, d, t in rows]

        api = MagicMock()

        def _list(**kw):
            if kw.get("pipeline_tag") != "text-generation":
                return []
            if kw.get("sort") == "downloads":
                return mk(by_downloads)
            if kw.get("sort") == "created_at":
                return mk(by_created)
            raise AssertionError(f"unexpected sort: {kw.get('sort')!r}")

        api.list_models.side_effect = _list
        return Model_Seeder(db=None, hf_api=api)

    _CHAT = ("conversational",)

    def test_recent_release_missing_from_downloads_top_is_discovered(self):
        # The downloads top is saturated with old checkpoints; the recent release
        # (high downloads, absent from that top) must surface via the freshness pass
        # WITHOUT displacing any downloads-ranked model.
        old_top = [(f"Qwen/Qwen-old-{i}-Instruct", 20_000_000 - i, self._CHAT) for i in range(3)]
        seeder = self._seeder(
            by_downloads=old_top,
            by_created=[("Qwen/Qwen3.8-27B-Instruct", 1_000_000, self._CHAT)] + old_top,
        )
        links = [c.link for c in seeder.discover_instruct_models("Qwen", "qwen", top_n=3)]
        assert "Qwen/Qwen3.8-27B-Instruct" in links
        for link, _, _ in old_top:
            assert link in links  # freshness ADDS; the downloads quota is untouched

    def test_recent_release_below_freshness_floor_is_not_discovered(self):
        # Below FRESH_MIN_DOWNLOADS the fresh candidate is noise, even though it
        # clears the regular min_downloads floor of the downloads pass.
        seeder = self._seeder(
            by_downloads=[("Qwen/Qwen3-8B", 20_000_000, self._CHAT)],
            by_created=[("Qwen/Qwen-Brand-New-3B-Instruct", 10_000, self._CHAT)],
        )
        links = [c.link for c in seeder.discover_instruct_models("Qwen", "qwen")]
        assert "Qwen/Qwen-Brand-New-3B-Instruct" not in links
        assert "Qwen/Qwen3-8B" in links

    def test_freshness_pass_respects_rejection_rules(self):
        # The freshness pass merges BEFORE filtering: a fresh quant/adapter derivative,
        # a raw pretrain, and an OCR model are still rejected on that pass too.
        seeder = self._seeder(
            by_downloads=[("Qwen/Qwen3-8B", 20_000_000, self._CHAT)],
            by_created=[
                ("Qwen/Qwen3.9-9B-Instruct-GGUF", 900_000, self._CHAT),  # artifact token
                ("Qwen/Qwen3.9-9B", 800_000, ()),  # raw pretrain
                ("Qwen/Qwen3.9-OCR-9B", 700_000, self._CHAT),  # non-chat task
                ("Qwen/Qwen3.9-9B-Instruct", 600_000, self._CHAT),  # keep
            ],
        )
        links = [c.link for c in seeder.discover_instruct_models("Qwen", "qwen")]
        assert links == ["Qwen/Qwen3-8B", "Qwen/Qwen3.9-9B-Instruct"]

    def test_model_in_both_passes_appears_once(self):
        seeder = self._seeder(
            by_downloads=[("Qwen/Qwen3-8B", 20_000_000, self._CHAT)],
            by_created=[("Qwen/Qwen3-8B", 20_000_000, self._CHAT)],
        )
        links = [c.link for c in seeder.discover_instruct_models("Qwen", "qwen")]
        assert links == ["Qwen/Qwen3-8B"]

    def test_freshness_quota_is_bounded(self):
        # The fresh pass has its own small cap (FRESH_TOP_N); a flood of recent
        # releases can't blow up the catalog.
        from src.database.seed import FRESH_TOP_N

        fresh = [
            (f"Org/New-{i}-Instruct", 500_000 - i, self._CHAT) for i in range(FRESH_TOP_N + 10)
        ]
        seeder = self._seeder(by_downloads=[], by_created=fresh)
        assert len(seeder.discover_instruct_models("Org", "x")) == FRESH_TOP_N


class TestRunnabilityPredicate:
    """is_runnable is now KNOWN_BROKEN-only: catalog entries are engine-format by
    construction (they came from a filter=FORMAT_TAG search), so the only thing to
    exclude is a quant that downloads but crashes at load."""

    def test_format_quants_are_runnable(self):
        assert MLX_Engine.is_runnable("mlx-community/gemma-3-1b-it-4bit") is True
        assert MLX_Engine.is_runnable("lmstudio-community/Hermes-4-70B-MLX-4bit") is True  # non-org
        assert CPU_Engine.is_runnable("unsloth/gemma-3-1b-it-GGUF") is True
        assert CUDA_Engine.is_runnable("mradermacher/Some-Cool-Distill-GGUF") is True  # community

    def test_known_broken_entries_are_not_runnable(self, monkeypatch):
        # The roster is empty since the 0.6.13 bump re-proved gemma-4 E2B loads
        # (#273); pin the MECHANISM with an injected entry, not a real model.
        monkeypatch.setattr(
            MLX_Engine, "KNOWN_BROKEN", frozenset({"mlx-community/broken-quant-4bit"})
        )
        assert MLX_Engine.is_runnable("mlx-community/broken-quant-4bit") is False

    def test_llama_cpp_known_broken_roster_is_empty(self):
        # llama.cpp (CPU/CUDA) never had known-broken quants.
        assert CPU_Engine.KNOWN_BROKEN == frozenset()

    def test_deepseek_vl_v1_and_nemotron_v1_quants_are_not_runnable(self):
        # DeepSeek-VL v1 (multi_modality/vision.py hard-imports cv2, not
        # bundled -- #512) and the Nemotron-NAS v1 quants (tokenizer_config.json
        # declares the abstract PreTrainedTokenizer -- #506) are declared
        # not runnable on MLX.
        broken = [
            "mlx-community/deepseek-vl-1.3b-chat-4bit",
            "mlx-community/deepseek-vl-1.3b-chat-8bit",
            "mlx-community/deepseek-vl-7b-chat-4bit",
            "mlx-community/deepseek-vl-7b-chat-8bit",
            "mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit",
            "mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-6bit",
        ]
        for link in broken:
            assert MLX_Engine.is_runnable(link) is False, link

    def test_deepseek_vl2_and_nemotron_v1_5_siblings_stay_runnable(self):
        # deepseek-vl2 does not hard-import cv2 (#512), and the v1_5 Nemotron
        # quant declares PreTrainedTokenizerFast, not the abstract base class
        # (#506) -- neither is affected, so both stay offered.
        assert MLX_Engine.is_runnable("mlx-community/deepseek-vl2-small-4bit") is True
        assert (
            MLX_Engine.is_runnable("mlx-community/Llama-3_3-Nemotron-Super-49B-v1_5-mlx-4Bit")
            is True
        )


class TestHumanizedNames:
    """Display names are derived from the real slug — exact, unambiguous."""

    @pytest.mark.parametrize(
        "link,expected",
        [
            ("google/gemma-3-270m-it", "Gemma 3 270M Instruct"),
            ("google/gemma-2-2b-it", "Gemma 2 2B Instruct"),
            ("google/gemma-3-4b-it", "Gemma 3 4B Instruct"),
            ("google/gemma-3-12b-it", "Gemma 3 12B Instruct"),
            ("google/gemma-4-E2B-it", "Gemma 4 E2B Instruct"),
            ("google/gemma-4-26b-a4b-it", "Gemma 4 26B A4B Instruct"),
            ("google/gemma-4-31b-it", "Gemma 4 31B Instruct"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3"),
            ("mistralai/Ministral-8B-Instruct-2410", "Ministral 8B Instruct 2410"),
            ("mistralai/Mistral-Nemo-Instruct-2407", "Mistral Nemo Instruct 2407"),
            ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct"),
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct"),
        ],
    )
    def test_humanize(self, link, expected):
        assert humanize_model_name(link) == expected

    def test_no_catalog_name_is_ambiguous_gemma(self):
        for link in CATALOG_LINKS:
            name = humanize_model_name(link)
            if name.startswith("Gemma"):
                assert name.split()[1].replace(".", "").isdigit(), f"ambiguous: {name}"


class TestRunnableExposedInResponse:
    """LLMResponse surfaces a computed `runnable` so the UI can disable/tag models."""

    def test_remote_quant_is_runnable(self, monkeypatch):
        from src.domains.llms.schemas import LLMResponse

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)
        r = LLMResponse(id=1, name="X", local=0, link="unsloth/gemma-3-1b-it-GGUF")
        assert r.runnable is True
        assert r.model_dump()["runnable"] is True

    def test_remote_known_broken_is_not_runnable(self, monkeypatch):
        from src.domains.llms.schemas import LLMResponse

        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        # Roster empty since #273; pin the mechanism with an injected entry.
        monkeypatch.setattr(
            MLX_Engine, "KNOWN_BROKEN", frozenset({"mlx-community/broken-quant-4bit"})
        )
        r = LLMResponse(id=2, name="Y", local=0, link="mlx-community/broken-quant-4bit")
        assert r.runnable is False

    def test_deepseek_vl_v1_quant_reports_not_runnable(self, monkeypatch):
        """#512: DeepSeek-VL v1 quants are declared not runnable on MLX, and
        the catalog response schema's computed `runnable` field reflects it."""
        from src.domains.llms.schemas import LLMResponse

        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        r = LLMResponse(
            id=4, name="DeepSeek VL 1.3B", local=0, link="mlx-community/deepseek-vl-1.3b-chat-4bit"
        )
        assert r.runnable is False
        assert r.model_dump()["runnable"] is False

    def test_downloaded_model_always_runnable(self, monkeypatch):
        from src.domains.llms.schemas import LLMResponse

        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        # local=1 (downloaded) → runnable even if its link were KNOWN_BROKEN
        r = LLMResponse(id=3, name="Z", local=1, link="/data/models/3")
        assert r.runnable is True


class TestReconcileRemoteCatalog:
    """reconcile_remote_catalog reconciles local=0 with a fresh model set (now the
    bundled snapshot — #131/#163) atomically: it preserves downloaded (local=1) and
    in-progress (local=2) models, updates survivors in place (stable ids), inserts
    new rows and deletes vanished ones."""

    def _llm(self, **kw):
        from src.entities.Llm import Llm

        kw.setdefault("type", "x")
        kw.setdefault("quantized", False)
        kw.setdefault("param_size", 1.0)
        kw.setdefault("is_base", False)  # real fresh models always carry the flag
        return Llm(**kw)

    def test_reconcile_replaces_remote_keeps_downloaded(self, test_db_session):
        from src.entities.Llm import Llm

        db = test_db_session
        db.add_all(
            [
                self._llm(name="Old Stale", local=0, link="stale/gone-GGUF"),
                self._llm(name="My Model", local=1, link="/data/models/9"),
                self._llm(name="Downloading", local=2, link="repo/wip-GGUF"),
            ]
        )
        db.commit()

        fresh = self._llm(
            name="Gemma 3 1B Instruct", local=0, link="unsloth/gemma-3-1b-it-GGUF", type="gemma"
        )
        res = Database_Seeder().reconcile_remote_catalog(db, [fresh], [])

        assert res["resynced"] is True
        rows = {r.link: r.local for r in db.query(Llm).all()}
        assert "stale/gone-GGUF" not in rows
        assert rows["unsloth/gemma-3-1b-it-GGUF"] == 0
        assert rows["/data/models/9"] == 1
        assert rows["repo/wip-GGUF"] == 2

    def test_derived_requant_of_base_is_deduped(self):
        """build_fresh_catalog (shared with the build-time snapshot generator) drops
        derived rows that are just another quant of a base model, keeps finetunes."""
        from unittest.mock import MagicMock

        base = self._llm(
            name="Gemma 3 1B Instruct", local=0, link="unsloth/gemma-3-1b-it-GGUF", type="gemma"
        )
        # another quant of the SAME base (normalizes to gemma-3-1b-it) → must drop
        requant = self._llm(
            name="dup", local=0, link="bartowski/google_gemma-3-1b-it-GGUF", type="gemma"
        )
        # a genuine finetune (different slug) → must stay
        finetune = self._llm(
            name="Dolphin",
            local=0,
            link="cognitivecomputations/dolphin-gemma-3-1b-GGUF",
            type="gemma",
        )
        ms = MagicMock()
        ms.build_base_models.return_value = [base]
        ms.build_derived_models.return_value = [requant, finetune]

        fresh_base, fresh_derived = Database_Seeder().build_fresh_catalog(ms)
        links = {m.link for m in fresh_base + fresh_derived}
        assert "unsloth/gemma-3-1b-it-GGUF" in links  # base kept
        assert "bartowski/google_gemma-3-1b-it-GGUF" not in links  # re-quant of base dropped
        assert "cognitivecomputations/dolphin-gemma-3-1b-GGUF" in links  # finetune kept

    def test_baseless_snapshot_does_not_wipe_catalog(self, test_db_session, monkeypatch):
        """A snapshot with no base models (broken/partial artifact) must not wipe
        the existing catalog — the reconcile no-ops."""
        from src.database import catalog_snapshot as snap_mod
        from src.entities.Llm import Llm

        db = test_db_session
        db.add(self._llm(name="Keep Me", local=0, link="keep/me-GGUF"))
        db.commit()

        monkeypatch.setattr(config, "LLM_Engine", type("_Eng", (), {"FORMAT_TAG": "gguf"}))
        monkeypatch.setattr(
            snap_mod,
            "load_catalog_snapshot",
            lambda tag: [
                {"name": "d", "link": "x/d-GGUF", "type": "x", "param_size": 1.0, "is_base": False}
            ],
        )

        res = Database_Seeder().reconcile_catalog_from_snapshot(db)
        assert res["resynced"] is False
        assert db.query(Llm).filter(Llm.link == "keep/me-GGUF").count() == 1

    def test_reconcile_keeps_ids_stable_in_place(self, test_db_session):
        """#123: a model still in the snapshot keeps its catalog id across reconciles
        (in-place update), and its fields are refreshed — no delete+reinsert churn."""
        from src.entities.Llm import Llm

        db = test_db_session
        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(
                    name="Gemma 3 1B Instruct",
                    local=0,
                    link="unsloth/gemma-3-1b-it-GGUF",
                    type="gemma",
                    param_size=1.0,
                )
            ],
            [],
        )
        first_id = db.query(Llm).filter(Llm.link == "unsloth/gemma-3-1b-it-GGUF").one().id

        # Second reconcile: SAME link, refreshed name + param_size.
        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(
                    name="Gemma 3 1B Instruct (refreshed)",
                    local=0,
                    link="unsloth/gemma-3-1b-it-GGUF",
                    type="gemma",
                    param_size=1.5,
                )
            ],
            [],
        )

        again = db.query(Llm).filter(Llm.link == "unsloth/gemma-3-1b-it-GGUF").one()
        assert again.id == first_id  # id stable → in-place update
        assert again.name == "Gemma 3 1B Instruct (refreshed)"
        assert again.param_size == 1.5

    def test_reconcile_keeps_category_when_fresh_row_is_unclassified(self, test_db_session):
        """#192 (regression of #184): fresh rows rebuilt from a pre-#122 snapshot
        carry category=None — the in-place refresh must KEEP the existing row's
        classification instead of clobbering it with the default bucket."""
        from src.entities.Llm import Llm

        db = test_db_session
        Database_Seeder().reconcile_remote_catalog(
            db, [self._llm(name="Coder", local=0, link="org/coder-GGUF", category="code")], []
        )

        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(name="Coder", local=0, link="org/coder-GGUF")  # category unset → None
            ],
            [],
        )

        row = db.query(Llm).filter(Llm.link == "org/coder-GGUF").one()
        assert row.category == "code"

    def test_reconcile_updates_category_from_classified_fresh_row(self, test_db_session):
        """A fresh set that carries a REAL category still propagates it in place."""
        from src.entities.Llm import Llm

        db = test_db_session
        Database_Seeder().reconcile_remote_catalog(
            db, [self._llm(name="Model", local=0, link="org/model-GGUF", category="code")], []
        )

        Database_Seeder().reconcile_remote_catalog(
            db, [self._llm(name="Model", local=0, link="org/model-GGUF", category="reasoning")], []
        )

        row = db.query(Llm).filter(Llm.link == "org/model-GGUF").one()
        assert row.category == "reasoning"

    def test_reconcile_inserts_unclassified_fresh_row_as_general(self, test_db_session):
        """Coalescing an unknown category to "general" applies ONLY at insert."""
        from src.entities.Llm import Llm

        db = test_db_session
        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(name="New", local=0, link="org/new-GGUF")  # category unset → None
            ],
            [],
        )

        row = db.query(Llm).filter(Llm.link == "org/new-GGUF").one()
        assert row.category == "general"

    def test_reconcile_inserts_new_and_deletes_disappeared(self, test_db_session):
        """#123: only new models are inserted, only models gone from the fresh set
        are deleted, and survivors keep their ids."""
        from src.entities.Llm import Llm

        db = test_db_session
        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(name="Keep", local=0, link="org/keep-GGUF", type="gemma"),
                self._llm(name="Gone", local=0, link="org/gone-GGUF", type="gemma"),
            ],
            [],
        )
        keep_id = db.query(Llm).filter(Llm.link == "org/keep-GGUF").one().id

        # Next fresh set: 'gone' disappeared, 'fresh' appeared, 'keep' survives.
        Database_Seeder().reconcile_remote_catalog(
            db,
            [
                self._llm(name="Keep", local=0, link="org/keep-GGUF", type="gemma"),
                self._llm(name="Fresh", local=0, link="org/fresh-GGUF", type="gemma"),
            ],
            [],
        )

        rows = {r.link: r.id for r in db.query(Llm).filter(Llm.local == 0).all()}
        assert "org/gone-GGUF" not in rows  # disappeared → deleted
        assert "org/fresh-GGUF" in rows  # new → inserted
        assert rows["org/keep-GGUF"] == keep_id  # survivor → id unchanged


class TestPlaceholderSeedIsBestEffort:
    """A missing/broken snapshot or offline fallback JSON must NOT crash boot.
    The catalog reconcile is snapshot-only (#131/#163) and the first-boot fallback
    is best-effort. Regression: the packaged backend lacked the JSON and crashed."""

    def _stub_side_steps(self, monkeypatch, seed_mod):
        """Keep the unrelated startup steps cheap + DB-only."""

        class _NoCleanup:
            def __init__(self, *a, **k):
                pass

            def cleanup_all_unfinished_jobs(self):
                return {}

        class _NoHw:
            def __init__(self, *a, **k):
                pass

            def initialize_if_needed(self):
                return False

        monkeypatch.setattr(seed_mod, "Job_Cleanup_Service", _NoCleanup)
        monkeypatch.setattr(seed_mod, "Hardware_Initializer", _NoHw)

    def test_boot_tolerates_missing_snapshot_and_fallback(self, test_db_session, monkeypatch):
        import asyncio
        from src.database import seed as seed_mod
        from src.database import catalog_snapshot as snap_mod
        from src.database.seed import Llm

        # Force the empty-catalog first-boot path.
        test_db_session.query(Llm).filter(Llm.local == 0).delete()
        test_db_session.commit()

        # No snapshot → the reconcile no-ops…
        monkeypatch.setattr(config, "LLM_Engine", type("_Eng", (), {"FORMAT_TAG": "gguf"}))
        monkeypatch.setattr(snap_mod, "load_catalog_snapshot", lambda tag: [])

        # …and no fallback JSON either → seed_initial_catalog returns 0 (its
        # internal best-effort, tested in test_catalog_snapshot). Must not crash.
        class _EmptySeeder:
            def __init__(self, *a, **k):
                pass

            def seed_initial_catalog(self):
                return 0

        monkeypatch.setattr(seed_mod, "Model_Seeder", _EmptySeeder)
        self._stub_side_steps(monkeypatch, seed_mod)

        res = asyncio.run(Database_Seeder().populate_startup_data(db=test_db_session))
        assert "needs_background_refresh" not in res  # background resync is gone
        assert res["base_models_added"] == 0

    def test_boot_reconciles_snapshot_with_zero_network(self, test_db_session, monkeypatch):
        """#131/#163 — boot reconciles the catalog from the bundled snapshot with
        ZERO network: no HTTP request, no HF client, no background task."""
        import asyncio
        from src.database import seed as seed_mod
        from src.database import catalog_snapshot as snap_mod
        from src.database.seed import Llm

        # Force the empty-catalog first-boot path.
        test_db_session.query(Llm).filter(Llm.local == 0).delete()
        test_db_session.commit()

        def _boom(*a, **k):
            raise AssertionError("no network call may run on the boot path (#131/#163)")

        # Nothing on the boot path may reach the network: neither an HF client
        # nor a bare requests call (the connectivity probe that used to live in
        # seed.py is gone -- #109/#166).
        import requests

        monkeypatch.setattr(requests, "head", _boom)
        monkeypatch.setattr(requests, "get", _boom)
        monkeypatch.setattr(config, "get_hf_api", _boom)

        monkeypatch.setattr(config, "LLM_Engine", type("_Eng", (), {"FORMAT_TAG": "gguf"}))
        monkeypatch.setattr(
            snap_mod,
            "load_catalog_snapshot",
            lambda tag: [
                {
                    "name": "A",
                    "link": "org/a-GGUF",
                    "type": "x",
                    "param_size": 1.0,
                    "is_base": True,
                },
                {
                    "name": "B",
                    "link": "org/b-GGUF",
                    "type": "x",
                    "param_size": 1.0,
                    "is_base": False,
                },
            ],
        )
        self._stub_side_steps(monkeypatch, seed_mod)

        res = asyncio.run(Database_Seeder().populate_startup_data(db=test_db_session))
        assert "needs_background_refresh" not in res  # nothing left to schedule
        assert res["base_models_added"] == 2  # snapshot applied at boot
        assert res.get("offline_mode") is False


class TestDownloadRunnabilityGuard:
    """download_llm rejects a KNOWN_BROKEN target up front (clear error, no crash)."""

    def test_rejects_known_broken(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        # Roster empty since #273; pin the mechanism with an injected entry.
        monkeypatch.setattr(
            MLX_Engine, "KNOWN_BROKEN", frozenset({"mlx-community/broken-quant-4bit"})
        )
        with pytest.raises(UnsupportedPlatformException):
            services._assert_runnable("mlx-community/broken-quant-4bit")

    def test_allows_normal_quant(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        services._assert_runnable("mlx-community/gemma-3-1b-it-4bit")  # no raise
