"""Database initialization and seeding for application startup.

This module provides a clean, type-safe API for database initialization:
- Table creation via SQLAlchemy ORM
- Model seeding from HuggingFace Hub
- Job cleanup and recovery
- Hardware profiling
- Startup state initialization

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │ Database_Seeder (Facade)                                │
    │  ├─> create_tables()                                    │
    │  ├─> populate_startup_data()                            │
    │  └─> delete_all_data() [DEV ONLY]                       │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Specialized Seeders                                     │
    │  ├─> Model_Seeder: Base + derived models               │
    │  ├─> Job_Cleanup_Service: Jobs + orphaned models       │
    │  ├─> Hardware_Initializer: System profiling            │
    │  └─> Startup_Initializer: First-run flags              │
    └─────────────────────────────────────────────────────────┘

Example:
    Automatic startup (production)::

        from src.database.seed import Database_Seeder

        seeder = Database_Seeder()
        await seeder.create_tables()
        await seeder.populate_startup_data()

    Manual reset (development only)::

        seeder = Database_Seeder()
        await seeder.delete_all_data()  # Requires confirmation

Design Principles:
    - Single Responsibility: Each class handles one seeding concern
    - Type Safety: Full type hints, Pydantic for validation
    - Error Handling: Custom exceptions with structured logging
    - Testability: Dependency injection for all external services
    - Idempotency: Safe to run multiple times
    - Separation of Concerns: Business logic separated from I/O

Note:
    This module uses a facade pattern to provide a simple API while
    maintaining clean separation of concerns internally.
"""

import os
import re
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.core.logging import logger
from src.database.generation_hints import capture_generation_hints, resolve_base_repo
from src.core import config
from src.database import core
from src.database.core import Base, SessionLocal
from src.core.exceptions import (
    DatabaseException,
    HuggingFaceAPIException,
    FileSystemException,
)

from src.utils.hf_model_metadata import (
    get_disk_size_after_quant,
    format_model_info_metadata,
    extract_parameter_pattern,
    humanize_model_name,
    measure_dir_size_bytes,
    measure_dir_size_gb,
    rewrite_size_in_metadata,
    ParameterScale,
)
from src.domains.hardware.repository import Hardware_Repository
from src.domains.llms.repository import dir_size_bytes, remove_tree_reporting
from src.domains.hardware.services import Hardware_Service
from src.engines.model_resolver import resolve_quant, base_key, is_gated
from src.database.catalog_classify import (
    categorize,
    is_conversational,
    is_derivative,
    is_nonchat_task,
    param_size_billions,
)

from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.entities.Message import Message
from src.entities.DownloadJob import DownloadJobModel
from src.entities.HardwareProfile import HardwareProfile
from src.entities.KnowledgeDocument import KnowledgeDocument
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.KBJob import KBJobModel
from src.entities.StartupVariables import StartupVariables


def load_base_models_fallback() -> List[Dict[str, Any]]:
    """Load base models from embedded JSON fallback file.

    Used when offline or HuggingFace API is unavailable. Provides minimal
    model metadata to allow app functionality without internet.

    Returns:
        List[Dict]: List of base model configurations from JSON file.

    Raises:
        FileSystemException: If fallback JSON file is missing or corrupted.

    Note:
        JSON file located at: src/database/base_models_fallback.json
        Contains 7 curated base models with essential metadata.

    Example:
        >>> models = load_base_models_fallback()
        >>> first = models[0]
        >>> print(first.get("name"))  # Gemma-1B
    """
    fallback_path = config.ROOT_DIR / "src" / "database" / "base_models_fallback.json"

    try:
        with open(fallback_path, "r") as f:
            models = json.load(f)

        logger.info(f"Loaded {len(models)} base models from offline fallback")
        return models

    except FileNotFoundError:
        raise FileSystemException(
            f"Base models fallback file not found: {fallback_path}", trace="FileNotFoundError"
        )
    except json.JSONDecodeError as e:
        raise FileSystemException(f"Failed to parse base models fallback JSON: {e}", trace=str(e))


def _safetensors_total(model_info) -> Optional[int]:
    """Extract the total parameter count from a ModelInfo's safetensors field.

    HF returns it either as an object with a ``.total`` attribute or a plain dict
    ``{"total": N}`` depending on version; tolerate both and return None otherwise.
    """
    st = getattr(model_info, "safetensors", None)
    if st is None:
        return None
    total = getattr(st, "total", None)
    if total is None and isinstance(st, dict):
        total = st.get("total")
    return int(total) if total else None


# ============ Configuration Data Classes ============


@dataclass(frozen=True)
class Model_Config:
    """Configuration for a base model to seed.

    The optional fields carry the signals captured at discovery time (one
    ``list_models(expand=[...])`` call) so classification doesn't need extra HF
    round-trips: ``safetensors_total`` → real param size, ``category`` → capability
    bucket (#122).
    """

    name: str
    link: str
    model_type: str
    safetensors_total: Optional[int] = None
    category: str = "general"

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if not self.name or not self.link or not self.model_type:
            raise ValueError(f"Invalid model config: {self}")


@dataclass(frozen=True)
class Search_Config:
    """Configuration for derived model search."""

    search_term: str
    model_type: str
    default_param_size: float

    def __post_init__(self) -> None:
        """Validate search configuration.

        An empty ``search_term`` is allowed and means the *global pass*: search the
        whole format-tagged space by downloads (no text filter)."""
        if not self.model_type:
            raise ValueError(f"Invalid search config: {self}")
        if self.default_param_size <= 0:
            raise ValueError(f"Invalid param size: {self.default_param_size}")


@dataclass(frozen=True)
class Quality_Filters:
    """Popularity floor for derived/community models. Deliberately just a
    downloads/likes threshold — NO content or keyword filtering: the catalog is
    open to all community models (distilled, RL, uncensored…), and the format tag
    already guarantees runnability. The floor keeps it from being all of HF, and
    doubles as a safeguard against a mistagged repo."""

    min_downloads: int = 50
    min_likes: int = 5


# Freshness pass tuning (#305). A downloads-only ranking structurally favors old
# checkpoints (established models accumulate CI-driven downloads a recent release
# cannot match for months), so org discovery runs a second, creation-date-sorted
# pass with its own small quota and a higher downloads floor to filter noise.
FRESH_TOP_N: int = 5
FRESH_MIN_DOWNLOADS: int = 50_000


# ============ Model Seeding Service ============


class Model_Seeder:
    """Handles seeding of base and derived models from HuggingFace.

    Supports both online and offline modes:
    - Online: Fetches fresh metadata from HuggingFace API
    - Offline: Uses embedded JSON fallback with minimal metadata
    """

    def __init__(
        self,
        db: Session,
        hf_api=None,
        quality_filters: Optional[Quality_Filters] = None,
        offline_mode: bool = False,
    ):
        """Initialize model seeder.

        Args:
            db: Active database session.
            hf_api: HuggingFace API client (None if offline).
            quality_filters: Quality filtering configuration.
            offline_mode: If True, skip API calls and use fallback data.
        """
        self.db = db
        self.hf_api = hf_api
        self.filters = quality_filters or Quality_Filters()
        self.offline_mode = offline_mode

    # Slug tokens marking a non-final / intermediate / non-LLM artifact, excluded
    # from org discovery (token-matched, so 'pt' won't hit 'gpt'). '-assistant'
    # distillates and '-qat-…-unquantized' intermediates are the #122 offenders.
    ARTIFACT_TOKENS: frozenset = frozenset(
        {
            "gguf",
            "mlx",
            "4bit",
            "8bit",
            "6bit",
            "gptq",
            "awq",
            "bnb",
            "lora",
            "adapter",
            "onnx",
            "pt",
            "pretrain",
            "draft",
            "mtp",
            "qat",
            "unquantized",
            "embedding",
            "reranker",
            "reward",
            "rm",
            "prm",
            "assistant",
            "fp8",
            "nvfp4",
        }
    )
    # Non-chat task filtering (NONCHAT_FAMILIES substring + pipeline denylist) is
    # shared with the community-search path and lives in catalog_classify
    # (is_nonchat_task), so both ingestion doors use one audited list (#242).
    # Pipelines we draw the Base catalog from: plain text chat + (per #122) the
    # multimodal VLMs whose primary pipeline is image-text-to-text / any-to-any.
    TEXT_PIPELINES: tuple = ("text-generation",)
    VISION_PIPELINES: tuple = ("image-text-to-text", "any-to-any")

    def discover_instruct_models(
        self,
        org: str,
        model_type: str,
        top_n: int = 14,
        vision_top_n: int = 8,
        min_downloads: int = 2000,
        fresh_top_n: int = FRESH_TOP_N,
        fresh_min_downloads: int = FRESH_MIN_DOWNLOADS,
    ) -> List[Model_Config]:
        """Discover an org's chat-capable models (text + multimodal) as base candidates.

        Text and vision get SEPARATE quotas (``top_n`` / ``vision_top_n``) so a busy
        text family never starves the multimodal pass — the real VLMs must reach Base
        (#122). For each relevant ``pipeline_tag`` (drops the org's CLIP / Whisper /
        BERT, and keeps VLMs out of the text bucket) it runs TWO ranked passes with
        ``expand`` so safetensors/tags/pipeline come back in ONE call each:

        - the classic top-repos-by-downloads pass (quota ``top_n``/``vision_top_n``);
        - a FRESHNESS pass sorted by creation date, newest first (#305): cumulative
          downloads structurally hide recent releases, so the newest repos get their
          own small quota (``fresh_top_n``) behind a higher downloads floor
          (``fresh_min_downloads``). Fresh candidates ADD to the downloads-ranked
          set — they never displace it.

        Both passes share the same rejection rules — quant/merge/adapter derivatives
        (``base_model`` relation tags), intermediate artifacts, raw pretrains — and
        dedup by normalized slug. Anything without an engine-format quant
        self-corrects later (the resolver returns None).
        """
        out: List[Model_Config] = []
        seen: set = set()

        def fetch(pipeline: str, sort: str) -> list:
            try:
                return list(
                    self.hf_api.list_models(
                        author=org,
                        pipeline_tag=pipeline,
                        sort=sort,
                        limit=80,
                        expand=[
                            "safetensors",
                            "cardData",
                            "tags",
                            "pipeline_tag",
                            "gated",
                            "downloads",
                        ],
                    )
                )
            except Exception as e:
                logger.warning(f"Org discovery failed for {org}/{pipeline} (sort={sort}): {e}")
                return []

        def accept(m, floor: int) -> Optional[Model_Config]:
            """One candidate through every rejection rule; None means rejected.
            Shared by both passes so the freshness door can't bypass a filter."""
            if (getattr(m, "downloads", 0) or 0) < floor:
                return None
            name = m.id.split("/")[-1]
            pipeline_tag = getattr(m, "pipeline_tag", None)
            tags = list(getattr(m, "tags", None) or [])
            if set(re.split(r"[-_.]", name.lower())) & self.ARTIFACT_TOKENS:
                return None
            if is_nonchat_task(name, pipeline_tag):
                return None
            if is_derivative(tags):
                return None
            # #182: the Base catalog is chat-only. Keep only conversational
            # (instruct/chat) releases; raw pretrains (Llama-3.2-1B,
            # Mistral-7B-v0.1) are dropped and reachable only via HF search.
            if not is_conversational(tags, name):
                return None
            key = base_key(m.id)
            if key in seen:
                return None
            seen.add(key)
            return Model_Config(
                name,
                m.id,
                model_type,
                safetensors_total=_safetensors_total(m),
                category=categorize(name, tags, pipeline_tag),
            )

        for pipelines, cap in ((self.TEXT_PIPELINES, top_n), (self.VISION_PIPELINES, vision_top_n)):
            added = 0
            fresh_added = 0
            for pipeline in pipelines:
                if added < cap:
                    for m in fetch(pipeline, "downloads"):
                        if added >= cap:
                            break
                        if (mc := accept(m, min_downloads)) is not None:
                            out.append(mc)
                            added += 1
                # Freshness pass (#305): newest repos first (`created_at` on the
                # installed huggingface_hub maps to the raw API's `createdAt`,
                # descending). Own quota, higher floor, same filters and dedup.
                if fresh_added < fresh_top_n:
                    for m in fetch(pipeline, "created_at"):
                        if fresh_added >= fresh_top_n:
                            break
                        if (mc := accept(m, fresh_min_downloads)) is not None:
                            out.append(mc)
                            fresh_added += 1
        return out

    # Suffix tokens that mark a chat-tuned release (vs its raw pretrain sibling).
    _INSTRUCT_SUFFIX: frozenset = frozenset({"it", "instruct", "chat"})

    def _prefer_instruct_siblings(self, candidates: List[Model_Config]) -> List[Model_Config]:
        """Drop a bare pretrain when its instruct sibling is also present (#122).

        Groups by family slug (normalized, minus trailing it/instruct/chat). If any
        member of a family carries an instruct suffix, the non-suffixed bare pretrains
        in that family are dropped — keeping ``gemma-2-9b-it`` over ``gemma-2-9b``,
        while a suffix-less lone release (``DeepSeek-V3``) is untouched.
        """

        def family(mc: Model_Config) -> str:
            toks = base_key(mc.link).split("-")
            while toks and toks[-1] in self._INSTRUCT_SUFFIX:
                toks.pop()
            return "-".join(toks)

        def has_instruct(mc: Model_Config) -> bool:
            return any(t in self._INSTRUCT_SUFFIX for t in base_key(mc.link).split("-"))

        groups: Dict[str, List[Model_Config]] = {}
        for mc in candidates:
            groups.setdefault(family(mc), []).append(mc)
        out: List[Model_Config] = []
        for members in groups.values():
            instruct = [m for m in members if has_instruct(m)]
            out.extend(instruct if instruct else members)
        return out

    def build_base_models(self, orgs) -> List[Llm]:
        """Discover each foundation org's chat models and build (don't persist) the
        base catalog rows, each resolved to its engine-format quant.

        Per candidate: discover → prefer-instruct → resolve_quant → build. A base
        with no quant for the active engine is skipped. Resolved quants are deduped
        (same repo never seeded twice — #122). HF metadata failure falls back to
        default metadata so one bad model never drops the rest. `orgs` is the
        FOUNDATION_ORGS list of (org, family_type, search_term).
        """
        out: List[Llm] = []
        seen_quant: set = set()
        tag = getattr(config.LLM_Engine, "FORMAT_TAG", None)
        for org, model_type, _term in orgs:
            candidates = self._prefer_instruct_siblings(
                self.discover_instruct_models(org, model_type)
            )
            for model_config in candidates:
                try:
                    quant_link = resolve_quant(model_config.link, tag, self.hf_api)
                except Exception as e:
                    logger.warning(f"resolve_quant failed for {model_config.link}: {e}")
                    quant_link = None
                if not quant_link or quant_link in seen_quant:
                    continue
                try:
                    out.append(self._create_base_llm(model_config, quant_link))
                    seen_quant.add(quant_link)
                except HuggingFaceAPIException as e:
                    logger.warning(f"HF metadata failed for {model_config.link}, fallback: {e}")
                    try:
                        out.append(self._create_base_llm_fallback(model_config, quant_link))
                        seen_quant.add(quant_link)
                    except Exception as fe:
                        # The model is skipped; the catalog goes on without it.
                        logger.warning(
                            f"Fallback build failed for {model_config.link}, skipping it: {fe}",
                            exc_info=True,
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to build base model {model_config.link}, skipping it: {e}",
                        exc_info=True,
                    )
        return out

    def seed_from_snapshot(self) -> int:
        """Seed the remote catalog (local=0) from the bundled build-time snapshot
        for the active engine format (#112). Instant, zero HF calls — this is the
        first-boot catalog. Returns the number of rows added (0 if no snapshot)."""
        from src.database.catalog_snapshot import load_catalog_snapshot, dict_to_llm

        tag = getattr(config.LLM_Engine, "FORMAT_TAG", None)
        if not tag:
            return 0
        entries = load_catalog_snapshot(tag)
        if not entries:
            return 0
        self.db.add_all([dict_to_llm(e) for e in entries])
        self.db.commit()
        logger.info(f"Seeded {len(entries)} catalog entries from the {tag} snapshot")
        return len(entries)

    def seed_initial_catalog(self) -> int:
        """First-boot catalog: the bundled build-time snapshot if present (instant,
        full, zero HF calls — #112), else the minimal offline fallback JSON.
        Best-effort: returns 0 rather than raising if neither is available, so boot
        never crashes on a missing artifact."""
        try:
            count = self.seed_from_snapshot()
            if count:
                return count
        except Exception as e:
            logger.warning(f"Snapshot seed skipped: {e}")
        try:
            return self.seed_base_models_offline()
        except Exception as e:
            logger.warning(f"Offline fallback seed skipped (catalog stays empty): {e}")
            return 0

    def seed_base_models_offline(self) -> int:
        """Seed base models from embedded JSON fallback (offline mode).

        Used when internet is unavailable or HuggingFace API fails. Loads
        models from static JSON file with minimal but sufficient metadata.

        Returns:
            Number of models successfully added from fallback.

        Raises:
            FileSystemException: If fallback JSON is missing or corrupted.
            DatabaseException: If database operations fail.

        Note:
            This method ONLY seeds base models. Derived models are skipped
            in offline mode as they require fresh HuggingFace searches.

        Example:
            >>> seeder = Model_Seeder(db, offline_mode=True)
            >>> count = seeder.seed_base_models_offline()
            >>> print(f"Seeded {count} base models in offline mode")
        """
        logger.warning("Seeding in OFFLINE mode using fallback data")

        fallback_models = load_base_models_fallback()
        added_count = 0

        for model_data in fallback_models:
            # Offline: there is no HF to resolve a quant, so seed the bundled link
            # as-is (the offline JSON should already carry resolved quant links).
            actual_link = model_data["link"]
            if self._link_exists(actual_link):
                logger.debug(f"Skipping existing model: {actual_link}")
                continue

            try:
                # Create model config from JSON data
                model_config = Model_Config(
                    name=model_data["name"], link=model_data["link"], model_type=model_data["type"]
                )

                # Create LLM entity with fallback metadata
                llm = self._create_base_llm_from_json(model_data)
                self.db.add(llm)
                self.db.flush()
                added_count += 1
                logger.info(f"Added base model (offline): {model_data['name']}")

            except Exception as e:
                logger.warning(
                    f"Failed to add offline model {model_data['name']}, skipping it: {e}",
                    exc_info=True,
                )
                continue

        self.db.commit()
        logger.info(f"Offline seeding complete: {added_count} base models added")
        return added_count

    def _create_base_llm_from_json(self, model_data: Dict[str, Any]) -> Llm:
        """Create LLM entity from JSON fallback data.

        Args:
            model_data: Dictionary from fallback JSON with keys:
                name, link, type, param_size, model_metadata

        Returns:
            Llm: Entity ready to be added to database.
        """
        # Offline: use the bundled link + flag as-is (no HF resolution available).
        actual_link = model_data["link"]
        is_quantized = bool(model_data.get("quantized", False))

        # Use embedded metadata and param_size from JSON
        return Llm(
            name=humanize_model_name(model_data["link"]),
            local=0,
            link=actual_link,
            type=model_data["type"],
            quantized=is_quantized,
            model_metadata=model_data["model_metadata"],
            param_size=model_data["param_size"],
            # The offline fallback seeds base models only (#86).
            is_base=True,
            # Base models are chat-ready (#182); honor the snapshot flag if present.
            conversational=bool(model_data.get("conversational", True)),
        )

    def build_derived_models(
        self,
        searches: List[Search_Config],
        top_per_search: int = 30,
        max_checked: int = 200,
    ) -> List[Llm]:
        """Build (do NOT persist) derived/community catalog rows from HF search.

        Best-effort: a failing search is logged and skipped, never aborts the rest.
        Returns detached Llm objects for an atomic swap (no add/commit here).
        """
        out: List[Llm] = []
        seen: set = set()
        for search_config in searches:
            try:
                # Bounded limit: get_hf_api() returns a retrying client that
                # materializes list_models (to retry 429s that surface during lazy
                # pagination), so an unbounded search must not be requested here.
                results = self.hf_api.list_models(
                    **config.LLM_Engine.community_search_kwargs(search_config.search_term),
                    sort="downloads",
                    limit=max_checked,
                )
            except Exception as e:
                logger.warning(f"HF search '{search_config.search_term}' failed, skipping: {e}")
                continue
            added = checked = 0
            for model_info in results:
                if added >= top_per_search or checked >= max_checked:
                    break
                checked += 1
                if not self._passes_quality_filters(model_info):
                    continue
                # #242: this path is filtered only by format tag, so ASR/embedding/
                # OCR/media-gen community repos would otherwise enter the catalog.
                # Reject them by name family + non-chat pipeline (needs the
                # pipeline_tag/tags expand added to community_search_kwargs).
                slug = model_info.id.split("/")[-1]
                if is_nonchat_task(slug, getattr(model_info, "pipeline_tag", None)):
                    continue
                # The app downloads with no HF token: a gated repo lists fine but
                # 401s on download, so it never enters the catalog. Checked BEFORE
                # the dedup so a public twin of the same model still gets in.
                if is_gated(model_info):
                    continue
                # Dedup by normalized key so the same finetune from two quanters
                # (bartowski/Foo-GGUF vs mradermacher/Foo-GGUF) appears once.
                mkey = base_key(model_info.id)
                if mkey in seen:
                    continue
                # Runnable by construction (came from filter=FORMAT_TAG); only drop
                # the rare KNOWN_BROKEN load-crashers.
                if not config.LLM_Engine.is_runnable(model_info.id):
                    continue
                try:
                    out.append(self._create_derived_llm(model_info, search_config))
                    seen.add(mkey)
                    added += 1
                except Exception as e:
                    logger.warning(f"Failed to build derived {model_info.id}: {e}")
        return out

    def _link_exists(self, link: str) -> bool:
        """Check if model already exists by link."""
        return self.db.query(Llm).filter(Llm.link == link).first() is not None

    def _create_base_llm(self, model_config: Model_Config, quant_link: str) -> Llm:
        """Create a base LLM entity (full metadata) for a resolved engine-format quant.

        `quant_link` is the public quant the resolver found for `model_config.link`;
        the display name stays derived from the clean base id.
        """
        model_info = self.hf_api.model_info(model_config.link)
        # One repo_info(files_metadata=True) on the QUANT repo (the one actually
        # downloaded) feeds both the Size line and the exact byte count below.
        size_estimate = get_disk_size_after_quant(quant_link, hf_api=self.hf_api)
        # Real param count from the base's safetensors.total (captured at discovery),
        # slug as sanity-checked fallback — no more blanket 7.0 (#122).
        param_size = param_size_billions(
            model_config.safetensors_total, model_config.link.split("/")[-1]
        )
        metadata = format_model_info_metadata(model_info, size_estimate, True)

        return Llm(
            name=humanize_model_name(model_config.link),
            local=0,
            link=quant_link,
            type=model_config.model_type,
            quantized=True,
            model_metadata=metadata,
            param_size=param_size,
            # Real download size: the chosen artifact's bytes when HF answered,
            # None when the Size line is an estimate (never laundered into bytes).
            artifact_size_bytes=size_estimate.size_bytes,
            # Curated foundation model (discovered from a FOUNDATION_ORG) — drives the
            # Base/Community split and "Models For You" recommendations in the UI (#86).
            is_base=True,
            # Base discovery keeps only conversational releases (#182), so every base
            # row is chat-ready by construction.
            conversational=True,
            category=model_config.category,
            # Pre-download tool detection is intentionally NOT done here: it required
            # downloading a tokenizer per catalog model, which is not viable at catalog
            # scale (#113). supports_tools stays null and is computed post-download
            # (where the tokenizer is already on disk).
            supports_tools=None,
            # Sampling facts (#388): model_config.link is the BASE id (the
            # resolver only rewrote the quant link). Cascade base
            # generation_config > quant generation_config > base model card,
            # tiny file fetches memoized per repo, best-effort (None on failure).
            generation_hints=capture_generation_hints(
                model_config.link, self.hf_api, quant_repo=quant_link
            ),
        )

    def _create_base_llm_fallback(self, model_config: Model_Config, quant_link: str) -> Llm:
        """Create a base LLM with fallback metadata when base HF metadata is missing."""
        size_estimate = get_disk_size_after_quant(quant_link, hf_api=self.hf_api)
        param_size = param_size_billions(
            model_config.safetensors_total, model_config.link.split("/")[-1]
        )

        fallback_metadata = (
            f"Size: {size_estimate.to_string()}\n"
            f"Model ID: {model_config.link}\n"
            f"Quantized: True\n"
            f"Author: Unknown\n"
            f"Library: Unknown"
        )

        return Llm(
            name=humanize_model_name(model_config.link),
            local=0,
            link=quant_link,
            type=model_config.model_type,
            quantized=True,
            model_metadata=fallback_metadata,
            param_size=param_size,
            # Real download size -- see _create_base_llm.
            artifact_size_bytes=size_estimate.size_bytes,
            # Curated foundation model — see _create_base_llm (#86).
            is_base=True,
            # Chat-ready by construction — see _create_base_llm (#182).
            conversational=True,
            category=model_config.category,
            # Deferred to post-download (see _create_base_llm / #113).
            supports_tools=None,
            # Sampling facts cascade -- see _create_base_llm (#388).
            generation_hints=capture_generation_hints(
                model_config.link, self.hf_api, quant_repo=quant_link
            ),
        )

    def _passes_quality_filters(self, model_info) -> bool:
        """Keep any model above the popularity floor — nothing else.

        No content/keyword/id filtering: the catalog is open to all community models
        (distilled, RL, uncensored…), and the format tag already guarantees the model
        is runnable. The floor just keeps the catalog from being all of HF.
        """
        return (
            model_info.downloads >= self.filters.min_downloads
            and model_info.likes >= self.filters.min_likes
        )

    def _create_derived_llm(self, model_info, search_config: Search_Config) -> Llm:
        """Create derived LLM entity from search result."""
        model_name = model_info.id.split("/")[-1]

        # Real download size, the same way as a base row: one
        # repo_info(files_metadata=True) on the community repo, summed over the
        # files the downloader would fetch (GGUF: the chosen quant, not the whole
        # multi-quant repo). It replaces the old full-precision family guess,
        # which overshot a 4-bit quant several times over. Estimate on failure.
        size_estimate = get_disk_size_after_quant(model_info.id, hf_api=self.hf_api)

        # Extract parameters. When the id carries no size token, leave param_size
        # unknown (None) rather than substituting a plausible default (#201): a
        # defaulted number is indistinguishable from a measured one downstream and
        # rates unmeasurable models as perfect hardware fits.
        param_count = extract_parameter_pattern(model_info.id)
        if param_count and param_count.scale == ParameterScale.BILLION:
            param_size = param_count.count
        elif param_count and param_count.scale == ParameterScale.MILLION:
            param_size = param_count.count / 1000.0
        else:
            param_size = None

        # Format metadata
        metadata = format_model_info_metadata(model_info, size_estimate, quantized=False)

        tags = list(getattr(model_info, "tags", None) or [])
        return Llm(
            name=humanize_model_name(model_info.id),
            local=0,
            link=model_info.id,
            type=search_config.model_type,
            # Came from a filter=FORMAT_TAG search → it IS an engine-format quant.
            quantized=True,
            model_metadata=metadata,
            param_size=param_size,
            # Real download size -- see _create_base_llm.
            artifact_size_bytes=size_estimate.size_bytes,
            # Derived/community quant (not a curated foundation model) (#86).
            is_base=False,
            # Chat-readiness so the UI can rank IT models first even among community
            # rows (#182). A community merge/pretrain without the tag or suffix sorts
            # below the instruct ones rather than being dropped.
            conversational=is_conversational(tags, model_name),
            category=categorize(model_name, tags, getattr(model_info, "pipeline_tag", None)),
            # Sampling facts (#388): a community quant inherits its base's
            # generation_config (first base_model:* card tag), else its own; the
            # quant repo itself is the cascade's second stage.
            generation_hints=capture_generation_hints(
                resolve_base_repo(model_info.id, tags), self.hf_api, quant_repo=model_info.id
            ),
        )


# ============ Job Cleanup Service ============


class Job_Cleanup_Service:
    """Handles cleanup of interrupted jobs and orphaned resources.

    Responsibilities:
    - Mark interrupted jobs (download, KB) as failed
    - Remove incomplete model files and temp directories
    - Cleanup orphaned model directories without database entries
    """

    def __init__(self, db: Session):
        """Initialize job cleanup service.

        Args:
            db: Active database session.
        """
        self.db = db

    def cleanup_all_unfinished_jobs(self) -> Dict[str, int]:
        """Mark all interrupted jobs as failed and cleanup resources.

        Returns:
            Dictionary with counts: {"download": N, "kb": N, "orphaned": N}
        """
        counts = {
            "download": self._cleanup_download_jobs(),
            "kb": self._cleanup_kb_jobs(),
            "orphaned": self._cleanup_orphaned_models(),
        }

        total = sum(counts.values())
        if total > 0:
            logger.info(
                f"Cleaned up {total} unfinished jobs: "
                f"download={counts['download']}, "
                f"kb={counts['kb']}, "
                f"orphaned={counts['orphaned']}"
            )

        return counts

    def _artifact_is_complete(self, job: DownloadJobModel, llm: Llm) -> bool:
        """Whether the model on disk is a usable artifact worth preserving (#314).

        A job row left in ``running``/``pending`` means "we do not know how this
        ended", NOT "the files are garbage": #291 strands the row at ``running``
        AFTER the transfer fully succeeded, so deleting on the strength of the
        status alone throws away a complete multi-GB download.

        Two independent signals, either one is enough:

        1. The active engine's own integrity gate (``validate_local_artifact``,
           #88) -- the same validator the download-completion path uses, so
           cleanup and finalization can never disagree on what "complete" means.
           It is metadata-only (GGUF magic + non-empty weights file), so it is
           cheap enough to run inside the boot sequence.
        2. The on-disk footprint covers the job's recorded ``total_bytes``. Used
           when no engine is bound, so a full artifact is never deleted merely
           because the engine could not be consulted. Compared in BYTES on both
           sides (#316): reading the footprint through a GB helper and scaling it
           back would silently change this threshold the day that helper's
           divisor changes, and a threshold that drifts upward marks a truncated
           download "complete" -- worse than the deletion this guards against,
           since a kept-but-broken artifact is never cleaned up again.

        Defensive: any failure answers False, i.e. falls back to the historical
        delete-and-retry behavior.
        """
        link = getattr(llm, "link", None)
        if not link or not os.path.exists(link):
            return False

        validator = getattr(config.LLM_Engine, "validate_local_artifact", None)
        if validator is not None:
            try:
                validator(str(link))
                return True
            except Exception as e:
                # Engine says the artifact is incomplete/corrupt: it is genuine
                # download debris, fall through and let it be removed. Said
                # out loud, because what follows deletes gigabytes.
                logger.warning(
                    f"Artifact at {link} failed the integrity check and will be removed: {e}"
                )
                return False

        total_bytes = getattr(job, "total_bytes", None) or 0
        if total_bytes > 0:
            try:
                measured_bytes = measure_dir_size_bytes(link)
            except Exception as e:
                logger.warning(f"Could not measure {link}; treating it as debris: {e}")
                return False
            return measured_bytes >= total_bytes
        return False

    def _cleanup_download_jobs(self) -> int:
        """Resolve interrupted download jobs, preserving complete artifacts (#314).

        Historically this deleted the model directory for every job left in
        ``running``/``pending``, purely on the strength of the status. That is
        data loss whenever a download finished but its row was never finalized
        (#291): the user downloaded 9 GB, the transfer worked, and the next boot
        silently rmtree'd it.

        Now every unfinished job is triaged first. A complete artifact is
        RESCUED (job finalized ``completed``, model flipped to ``local=1``);
        only genuine debris is removed, and every deletion is logged with its
        path and reclaimed size so a multi-GB delete is never silent again.

        Capability columns are deliberately left NULL on a rescued row: the
        post-ready ``backfill_wire_tools`` and the boot-time
        ``backfill_local_model_sizes`` fill them in exactly as they do for any
        other legacy local model, so no capability probe runs inside boot.
        """
        unfinished = (
            self.db.query(DownloadJobModel)
            .filter(DownloadJobModel.status.in_(["running", "pending"]))
            .all()
        )

        count = 0
        rescued = 0
        for job in unfinished:
            try:
                llm = self.db.query(Llm).filter(Llm.id == job.local_model_id).first()

                if llm and self._artifact_is_complete(job, llm):
                    # The download actually succeeded; only the bookkeeping was
                    # lost. Finalize it instead of destroying the artifact.
                    llm.local = 1
                    job.status = "completed"
                    job.progress = 100.0
                    job.error_message = None
                    rescued += 1
                    logger.info(
                        f"Download job {job.id}: artifact at {llm.link} is "
                        f"complete; finalizing as completed instead of deleting "
                        f"(model {llm.id} kept)"
                    )
                elif llm and os.path.exists(llm.link):
                    # Genuine debris: log what is destroyed BEFORE destroying it.
                    reclaimed_gb = 0.0
                    try:
                        reclaimed_gb = measure_dir_size_gb(llm.link)
                    except Exception:
                        pass
                    logger.info(
                        f"Download job {job.id}: removing incomplete model "
                        f"{llm.id} at {llm.link} (reclaiming ~{reclaimed_gb:.2f} GB)"
                    )
                    shutil.rmtree(llm.link, ignore_errors=True)
                    self.db.delete(llm)
                    job.status = "failed"
                    job.error_message = "Download interrupted due to application shutdown"
                else:
                    job.status = "failed"
                    job.error_message = "Download interrupted due to application shutdown"

                # Delete temp files. The staging dir is scratch space in every
                # case: on a rescue its contents were already moved into place.
                #
                # On a TRUNCATED download this is where nearly all the bytes are:
                # killing the app at 26% of a 4.7 GB model left an empty final
                # dir and 1.28 GB of staging, so the line above announced
                # "reclaiming ~0.00 GB" and this deleted the real 1.28 GB without
                # a word. Measure and name it too -- the comment above already
                # asks for exactly that.
                if job.temp_local_model_link and os.path.exists(job.temp_local_model_link):
                    staged = dir_size_bytes(job.temp_local_model_link)
                    logger.info(
                        f"Download job {job.id}: removing staging directory "
                        f"{job.temp_local_model_link} (reclaiming {staged} bytes)"
                    )
                    left = remove_tree_reporting(job.temp_local_model_link)
                    if left:
                        logger.warning(
                            f"Download job {job.id}: staging directory "
                            f"{job.temp_local_model_link} not fully removed "
                            f"({left} bytes still on disk)"
                        )

                # The temp Llm delete above nulls local_model_id server-side
                # (FK SET NULL); updated_at is stamped by onupdate=func.now().
                job.temp_local_model_link = ""

                count += 1
            except Exception as e:
                # One job's cleanup must not sink the boot: log it with the
                # traceback and move on to the next.
                logger.error(f"Failed to clean up download job {job.id}: {e}", exc_info=True)
                continue

        if count > 0:
            self.db.commit()
        if rescued > 0:
            logger.info(
                f"Preserved {rescued} fully-downloaded model(s) whose job row "
                f"was left unfinished"
            )

        return count

    def _cleanup_kb_jobs(self) -> int:
        """Cleanup interrupted knowledge base jobs.

        Creation jobs are rolled back: the specialized LLM and the KB are
        deleted (KnowledgeDocument rows follow through ON DELETE CASCADE, the
        job's refs are nulled server-side by the FKs). Update jobs
        (new_model_id == base_model_id) leave the existing KB and assistant
        untouched — the corpus indexed before the interruption is still valid.
        """
        unfinished = (
            self.db.query(KBJobModel).filter(KBJobModel.status.in_(["running", "pending"])).all()
        )

        count = 0
        for job in unfinished:
            try:
                is_update = job.new_model_id == job.base_model_id

                if not is_update:
                    new_llm = self.db.query(Llm).filter(Llm.id == job.new_model_id).first()
                    if new_llm:
                        self.db.delete(new_llm)

                    kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == job.kb_id).first()
                    if kb:
                        self.db.delete(kb)

                job.status = "failed"
                job.error_message = (
                    "KB update interrupted due to application shutdown"
                    if is_update
                    else "KB creation interrupted due to application shutdown"
                )

                count += 1
            except Exception as e:
                logger.error(f"Error cleaning KB job {job.id}: {e}")
                continue

        if count > 0:
            self.db.commit()

        return count

    def _cleanup_orphaned_models(self) -> int:
        """Cleanup orphaned model files without corresponding database entries.

        This handles cases where:
        - The app is reinstalled but Application Support data persists
        - Temp directories from interrupted downloads remain

        Returns:
            Total count of orphaned models and temp directories removed.

        Raises:
            FileSystemException: If critical filesystem operations fail.
        """
        models_dir = config.LLM_DIR

        # Return early if models directory doesn't exist
        if not models_dir.exists():
            logger.debug("Models directory doesn't exist, nothing to clean up")
            return 0

        # Get all valid model IDs from database
        try:
            local_models = self.db.query(Llm).filter(Llm.local == 1).all()
            valid_model_ids = {str(model.id) for model in local_models}
        except DatabaseException as e:
            logger.error(f"Database error fetching local models: {e}")
            return 0

        # Scan and cleanup orphaned directories
        cleaned_count = 0
        temp_cleaned_count = 0
        reclaimed_bytes = 0

        def _sweep(item, kind: str) -> bool:
            """Remove one directory, reporting its path and the bytes it freed.

            Deleting a user's multi-gigabyte artifacts must never be silent: the
            log has to name the path and the size so a surprised user can see
            what went and how much came back.
            """
            nonlocal reclaimed_bytes
            size = dir_size_bytes(item)
            logger.info(f"Removing {kind}: {item} ({size} bytes)")
            try:
                left = remove_tree_reporting(item)
            except Exception as e:
                logger.error(f"Failed to remove {kind} {item.name}: {e}")
                return False
            if left:
                logger.error(f"Failed to remove {kind} {item.name}: {left} bytes still on disk")
                return False
            reclaimed_bytes += size
            return True

        try:
            for item in models_dir.iterdir():
                if not item.is_dir():
                    continue

                dir_name = item.name

                # Cleanup temp directories (they start with "temp_")
                if dir_name.startswith("temp_"):
                    if _sweep(item, "temporary model directory"):
                        temp_cleaned_count += 1
                    continue

                # Cleanup orphaned model directories
                if dir_name not in valid_model_ids:
                    if _sweep(item, "orphaned model directory"):
                        cleaned_count += 1

            total_cleaned = cleaned_count + temp_cleaned_count
            if total_cleaned > 0:
                logger.info(
                    f"Cleaned up {cleaned_count} orphaned model(s) and "
                    f"{temp_cleaned_count} temp directory(ies), "
                    f"reclaiming {reclaimed_bytes} bytes"
                )
            else:
                logger.debug("No orphaned models or temp directories found")

            return total_cleaned

        except FileSystemException as e:
            logger.error(f"Filesystem error during orphaned model cleanup: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during orphaned model cleanup: {e}")
            return cleaned_count + temp_cleaned_count


# ============ Hardware Initialization Service ============


class Hardware_Initializer:
    """Handles system hardware profiling and persistence."""

    def __init__(self, db: Session):
        """Initialize hardware initializer.

        Args:
            db: Active database session.
        """
        self.db = db
        self.service = Hardware_Service(Hardware_Repository(db))

    def initialize_if_needed(self) -> bool:
        """Initialize hardware info if not already present.

        Uses service layer to get or create hardware profile.

        Returns:
            True if initialization was performed, False if already existed.
        """
        try:
            # Deliberately no "a row exists, skip" short-circuit here. That is
            # what stranded every existing install on the numbers its FIRST boot
            # produced: #365 fixed a 448 GB/s card being profiled at 13 GB/s, and
            # the corrected build still served the stored 13 because this method
            # returned before the service could look at it. The only way out was
            # Clear All Data, which also destroys models and conversations.
            #
            # get_or_create_profile owns validity (backend match AND profiling
            # version) and returns the cached row untouched when it is still
            # good, so the common boot stays as cheap as it was.
            previous = self.db.query(HardwareProfile).first()
            previous_id = previous.id if previous else None

            profile = self.service.get_or_create_profile()
            self.db.commit()

            if previous_id is None:
                logger.info(f"Hardware info initialized: backend={profile.backend_type}")
                return True
            if profile.id != previous_id:
                logger.info(f"Hardware info re-profiled: backend={profile.backend_type}")
                return True

            logger.debug("Hardware info already initialized, skipping")
            return False

        except Exception as e:
            logger.exception(f"Hardware initialization failed: {e}")
            self.db.rollback()
            # Create fallback profile on error
            self._create_fallback_profile()
            return True

    def _create_fallback_profile(self) -> None:
        """Create fallback hardware profile on initialization failure."""
        try:
            fallback_data = {
                "backend_type": "cpu",
                "cpu_model": "Unknown CPU",
                "total_memory_gb": 8.0,
                "available_memory_gb": 4.0,
                "disk_total_gb": 100.0,
                "disk_available_gb": 50.0,
                "global_inference_score": 20.0,
                "global_inference_label": "Poor",
                "cpu_score": 30.0,
                "memory_score": 25.0,
                "system_platform": "Unknown",
            }

            profile = HardwareProfile(**fallback_data)
            self.db.add(profile)
            self.db.commit()
            logger.warning("Fallback hardware profile created")

        except Exception as e:
            logger.exception(f"Failed to create fallback profile: {e}")
            self.db.rollback()


# ============ Startup Variables Initialization ============


class Startup_Initializer:
    """Handles initialization of startup state variables."""

    def __init__(self, db: Session):
        """Initialize startup initializer.

        Args:
            db: Active database session.
        """
        self.db = db

    def initialize_if_needed(self) -> bool:
        """Initialize startup variables if not already present.

        Returns:
            True if initialization was performed, False if already existed.
        """
        existing = self.db.query(StartupVariables).first()
        if existing:
            logger.debug("Startup variables already initialized, skipping")
            return False

        variables = StartupVariables(welcome_popup_has_already_displayed=False)
        self.db.add(variables)
        self.db.commit()
        logger.info("Startup variables initialized successfully")
        return True


# ============ Main Database Seeder (Facade) ============


class Database_Seeder:
    """Facade for all database seeding operations.

    Provides a simple, high-level API for database initialization
    while maintaining clean separation of concerns internally.

    Example:
        ::

            seeder = Database_Seeder()
            await seeder.create_tables()
            await seeder.populate_startup_data()
    """

    # Foundation publishers we watch: (HF org, family type, derived-search term).
    # The base catalog auto-discovers each org's instruct/chat models (no hand list
    # of model ids); the resolver maps each to its engine-format quant. A new model
    # from a known org appears automatically; a new publisher is just one line here.
    FOUNDATION_ORGS = [
        ("meta-llama", "llama", "Llama"),
        ("Qwen", "qwen", "Qwen"),
        ("mistralai", "mistral", "Mistral"),
        ("google", "gemma", "Gemma"),
        ("deepseek-ai", "deepseek", "DeepSeek"),
        ("microsoft", "phi", "Phi"),
        ("openai", "gpt-oss", "gpt-oss"),
        ("ibm-granite", "granite", "Granite"),
        ("zai-org", "glm", "GLM"),
        ("CohereLabs", "cohere", "Command"),
        ("nvidia", "nemotron", "Nemotron"),
        ("01-ai", "yi", "Yi"),
        ("internlm", "internlm", "InternLM"),
        ("tiiuae", "falcon", "Falcon"),
        ("allenai", "olmo", "OLMo"),
        ("HuggingFaceTB", "smollm", "SmolLM"),
        ("openbmb", "minicpm", "MiniCPM"),
        ("NousResearch", "hermes", "Hermes"),
        ("OpenLLM-France", "lucie", "Lucie"),
    ]

    async def create_tables(self) -> None:
        """Create all database tables from SQLAlchemy models.

        Idempotent operation - safe to call multiple times. Requires
        init_database() to have run first. Anti-B1: reads the LIVE engine via
        attribute access — an imported-by-value `db_engine` would stay frozen
        at None forever.
        """
        if core.db_engine is None:
            raise RuntimeError(
                "Database not initialized: call init_database() before create_tables()"
            )
        try:
            # Create MISSING tables from the models. Schema EVOLUTION of an
            # existing (persisted) database is handled by Alembic at startup
            # (src.database.migrations.run_migrations), not here — this primitive
            # is the from-scratch creation used by tests and first boot.
            Base.metadata.create_all(bind=core.db_engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}", exc_info=True)
            raise

    def build_fresh_catalog(self, model_seeder: "Model_Seeder") -> Tuple[List[Llm], List[Llm]]:
        """Fetch + build the fresh remote catalog (base + derived, deduped) from HF
        for the active engine format. NO DB writes — returns detached Llm objects.

        Shared by the runtime resync (atomic swap) and the build-time snapshot
        generator (src/database/catalog_snapshot.py), so both produce an identical
        catalog from the same discovery + resolver + dedup path.
        """
        fresh_base = model_seeder.build_base_models(self.FOUNDATION_ORGS)
        # Derived: one engine-format search per foundation family + a global
        # top-downloads pass (empty term), so popular community fine-tunes surface
        # whether or not they carry a family name. All runnable by construction.
        searches = [Search_Config(term, ftype, 7.0) for _org, ftype, term in self.FOUNDATION_ORGS]
        searches.append(Search_Config("", "community", 7.0))
        fresh_derived = model_seeder.build_derived_models(
            searches, top_per_search=30, max_checked=200
        )
        # Drop derived rows that are just another quant of a base model (same
        # normalized slug), so each base appears once (as the curated ⭐ entry).
        base_keys = {base_key(m.link) for m in fresh_base}
        fresh_derived = [d for d in fresh_derived if base_key(d.link) not in base_keys]
        return fresh_base, fresh_derived

    # Mutable catalog fields refreshed in place on a resync. supports_tools is
    # excluded on purpose: it is detected post-download and must not be clobbered.
    _RESYNC_FIELDS = (
        "name",
        "type",
        "param_size",
        "model_metadata",
        "quantized",
        "is_base",
        "conversational",
        "category",
        "description",
        "generation_hints",
        "artifact_size_bytes",
    )
    # Fields where a None in the fresh set means "unknown", never "clear": the
    # existing value is preserved (#192 for category; #388 for the sampling
    # facts, which an old snapshot simply does not carry; likewise the real
    # artifact size, absent from snapshots that predate it or estimate-backed).
    _PRESERVE_ON_NONE = ("category", "generation_hints", "artifact_size_bytes")

    def reconcile_remote_catalog(
        self, db: Session, fresh_base: List[Llm], fresh_derived: List[Llm]
    ) -> Dict[str, Any]:
        """Reconcile the remote catalog (local=0) with a fresh model set IN PLACE (#123).

        Matches existing rows by ``link`` (the HF repo id): existing models are
        updated in place, genuinely new ones inserted, and models that vanished from
        the fresh set deleted. Rows are no longer dropped-and-reinserted, so catalog
        IDs stay stable across restarts (the frontend's fetched IDs never go stale).
        Downloaded (local=1) and in-progress (local=2) models are NEVER touched. The
        fresh set is fully loaded BEFORE any write, so a missing/broken source leaves
        the existing catalog intact.
        """
        existing = {row.link: row for row in db.query(Llm).filter(Llm.local == 0).all()}
        added = updated = 0
        seen_links: set = set()
        for fresh in fresh_base + fresh_derived:
            if fresh.link in seen_links:  # guard against dup links in the fresh set
                continue
            seen_links.add(fresh.link)
            current = existing.get(fresh.link)
            if current is None:
                # category is NOT NULL: an unclassified fresh row (pre-#122
                # snapshot / bare fixture) lands on the default bucket — the
                # None -> "general" coalesce applies ONLY at insert (#192).
                if fresh.category is None:
                    fresh.category = "general"
                db.add(fresh)  # genuinely new → insert (new id)
                added += 1
            else:
                for field in self._RESYNC_FIELDS:  # refresh in place → id preserved
                    value = getattr(fresh, field)
                    # An unclassified fresh row (category=None: pre-#122 snapshot
                    # or bare test fixture) must NEVER clobber a classified one —
                    # KEEP the existing category (#192, regression of #184: stale
                    # snapshots collapsed the whole catalog to "general" at every
                    # boot). Fresh rows carrying a REAL category still propagate.
                    if field in self._PRESERVE_ON_NONE and value is None:
                        continue
                    setattr(current, field, value)
                updated += 1

        removed = 0
        for link, row in existing.items():
            if link not in seen_links:  # gone from HF → drop the suggestion
                db.delete(row)
                removed += 1
        db.commit()
        logger.info(
            f"Remote catalog reconciled in place: {added} added, {updated} updated, "
            f"{removed} removed ({len(fresh_base)} base + {len(fresh_derived)} derived; "
            f"downloaded/in-progress untouched)"
        )
        return {
            "base_models_added": added,
            "derived_models_added": updated,
            "base_models_removed": removed,
            "resynced": True,
        }

    def reconcile_catalog_from_snapshot(self, db: Session) -> Dict[str, Any]:
        """Reconcile the remote catalog (local=0) with the BUNDLED snapshot (#131, #163).

        Runs at every boot: the catalog follows app releases (the snapshot ships
        with the build) instead of a live HF resync, so it is fully offline —
        zero network calls, no HF client — and never mutates mid-session. Loads
        the snapshot through the same mechanism as first-boot seeding
        (``load_catalog_snapshot`` + ``dict_to_llm``), keeps the existing catalog
        untouched when no snapshot (or a base-model-less one) is available, and
        stamps ``StartupVariables`` (models_seeded / last_seeded_at / offline_mode)
        on success.
        """
        from src.database.catalog_snapshot import load_catalog_snapshot, dict_to_llm

        tag = getattr(config.LLM_Engine, "FORMAT_TAG", None)
        entries = load_catalog_snapshot(tag) if tag else []
        if not entries:
            logger.info("No bundled catalog snapshot - keeping the existing catalog")
            return {"resynced": False}
        fresh = [dict_to_llm(e) for e in entries]
        fresh_base = [m for m in fresh if m.is_base]
        fresh_derived = [m for m in fresh if not m.is_base]
        if not fresh_base:
            logger.warning("Snapshot carries no base models - keeping the existing catalog")
            return {"resynced": False}
        res = self.reconcile_remote_catalog(db, fresh_base, fresh_derived)
        if res.get("resynced"):
            startup_vars = db.query(StartupVariables).first()
            if startup_vars:
                startup_vars.models_seeded = True
                startup_vars.last_seeded_at = datetime.now()
                # The full snapshot is NOT the degraded JSON fallback: clear the
                # offline flag a fallback-only first boot may have left behind.
                startup_vars.offline_mode = False
                db.commit()
        return res

    def backfill_local_model_sizes(self, db: Session) -> int:
        """One-shot: correct already-downloaded models' displayed size from disk (#220).

        Legacy local rows carry a catalog-time size guess (whole-repo sum or an
        FP16-ish estimate) that was NEVER measured. For every local (local==1)
        model whose weights still exist, measure the real on-disk footprint and
        rewrite ``metadata.size`` / ``disk_size_gb`` only when it diverges from the
        stored value. A handful of local dirs -> a cheap filesystem walk, so boot
        is not slowed measurably. Defensive: missing dirs are skipped silently
        (orphans are legitimate since #225/#208) and any per-row error is
        swallowed so the backfill never blocks boot.

        Returns:
            Number of rows whose metadata was corrected.
        """
        corrected = 0
        measured_cache: Dict[str, float] = {}
        try:
            local_models = db.query(Llm).filter(Llm.local == 1).all()
        except Exception as e:
            logger.warning(f"Model-size backfill skipped (query failed): {e}")
            return 0

        for llm in local_models:
            try:
                link = llm.link
                if not link or not Path(link).is_dir():
                    continue
                measured = measured_cache.get(link)
                if measured is None:
                    measured = measure_dir_size_gb(link)
                    measured_cache[link] = measured
                if measured <= 0:
                    continue
                new_meta = rewrite_size_in_metadata(llm.model_metadata, measured)
                if new_meta != llm.model_metadata:
                    llm.model_metadata = new_meta
                    corrected += 1
            except Exception as e:
                logger.warning(
                    f"Model-size backfill skipped for LLM {getattr(llm, 'id', '?')}: {e}"
                )
                continue

        if corrected:
            db.commit()
        logger.info(f"Model-size backfill: {corrected} local model(s) corrected from disk")
        return corrected

    def backfill_wire_tools(self, db: Session) -> int:
        """One-shot per model: verify the tool-call wire capability (#298).

        Models downloaded before the ``supports_tools_wire`` column existed
        carry NULL (unverified) and therefore route every KB turn systematic.
        For every local (local==1) row still NULL whose artifact exists,
        compute the verdict via the active engine (tokenizer-level, no model
        load) and persist it. A None verdict (engine unavailable, unreadable
        artifact) leaves the row NULL so a transient failure is retried on the
        next boot instead of being pinned to False. Shared links (a KB
        assistant and its base) are computed once. Defensive like
        ``backfill_local_model_sizes``: any per-row error is swallowed so the
        backfill never takes the app down.

        Returns:
            Number of rows whose wire capability was persisted.
        """
        from src.domains.llms.repository import detect_wire_tools

        updated = 0
        verdict_cache: Dict[str, Optional[bool]] = {}
        try:
            pending = db.query(Llm).filter(Llm.local == 1, Llm.supports_tools_wire.is_(None)).all()
        except Exception as e:
            logger.warning(f"Wire-capability backfill skipped (query failed): {e}")
            return 0

        for llm in pending:
            try:
                link = llm.link
                if not link or not Path(link).exists():
                    continue
                if link in verdict_cache:
                    verdict = verdict_cache[link]
                else:
                    verdict = detect_wire_tools(link)
                    verdict_cache[link] = verdict
                if verdict is None:
                    continue
                llm.supports_tools_wire = verdict
                updated += 1
            except Exception as e:
                logger.warning(
                    f"Wire-capability backfill skipped for LLM {getattr(llm, 'id', '?')}: {e}"
                )
                continue

        if updated:
            db.commit()
        logger.info(f"Wire-capability backfill: {updated} local model(s) verified")
        return updated

    async def populate_startup_data(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Populate database with startup data.

        Args:
            db: Optional database session. If None, creates new session.

        Returns:
            Dictionary with operation results and counts.

        Raises:
            Exception: If any critical seeding step fails.
        """
        should_close = db is None
        if db is None:
            db = SessionLocal()

        try:
            results = {
                "base_models_added": 0,
                "derived_models_added": 0,
                "jobs_cleaned": {},
                "hardware_initialized": False,
                "startup_vars_initialized": False,
                "offline_mode": False,
                "models_seeded": False,
            }

            # Initialize startup variables first
            logger.info("Initializing startup variables...")
            startup_init = Startup_Initializer(db)
            results["startup_vars_initialized"] = startup_init.initialize_if_needed()

            # Get or create startup variables singleton
            startup_vars = db.query(StartupVariables).first()
            if not startup_vars:
                startup_vars = StartupVariables()
                db.add(startup_vars)
                db.commit()
                db.refresh(startup_vars)

            # Reconcile the catalog with the bundled snapshot at EVERY boot
            # (#131, #163): zero network, so downloads are never starved by a
            # live HF resync and the catalog never mutates mid-session — it
            # follows app releases instead.
            reconcile = self.reconcile_catalog_from_snapshot(db)
            if reconcile.get("resynced"):
                results["base_models_added"] = reconcile.get("base_models_added", 0)
                results["derived_models_added"] = reconcile.get("derived_models_added", 0)
                results["models_seeded"] = True
            elif db.query(Llm).filter(Llm.local == 0).count() == 0:
                # No snapshot bundled (e.g. bare dev tree) and the catalog is
                # still empty → minimal offline fallback so the app has models.
                results["base_models_added"] = Model_Seeder(
                    db, offline_mode=True
                ).seed_initial_catalog()
            results["offline_mode"] = False

            # Cleanup jobs (always run)
            logger.info("Cleaning up unfinished jobs...")
            job_cleanup = Job_Cleanup_Service(db)
            results["jobs_cleaned"] = job_cleanup.cleanup_all_unfinished_jobs()

            # Correct already-downloaded models' displayed size from disk (#220):
            # the catalog value was a guess, never measured. Runs after orphan
            # cleanup so a to-be-removed dir is never measured; cheap and defensive.
            self.backfill_local_model_sizes(db)

            # Initialize hardware
            logger.info("Initializing hardware info...")
            hw_init = Hardware_Initializer(db)
            results["hardware_initialized"] = hw_init.initialize_if_needed()

            logger.info(
                f"Startup population completed: "
                f"base={results['base_models_added']}, "
                f"derived={results['derived_models_added']}, "
                f"jobs_cleaned={sum(results['jobs_cleaned'].values())}, "
                f"offline_mode={results['offline_mode']}, "
                f"models_seeded={results['models_seeded']}"
            )

            return results

        except Exception as e:
            logger.error(f"Error during startup population: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    async def delete_all_data(self) -> None:
        """Delete all data from database and file storage.

        **DESTRUCTIVE OPERATION - DEVELOPMENT ONLY**

        Requires interactive confirmation before proceeding.

        Warning:
            Never expose this in production. No undo available.
        """
        logger.warning("Preparing to delete all data from the database")
        response = input("Are you sure you want to delete ALL data? (yes/no): ")

        if response.lower() not in ("yes", "y"):
            logger.info("Database deletion cancelled")
            return

        db = SessionLocal()
        try:
            logger.warning("Deleting all data...")

            # Delete file storage
            self._delete_storage_directories()

            # Delete database records
            db.query(StartupVariables).delete()
            db.query(KBJobModel).delete()
            db.query(KnowledgeDocument).delete()
            db.query(KnowledgeBase).delete()
            db.query(HardwareProfile).delete()
            db.query(DownloadJobModel).delete()
            db.query(Message).delete()
            db.query(Conversation).delete()
            db.query(Llm).delete()

            db.commit()
            logger.warning("All data deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting data: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()

    def _delete_storage_directories(self) -> None:
        """Delete and recreate storage directories."""
        directories = [str(config.LLM_DIR)]

        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Recreated directory: {directory}")


# ============ Legacy API Compatibility ============


async def create_tables() -> None:
    """Legacy API: Create database tables.

    Deprecated: Use Database_Seeder().create_tables() instead.
    """
    seeder = Database_Seeder()
    await seeder.create_tables()


async def startup_populate_database() -> Dict[str, Any]:
    """Populate startup data and return the result dict.

    The catalog is reconciled from the bundled snapshot inside (zero network,
    #131/#163) — there is nothing left for the caller to schedule.
    """
    seeder = Database_Seeder()
    return await seeder.populate_startup_data()


def backfill_wire_tools_startup() -> int:
    """Threadpool entrypoint for the post-ready wire-capability backfill (#298).

    Verifying the wire capability loads a tokenizer per unverified model
    (seconds each, GGUF especially), so it must NOT run inside the awaited
    boot sequence — the lifespan schedules this AFTER the app is ready, in a
    threadpool (same rationale as the post-ready catalog resync of #109).
    Opens and closes its own session; swallows everything: NULL rows simply
    keep routing systematic until a later boot verifies them.
    """
    db = None
    try:
        db = SessionLocal()
        return Database_Seeder().backfill_wire_tools(db)
    except Exception as e:
        # Runs after ready in a task nobody awaits: this is its only record.
        logger.warning(f"Wire-capability backfill failed: {e}", exc_info=True)
        return 0
    finally:
        if db is not None:
            db.close()


async def delete_all_data() -> None:
    """Legacy API: Delete all data.

    Deprecated: Use Database_Seeder().delete_all_data() instead.
    """
    seeder = Database_Seeder()
    await seeder.delete_all_data()
