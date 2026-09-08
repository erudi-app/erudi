"""LLM model management API endpoints for browsing, downloading, and deleting models.

This module provides REST endpoints for:
- Listing available LLMs (all, local only, remote only)
- Fetching individual LLM details
- Downloading models from HuggingFace
- Deleting local models and freeing disk space
- Tracking download progress via background jobs
- Searching models by name

Model States:
    - local=0: Remote model (available on HuggingFace, not downloaded)
    - local=1: Local model (downloaded and ready for inference)
    - local=2: Downloading (background job in progress)

Architecture:
    LLM Lifecycle:
    ┌────────────────────────────────────────────────────────────┐
    │ GET /llms/remote                                           │
    │  └─> Browse available models from HuggingFace              │
    └────────────────────────────────────────────────────────────┘
                            ↓
    ┌────────────────────────────────────────────────────────────┐
    │ POST /llms/{llm_id}/download                               │
    │  1. Create local LLM record (local=2)                      │
    │  2. Create DownloadJob record                              │
    │  3. Start background download task                         │
    └────────────────────────────────────────────────────────────┘
                            ↓
    ┌────────────────────────────────────────────────────────────┐
    │ GET /llms/download_jobs/{job_id}                           │
    │  └─> Poll download progress (percentage, ETA)              │
    └────────────────────────────────────────────────────────────┘
                            ↓
    ┌────────────────────────────────────────────────────────────┐
    │ GET /llms/local                                            │
    │  └─> List downloaded models (local=1)                      │
    └────────────────────────────────────────────────────────────┘

Download Process:
    1. List the repo files and refuse the repo if it ships no artefact in the
       engine's format (MLX repo tag / .gguf file) -- before any byte (#408)
    2. Fetch the pre-built quant from HuggingFace (no local conversion)
    3. Save to config.LLM_DIR/{llm_id}/
    4. Update LLM record: local=1, link=local_path
    5. Mark DownloadJob as completed

Endpoints:
    - GET / → List all LLMs
    - GET /local → List local (downloaded) LLMs
    - GET /remote → List remote (HuggingFace) LLMs
    - GET /{llm_id} → Get LLM details
    - GET /search?name=<query> → Search LLMs by name
    - POST /{llm_id}/download → Start model download
    - DELETE /{llm_id} → Delete local model files
    - PUT /{llm_id} → Update LLM metadata
    - GET /download_jobs → List all download jobs
    - GET /download_jobs/{job_id} → Get download job status
    - DELETE /download_jobs/{job_id} → Cancel download job

Example:
    Download and use a model::

        # 1. List available remote models
        GET /erudi/llms/remote
        → [{"id": 42, "name": "Mistral-7B", "local": 0, ...}]

        # 2. Start download
        POST /erudi/llms/42/download
        → {"job_id": 123, "status": "running", "progress": 0.0}

        # 3. Poll progress
        GET /erudi/llms/download_jobs/123
        → {"job_id": 123, "status": "running", "progress": 45.2, "eta_seconds": 120}

        # 4. Wait for completion
        GET /erudi/llms/download_jobs/123
        → {"job_id": 123, "status": "completed", "progress": 100.0}

        # 5. Use model in conversation
        POST /erudi/conversations/
        {"llm_id": 42, ...}

Note:
    - Downloads run in background via FastAPI BackgroundTasks
    - Large models (7B+) take 5-30 minutes depending on connection
    - Only pre-built quants are downloaded (MLX repos / GGUF files); nothing is
      converted or quantized locally
    - Disk space check performed before download

Warning:
    DELETE /{llm_id} permanently removes model files. Cannot be undone.
    Ensure model is not in use (check conversations) before deleting.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, APIRouter, status as http_status
from sqlalchemy.orm import Session
from src.database.core import get_db, SessionLocal

from src.entities.DownloadJob import DownloadJobModel
from src.entities.Llm import Llm
from src.domains.llms.schemas import (
    LLMCreate,
    LLMResponse,
    DownloadJobResponse,
    HFSearchResult,
    HFDownloadRequest,
    DependentsResponse,
    RebindRequest,
)
from src.domains.llms.services import download_llm, cancel_download_job
from src.domains.llms.hf_search import search_huggingface
from src.domains.llms.repository import Llm_Repository, Download_Job_Repository
from src.domains.knowledge_base.repository import COPIED_FIELDS
from src.utils.hf_model_metadata import (
    BYTES_PER_GB,
    humanize_model_name,
    measure_dir_size_bytes,
    rewrite_size_in_metadata,
)

from src.core.logging import logger
from src.core import config
from src.database.generation_hints import (
    capture_generation_hints,
    read_local_generation_hints,
    resolve_base_repo,
)
from src.core.exceptions import (
    ModelNotFoundException,
    DatabaseException,
    InvalidInputException,
    StateConflictException,
    FileSystemException,
    DownloadJobNotFoundException,
)

router = APIRouter(prefix="/llms", tags=["llms"])


# ============ Dependency Injection ============


def get_llm_repository(db: Session = Depends(get_db)) -> Llm_Repository:
    """Dependency injection for Llm_Repository.

    Args:
        db: Database session from FastAPI.

    Returns:
        Llm_Repository: Repository instance with injected session.
    """
    return Llm_Repository(db)


def get_download_job_repository(db: Session = Depends(get_db)) -> Download_Job_Repository:
    """Dependency injection for Download_Job_Repository.

    Args:
        db: Database session from FastAPI.

    Returns:
        Download_Job_Repository: Repository instance with injected session.
    """
    return Download_Job_Repository(db)


# ============ Download Helpers (shared by id-based and link-based downloads) ============


def _assert_downloaded_artifact_ok(final_save_dir, temp_save_dir) -> None:
    """Integrity gate at download completion (#88).

    Validate the just-downloaded model via the ACTIVE engine (it knows which
    files its build type actually needs) BEFORE the model can become selectable.
    On failure, remove the downloaded artifacts -- mirroring ``delete_llm``:
    rmtree the final model dir and the temp dir -- then re-raise so the caller
    finalizes the job in its error state carrying the engine's explicit,
    user-facing message. A model therefore never reaches ``local=1`` without
    passing validation, and a bad artifact never lingers on disk.
    """
    validator = getattr(config.LLM_Engine, "validate_local_artifact", None)
    if validator is None:
        return
    try:
        validator(str(final_save_dir))
    except Exception:
        for stale_dir in (final_save_dir, temp_save_dir):
            try:
                if stale_dir and os.path.exists(str(stale_dir)):
                    shutil.rmtree(str(stale_dir), ignore_errors=True)
            except Exception:
                logger.warning(f"Integrity cleanup could not remove {stale_dir}")
        raise


def _run_download_task(
    model_link: str, model_id: int, temp_save_dir, final_save_dir, job_id: int
) -> None:
    """Background body of a download: flip the job to running, run the download
    (which spawns its own progress updater), and mark failed on error. Module-level
    so both the by-id and by-link download routes share one implementation."""
    from src.domains.llms.repository import (
        detect_supports_tools,
        detect_supports_vision,
        detect_wire_tools,
    )

    session = SessionLocal()
    try:
        job_obj = session.query(DownloadJobModel).get(job_id)
        job_obj.status = "running"
        session.commit()
        logger.info(f"Started download job {job_id}")
        asyncio.run(
            download_llm(
                model_link=model_link,
                model_id=model_id,
                temp_save_dir=temp_save_dir,
                final_save_dir=final_save_dir,
                job_id=job_id,
            )
        )
        job_obj = session.query(DownloadJobModel).get(job_id)
        if job_obj is None:
            logger.warning(f"Download job {job_id} row vanished before finalization")
        elif job_obj.status == "cancelled":
            # The cancel endpoint already finalized the row and cleaned the temp
            # dir; the transfer aborted on its flag (#377). Say so instead of
            # claiming success for a job the user stopped.
            logger.info(f"Download job {job_id} was cancelled; transfer stopped, nothing finalized")
        else:
            # Integrity gate (#88): a model must never flip to local=1 without its
            # essential files. On failure this cleans the artifacts and raises, so
            # the except below finalizes the job as failed with an explicit message.
            _assert_downloaded_artifact_ok(final_save_dir, temp_save_dir)

            # Finalize FIRST, before any capability probe (#291/#313). The transfer
            # is done and the artifact has passed the integrity gate, so the job is
            # genuinely complete at this point. Probing before committing is what
            # left job 40 stuck at "running" with a complete 9GB model on disk: the
            # probe hung, the UI polled a non-terminal status forever, and the next
            # boot's cleanup deleted the artifact. Capabilities are refinements of a
            # finished download, never preconditions for calling it finished.
            llm_obj = session.query(Llm).get(model_id)
            if llm_obj:
                llm_obj.local = 1
                # Measure the real footprint BEFORE the job goes terminal (#349).
                # This is a plain directory walk, not a probe -- no subprocess, no
                # model load -- so it costs milliseconds and does not reintroduce
                # what the comment above forbids. It has to happen here because the
                # UI refreshes the instant it polls "completed": behind the probes
                # the rewrite landed ~10s late, and the freshly installed card sat
                # on the catalog's pre-download guess ("Size: ~4.0 GB" against a
                # measured ~4.7 GB) until the page was reloaded by hand.
                # One walk, in bytes: the exact figure goes to
                # artifact_size_bytes (the installed row is exact even when the
                # catalog only had an estimate) and its GB form to the metadata
                # string the cards parse.
                try:
                    measured_bytes = measure_dir_size_bytes(final_save_dir)
                    if measured_bytes > 0:
                        llm_obj.artifact_size_bytes = measured_bytes
                        llm_obj.model_metadata = rewrite_size_in_metadata(
                            llm_obj.model_metadata, measured_bytes / BYTES_PER_GB
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not measure the on-disk size of LLM {model_id}; "
                        f"leaving the catalog estimate in place: {e}",
                        exc_info=True,
                    )
            job_obj.status = "completed"
            job_obj.progress = 100.0
            session.commit()

            # Best-effort enrichment. A failure here leaves the model usable with
            # unset capabilities (NULL = unknown, which every consumer already
            # treats conservatively) instead of stranding a finished download.
            if llm_obj:
                try:
                    llm_obj.supports_tools = detect_supports_tools(llm_obj.link)
                    # Verified wire capability (#298): does the engine's server
                    # actually parse this model's tool calls? Gates agentic KB.
                    llm_obj.supports_tools_wire = detect_wire_tools(llm_obj.link)
                    llm_obj.supports_vision = detect_supports_vision(llm_obj.link)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.warning(
                        f"Capability detection failed for LLM {model_id} after a "
                        f"successful download; leaving capabilities unset: {e}",
                        exc_info=True,
                    )
                # Sampling facts (#388), same best-effort stance. Catalog rows
                # already carry the base repo's hints (copied at _start_download);
                # a by-link download reads the artifact first (MLX dirs ship
                # generation_config.json) and only then asks the network.
                if getattr(llm_obj, "generation_hints", None) is None:
                    try:
                        hints = _capture_hints_for_download(model_link, final_save_dir)
                        if hints:
                            llm_obj.generation_hints = hints
                            session.commit()
                    except Exception as e:
                        session.rollback()
                        logger.warning(
                            f"Generation hints capture failed for LLM {model_id} after a "
                            f"successful download; keeping the fallback sampling: {e}",
                            exc_info=True,
                        )
            logger.info(f"Download job {job_id} completed successfully")
    except Exception as e:
        # The task boundary: nobody awaits a BackgroundTask, so this is the
        # record of the failed download, with the model and the traceback.
        logger.exception(f"Download job {job_id} failed for LLM {model_id} ({model_link}): {e}")
        job_obj = session.query(DownloadJobModel).get(job_id)
        job_obj.status = "failed"
        job_obj.error_message = str(e)
        session.commit()
    finally:
        session.close()


def _capture_hints_for_download(model_link: str, final_save_dir) -> Optional[dict]:
    """Sampling facts for a just-downloaded model (#388): the local artifact
    first (offline, MLX dirs ship the files), and when it carries no usable
    sampling value the network cascade -- the base repo named by the quant's
    ``base_model:*`` card tag (else the repo itself) with the repo as the
    quant stage. Facts read from the artifact fill what the network lacks."""
    local = read_local_generation_hints(Path(final_save_dir), base_repo=model_link)
    if local and local.get("generation_config"):
        return local
    hf_api = config.get_hf_api()
    base_repo = model_link
    try:
        info = hf_api.model_info(model_link)
        base_repo = resolve_base_repo(model_link, getattr(info, "tags", None))
    except Exception as e:
        logger.info(f"Could not read the card of {model_link} for its base model: {e}")
    hints = capture_generation_hints(base_repo, hf_api, quant_repo=model_link)
    if hints is None and base_repo != model_link:
        hints = capture_generation_hints(model_link, hf_api)
    if hints is None:
        return local
    if local:
        for fact in ("context_length", "supports_thinking"):
            if hints.get(fact) is None and local.get(fact) is not None:
                hints[fact] = local[fact]
    return hints


def _start_download(
    *,
    remote_model_id: str,
    remote_link: str,
    name: str,
    type: str,
    description,
    model_metadata,
    quantized: bool,
    param_size: Optional[float],
    category: str,
    llm_repo: Llm_Repository,
    job_repo: Download_Job_Repository,
    db: Session,
    background_tasks: BackgroundTasks,
    generation_hints: Optional[dict] = None,
    artifact_size_bytes: Optional[int] = None,
) -> DownloadJobModel:
    """Create the local=2 placeholder + DownloadJob and enqueue the download.

    Shared by the catalog (by-id) and HF-search (by-link) download routes so the
    placeholder/job/enqueue logic lives in exactly one place. ``remote_link`` is the
    HF repo id actually fetched; ``remote_model_id`` is the catalog id (by-id) or the
    repo id (by-link), for traceability on the job.
    """
    local_llm = llm_repo.create(
        name=name,
        local=2,
        type=type,
        description=description,
        model_metadata=model_metadata,
        quantized=quantized,
        param_size=param_size,
        category=category,
        # Sampling facts ride along from the catalog row (#388); a by-link
        # download has none yet and gets them post-download, best-effort.
        generation_hints=generation_hints,
        # The catalog's real size keeps the card exact while the download runs;
        # completion overwrites it with the measured footprint either way.
        artifact_size_bytes=artifact_size_bytes,
    )
    temp_path = config.LLM_DIR / f"temp_{local_llm.id}"
    final_path = config.LLM_DIR / str(local_llm.id)
    if temp_path.exists() or final_path.exists():
        llm_repo.delete(local_llm)
        db.rollback()
        raise FileSystemException("Model path already exists - delete existing files first")
    llm_repo.update(local_llm, link=str(final_path))
    db.commit()
    logger.info(f"Created local LLM entry {local_llm.id}: {local_llm.name} -> {final_path}")

    job = job_repo.create(
        remote_model_id=remote_model_id,
        local_model_id=local_llm.id,
        remote_model_link=remote_link,
        temp_local_model_link=str(temp_path),
        final_local_model_link=str(final_path),
        status="pending",
    )
    db.commit()
    logger.info(f"Created download job {job.id} for model {local_llm.name}")
    background_tasks.add_task(
        _run_download_task, remote_link, local_llm.id, temp_path, final_path, job.id
    )
    return job


# ============ Dependents Helper ============


def _dependents_payload(llm_repo: Llm_Repository, base_llm: Llm) -> dict:
    """Build the dependents payload for a model (assistants + conversation counts).

    Shared by GET /{id}/dependents and the guarded base delete so both always
    agree on the exact shape the frontend renders. ``assistants`` are the KB
    assistants sharing the model's weights (COPIED link, #209); counts cover the
    conversations a delete would orphan.
    """
    assistants = llm_repo.get_dependent_assistants(base_llm)
    entries = [
        {
            "id": a.id,
            "name": a.name,
            "kb_id": a.kb_id,
            "conversation_count": llm_repo.count_conversations(a.id),
        }
        for a in assistants
    ]
    own = llm_repo.count_conversations(base_llm.id)
    total = own + sum(e["conversation_count"] for e in entries)
    return {
        "assistants": entries,
        "own_conversation_count": own,
        "total_conversation_count": total,
    }


# ============ LLM CRUD Endpoints ============


@router.get("/", response_model=List[LLMResponse])
async def get_all_llms(llm_repo: Llm_Repository = Depends(get_llm_repository)):
    """List all LLMs (local and remote).

    Args:
        llm_repo: Injected LLM repository.

    Returns:
        List[LLMResponse]: All LLM models with metadata.
    """
    # Read-only operation, no commit needed
    llms = llm_repo.get_all()
    return llms


@router.get("/local", response_model=List[LLMResponse])
async def get_local_llms(llm_repo: Llm_Repository = Depends(get_llm_repository)):
    """List only local (downloaded) LLMs ready for inference.

    Args:
        llm_repo: Injected LLM repository.

    Returns:
        List[LLMResponse]: LLMs with local=1.
    """
    # Read-only operation, no commit needed
    llms = llm_repo.get_all_local()
    return llms


@router.get("/remote", response_model=List[LLMResponse])
async def get_remote_llms(llm_repo: Llm_Repository = Depends(get_llm_repository)):
    """List only remote (HuggingFace) LLMs available for download.

    Args:
        llm_repo: Injected LLM repository.

    Returns:
        List[LLMResponse]: LLMs with local=0.
    """
    # Read-only operation, no commit needed
    llms = llm_repo.get_all_remote()
    return llms


@router.get("/search", response_model=List[LLMResponse])
async def search_llms(name: str, llm_repo: Llm_Repository = Depends(get_llm_repository)):
    """Search LLMs by name (case-insensitive partial match).

    Args:
        name: Search query string.
        llm_repo: Injected LLM repository.

    Returns:
        List[LLMResponse]: Matching LLMs.
    """
    # Read-only operation, no commit needed
    llms = llm_repo.search_by_name(name)
    return llms


@router.get("/search/huggingface", response_model=List[HFSearchResult])
async def search_huggingface_route(q: str, limit: int = 30):
    """Live HuggingFace search beyond the curated catalog (#122 follow-up).

    Searches HF directly for runnable models matching ``q`` in the active engine's
    format, filtered to chat/vision LLMs (so a query like "french" doesn't return
    token-classification repos). Results are ephemeral — they are NOT added to the
    catalog; download a chosen one via POST /download/huggingface.

    Args:
        q: Free-text query (model name, family, trait like "uncensored").
        limit: Max results to return (default 30).

    Returns:
        List[HFSearchResult]: Downloadable matches with metadata + category.
    """
    return search_huggingface(q, limit=limit)


@router.get("/{llm_id}", response_model=LLMResponse)
async def get_llm_by_id(llm_id: int, llm_repo: Llm_Repository = Depends(get_llm_repository)):
    """Get LLM details by ID.

    Args:
        llm_id: ID of the LLM to retrieve.
        llm_repo: Injected LLM repository.

    Returns:
        LLMResponse: LLM metadata.

    Raises:
        ModelNotFoundException: If LLM not found.
    """
    # Read-only operation, no commit needed
    llm = llm_repo.get_by_id(llm_id)
    if not llm:
        raise ModelNotFoundException(f"LLM {llm_id}")
    return llm


@router.get("/{llm_id}/dependents", response_model=DependentsResponse)
async def get_llm_dependents(
    llm_id: int,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
):
    """List the KB assistants that depend on this model's weights, plus the
    conversation counts a delete would orphan.

    Dependents share the model's COPIED ``link`` (#209): deleting the base
    leaves their weights dangling. A model with no dependents returns an empty
    list with zero-or-more conversation counts. The frontend calls this before a
    base-model delete to render the choice dialog.

    A KB assistant NEVER has dependents (#317): deleting it is a direct 200
    that keeps the shared weights (they belong to the base), so sibling
    assistants copying the same link are not reported — this endpoint answers
    with the exact truth the guarded DELETE's 409 uses.

    Args:
        llm_id: ID of the (base) model to inspect.
        llm_repo: Injected LLM repository.

    Returns:
        DependentsResponse: assistants + own/total conversation counts.

    Raises:
        ModelNotFoundException: If the model is not found.
    """
    llm = llm_repo.get_by_id(llm_id)
    if not llm:
        raise ModelNotFoundException(f"LLM {llm_id}")
    if llm.is_attached_to_kb:
        own = llm_repo.count_conversations(llm.id)
        return {
            "assistants": [],
            "own_conversation_count": own,
            "total_conversation_count": own,
        }
    return _dependents_payload(llm_repo, llm)


@router.put("/{llm_id}", response_model=LLMResponse)
async def update_llm(
    llm_id: int,
    llm_data: LLMCreate,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    db: Session = Depends(get_db),
):
    """Update LLM metadata (name, description, etc.).

    Args:
        llm_id: ID of the LLM to update.
        llm_data: LLMCreate schema with new values.
        llm_repo: Injected LLM repository.
        db: Database session for transaction control.

    Returns:
        LLMResponse: Updated LLM.

    Raises:
        ModelNotFoundException: If LLM not found.
        DatabaseException: If update fails.
    """
    try:
        llm = llm_repo.get_by_id(llm_id)
        if not llm:
            raise ModelNotFoundException(f"LLM {llm_id}")

        # Update fields from request
        updated_llm = llm_repo.update(llm, **llm_data.dict())
        db.commit()
        return updated_llm
    except ModelNotFoundException:
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to update LLM", trace=str(e))


@router.delete("/{llm_id}")
async def delete_llm(
    llm_id: int,
    orphan_dependents: bool = False,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    db: Session = Depends(get_db),
):
    """Delete local LLM (permanent deletion).

    Regular local model: model files are removed from disk along with the
    database record. If the model has dependent KB assistants (rows sharing its
    COPIED ``link``, #209), the delete is guarded: without ``orphan_dependents``
    it raises 409 carrying the dependents payload so the client can offer to
    orphan them and proceed. With ``orphan_dependents=true`` the base is deleted,
    its assistants REMAIN (their link now dangles, re-bindable via /rebind), and
    every conversation bound to the base is nulled server-side (SET NULL, #225).

    Specialized KB assistant: its ``link`` is a COPY of the base model's (set at
    creation) — the files belong to the base model and are left untouched; the
    assistant's KnowledgeBase is deleted instead, and server-side cascades sweep
    its documents and ``rag.kb_chunks``. Its own conversations survive (SET NULL).

    Args:
        llm_id: ID of the LLM to delete.
        orphan_dependents: When true, delete a base model that has dependent KB
            assistants anyway (they become orphans). Ignored for assistants and
            for base models with no dependents.
        llm_repo: Injected LLM repository.
        db: Database session for transaction control.

    Returns:
        dict: Success message.

    Raises:
        ModelNotFoundException: If LLM not found.
        StateConflictException: 400 if currently downloading; 409 (with the
            dependents payload) if a base model has dependents and
            ``orphan_dependents`` was not set.
        DatabaseException: If deletion fails.

    Warning:
        Deletes model files from disk (regular models). Cannot be undone.
    """
    try:
        llm = llm_repo.get_by_id(llm_id)
        if not llm:
            raise ModelNotFoundException(f"LLM {llm_id}")

        if llm.local == 2:
            raise StateConflictException("Cannot delete LLM while downloading")

        if llm.is_attached_to_kb:
            # KB assistant: never touch the (shared) model files. Deleting
            # the KB cascades to its documents/chunks, and the ORM cascade
            # (KnowledgeBase.llm, delete-orphan) removes the assistant row.
            # The assistant's conversations survive server-side (llm_id SET NULL).
            from src.entities.KnowledgeBase import KnowledgeBase

            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == llm.kb_id).first()
            if kb:
                db.delete(kb)
            else:
                # Inconsistent state (flag set, KB gone): drop the row alone.
                llm_repo.delete(llm)
            db.commit()
            logger.info(f"Deleted KB assistant {llm_id} (KB {llm.kb_id}, files kept)")
            return {"message": "LLM deleted successfully"}

        # Base model: guard against silently orphaning KB assistants that share
        # its weights. Without opt-in, surface the dependents (409) so the client
        # can confirm; with orphan_dependents=true, proceed and let them dangle.
        dependents = _dependents_payload(llm_repo, llm)
        if dependents["assistants"] and not orphan_dependents:
            raise StateConflictException(
                f"Base model '{llm.name}' has {len(dependents['assistants'])} "
                "dependent KB assistant(s). Retry with orphan_dependents=true to "
                "delete it anyway; the assistants remain and can be rebound to "
                "another base.",
                status_code=http_status.HTTP_409_CONFLICT,
                detail=dependents,
            )

        # Delete files from disk if they exist
        if llm.link and os.path.exists(llm.link):
            shutil.rmtree(llm.link, ignore_errors=True)
            logger.info(f"Deleted model files: {llm.link}")

            # Check and delete residual temp files (e.g., temp_36 for llm.link = data/models/36)
            temp_path = config.LLM_DIR / f"temp_{llm.id}"
            if os.path.exists(str(temp_path)):
                shutil.rmtree(str(temp_path), ignore_errors=True)
                logger.warning(f"Deleting residual temp files associated to llm: {temp_path}")

        # Delete database record
        llm_repo.delete(llm)
        db.commit()

        return {"message": "LLM deleted successfully"}

    except (ModelNotFoundException, StateConflictException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to delete LLM", trace=str(e))


@router.post("/{assistant_id}/rebind", response_model=LLMResponse)
async def rebind_assistant(
    assistant_id: int,
    payload: RebindRequest,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    db: Session = Depends(get_db),
):
    """Re-point an orphaned KB assistant at a new base model.

    Re-copies the base's weights ``link`` plus every descriptive COPIED_FIELDS
    entry (#209) onto the assistant, keeping its own name/description and KB
    wiring (kb_id, is_attached_to_kb). Used to recover an assistant whose base
    was deleted (weights_available False). The target must be a local,
    non-assistant model whose weights exist on disk.

    Args:
        assistant_id: ID of the KB assistant to rebind.
        payload: RebindRequest with ``new_base_llm_id``.
        llm_repo: Injected LLM repository.
        db: Database session for transaction control.

    Returns:
        LLMResponse: The updated assistant (weights_available now True).

    Raises:
        ModelNotFoundException: 404 if the assistant or target base is missing.
        StateConflictException: 409 if the assistant is not a KB assistant, or
            the target is itself an assistant / not downloaded / has no weights.
        DatabaseException: If the rebind fails.
    """
    try:
        assistant = llm_repo.get_by_id(assistant_id)
        if not assistant:
            raise ModelNotFoundException(f"LLM {assistant_id}")
        if not assistant.is_attached_to_kb:
            raise StateConflictException(
                f"Model '{assistant.name}' is not a KB assistant; only KB "
                "assistants can be rebound to a new base.",
                status_code=http_status.HTTP_409_CONFLICT,
            )

        new_base = llm_repo.get_by_id(payload.new_base_llm_id)
        if not new_base:
            raise ModelNotFoundException(f"LLM {payload.new_base_llm_id}")
        if new_base.is_attached_to_kb:
            raise StateConflictException(
                f"Target model '{new_base.name}' is itself a KB assistant; pick "
                "a standalone base model.",
                status_code=http_status.HTTP_409_CONFLICT,
            )
        if new_base.local != 1:
            raise StateConflictException(
                f"Target model '{new_base.name}' is not downloaded; its weights "
                "must exist to rebind onto it.",
                status_code=http_status.HTTP_409_CONFLICT,
            )
        if not new_base.link or not os.path.exists(new_base.link):
            raise StateConflictException(
                f"Target model '{new_base.name}' weights are missing on disk; "
                "cannot rebind onto it.",
                status_code=http_status.HTTP_409_CONFLICT,
            )

        # Re-copy link + descriptive columns; name/description/KB wiring untouched.
        updated = llm_repo.update(
            assistant,
            **{field: getattr(new_base, field) for field in COPIED_FIELDS},
        )
        db.commit()
        db.refresh(updated)
        logger.info(f"Rebound assistant {assistant_id} onto base {new_base.id}")
        return updated

    except (ModelNotFoundException, StateConflictException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to rebind assistant", trace=str(e))


# ============ Download Management Endpoints ============


@router.post(
    "/{llm_id}/download",
    response_model=DownloadJobResponse,
    status_code=http_status.HTTP_200_OK,
)
async def download_llm_route(
    llm_id: int,
    background_tasks: BackgroundTasks,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    job_repo: Download_Job_Repository = Depends(get_download_job_repository),
    db: Session = Depends(get_db),
):
    """Start background download of LLM from HuggingFace.

    Creates a DownloadJob and runs download in background. The catalog link is
    already a pre-built quant for this engine (MLX repo on Apple Silicon, GGUF
    file elsewhere); nothing is converted locally.

    Args:
        llm_id: ID of the remote LLM to download.
        background_tasks: FastAPI background tasks manager.
        llm_repo: Injected LLM repository.
        job_repo: Injected download job repository.
        db: Database session for transaction control.

    Returns:
        DownloadJobResponse: Job record with job_id, status, progress.

    Raises:
        HTTPException: 404 if LLM not found, 500 if paths already exist.

    Note:
        Download progress can be polled via GET /downloads/{job_id}/status.
        Large models (7B+) take 5-30 minutes.
    """
    try:
        # Get remote LLM metadata
        remote_llm = llm_repo.get_by_id(llm_id)
        if not remote_llm:
            raise ModelNotFoundException(f"LLM {llm_id}")

        return _start_download(
            remote_model_id=str(llm_id),
            remote_link=remote_llm.link,
            name=remote_llm.name,
            type=remote_llm.type,
            description=remote_llm.description,
            model_metadata=remote_llm.model_metadata,
            quantized=remote_llm.quantized,
            param_size=remote_llm.param_size,
            category=getattr(remote_llm, "category", "general"),
            llm_repo=llm_repo,
            job_repo=job_repo,
            db=db,
            background_tasks=background_tasks,
            generation_hints=getattr(remote_llm, "generation_hints", None),
            artifact_size_bytes=remote_llm.artifact_size_bytes,
        )

    except (ModelNotFoundException, InvalidInputException, FileSystemException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to start download", trace=str(e))


@router.post(
    "/download/huggingface",
    response_model=DownloadJobResponse,
    status_code=http_status.HTTP_200_OK,
)
async def download_huggingface_route(
    payload: HFDownloadRequest,
    background_tasks: BackgroundTasks,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    job_repo: Download_Job_Repository = Depends(get_download_job_repository),
    db: Session = Depends(get_db),
):
    """Download a model chosen from HF search by its repo id (not via the catalog).

    Mirrors POST /{llm_id}/download but takes the HF repo id directly, so search
    results never need to be persisted into the catalog. Creates the local=2
    placeholder + DownloadJob and runs the download in the background.

    Args:
        payload: The chosen HF repo id plus display metadata from the search hit.

    Returns:
        DownloadJobResponse: Job record to poll via GET /downloads/{job_id}/status.
    """
    try:
        name = payload.name or humanize_model_name(payload.link)
        # Family type is metadata-only for a local row; fall back to the category.
        type_ = payload.type or payload.category or "community"
        return _start_download(
            remote_model_id=payload.link,
            remote_link=payload.link,
            name=name,
            type=type_,
            description=None,
            model_metadata=None,
            quantized=payload.quantized,
            param_size=payload.param_size,
            category=payload.category,
            llm_repo=llm_repo,
            job_repo=job_repo,
            db=db,
            background_tasks=background_tasks,
        )
    except (FileSystemException, InvalidInputException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to start download", trace=str(e))


@router.post(
    "/downloads/{job_id}/cancel",
    status_code=200,
)
def cancel_download(
    job_id: int,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    job_repo: Download_Job_Repository = Depends(get_download_job_repository),
    db: Session = Depends(get_db),
):
    """Cancel an active download job and cleanup partial files."""
    try:
        return cancel_download_job(job_id, job_repo, llm_repo, db)
    except (
        DownloadJobNotFoundException,
        ModelNotFoundException,
        InvalidInputException,
        StateConflictException,
    ):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to cancel download", trace=str(e))


@router.get(
    "/downloads/{job_id}/status",
    response_model=DownloadJobResponse,
    status_code=200,
)
def get_download_status_by_jobId(
    job_id: int,
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    job_repo: Download_Job_Repository = Depends(get_download_job_repository),
    db: Session = Depends(get_db),
):
    """Get download job status by ID with automatic cleanup for failed/cancelled jobs.

    Polls the download job status and performs cleanup operations for terminal states
    (failed/cancelled/completed). Failed/cancelled jobs delete the temp LLM entry and
    clean up temporary files. Completed jobs mark the LLM as local=1 (ready).

    Args:
        job_id: The database ID of the download job to query.
        llm_repo: Injected LLM repository.
        job_repo: Injected download job repository.
        db: Database session for transaction control.

    Returns:
        DownloadJobResponse: Download job with current status, progress, ETA, and file paths.

    Raises:
        DownloadJobNotFoundException: If job_id not found.
        ModelNotFoundException: If a completed job's associated LLM is missing.
        DatabaseException: If status fetch fails.

    Example:
        GET /llms/downloads/42/status
        Response: {"id": 42, "status": "running", "progress": 65.0, "eta_seconds": 120, ...}
    """
    try:
        job = job_repo.get_by_id(job_id)
        if not job:
            raise DownloadJobNotFoundException(job_id)

        # Handle failed/cancelled jobs: cleanup temp files and LLM entry.
        # `cancel_download_job` already deletes the temp LLM (and Postgres
        # nulls local_model_id through the FK) before the job reaches
        # `cancelled`, so a missing row here is the common case, not an
        # error - only run cleanup when there is still something to clean.
        if job.status in ["failed", "cancelled"]:
            llm = llm_repo.get_by_id(job.local_model_id)
            if llm:
                # Delete temp LLM entry
                llm_repo.delete(llm)

                # Clean up temp files using repository method
                job_repo.cleanup_job_files(job)

                db.commit()
                db.refresh(job)
                logger.info(f"Cleaned up {job.status} download job {job_id}")

        # Handle completed jobs: mark LLM as ready, ONCE.
        #
        # The frontend polls this endpoint every 2s while a download is in
        # flight, and keeps polling a completed job until it navigates away. The
        # write below used to run on every one of those ticks: a GET handler
        # issuing an UPDATE plus a COMMIT, and an INFO log line, for a row that
        # already said local = 1. Guarding on the current value makes the write
        # idempotent in the real sense (it happens once) rather than merely
        # convergent (it happens forever but lands on the same value).
        elif job.status == "completed":
            llm = llm_repo.get_by_id(job.local_model_id)
            if not llm:
                raise ModelNotFoundException(f"LLM {job.local_model_id}")
            if llm.local != 1:
                llm_repo.update(llm, local=1)
                db.commit()
                db.refresh(llm)
                logger.info(f"Marked LLM {llm.id} as ready (download job {job_id} completed)")

        return job

    except (DownloadJobNotFoundException, ModelNotFoundException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to get download status", trace=str(e))


@router.get(
    "/downloads/status",
    response_model=DownloadJobResponse,
    status_code=200,
)
def get_download_status_without_jobId(
    llm_repo: Llm_Repository = Depends(get_llm_repository),
    job_repo: Download_Job_Repository = Depends(get_download_job_repository),
    db: Session = Depends(get_db),
):
    """Get most recent active download job (for single-download UI polling).

    Finds the most recently updated download job that is still running/pending within
    the last 60 seconds. Useful for UIs that only support one active download at a time
    and need to poll without tracking job IDs. Performs same cleanup as get_by_id.

    Args:
        llm_repo: Injected LLM repository.
        job_repo: Injected download job repository.
        db: Database session for transaction control.

    Returns:
        DownloadJobResponse: The most recent active job with status and progress.

    Raises:
        DownloadJobNotFoundException: If no active job found in last 60 seconds.
        ModelNotFoundException: If a completed job's associated LLM is missing.
        DatabaseException: If status fetch fails.

    Example:
        GET /llms/downloads/status
        Response: {"id": 42, "status": "running", "progress": 65.0, ...}
    """
    try:
        # Get most recent active job
        job = job_repo.get_most_recent_active()
        if not job:
            raise DownloadJobNotFoundException("recent active")

        # Handle failed jobs: cleanup temp files and LLM entry. Same
        # already-cleaned-up guard as get_download_status_by_jobId - see #509.
        if job.status == "failed":
            llm = llm_repo.get_by_id(job.local_model_id)
            if llm:
                # Delete temp LLM entry
                llm_repo.delete(llm)

                # Clean up temp files using repository method
                job_repo.cleanup_job_files(job)

                db.commit()
                db.refresh(job)
                logger.info(f"Cleaned up failed download job {job.id}")

        # Handle completed jobs: mark LLM as ready, ONCE. Same idempotency
        # guard as get_download_status_by_jobId - see the comment there.
        elif job.status == "completed":
            llm = llm_repo.get_by_id(job.local_model_id)
            if not llm:
                raise ModelNotFoundException(f"LLM {job.local_model_id}")
            if llm.local != 1:
                llm_repo.update(llm, local=1)
                db.commit()
                db.refresh(llm)
                logger.info(f"Marked LLM {llm.id} as ready (download job {job.id} completed)")

        return job

    except (DownloadJobNotFoundException, ModelNotFoundException):
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to get download status", trace=str(e))
