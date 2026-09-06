"""REST API endpoints for Knowledge Base creation and RAG attachment to LLMs.

Atomic endpoints following REST principles. Each endpoint delegates business
logic to services.

Architecture:
    Endpoints -> Services -> Repository -> Database

Endpoints:
    GET  /knowledge_base/{llm_id}/status - Poll KB job status
    POST /knowledge_base/create - Create new KB assistant or update existing
"""

from typing import List
from fastapi import BackgroundTasks, Depends, APIRouter
from sqlalchemy.orm import Session

from src.database.core import get_db, SessionLocal
from src.domains.knowledge_base.services import KB_Service
from src.domains.knowledge_base.schemas import KnowledgeBaseCreate, KnowledgeBaseResponse
from src.core.logging import logger
from src.ingestion.embedding_model import download_state, start_download
from src.core.exceptions import (
    AppBaseException,
    KnowledgeBaseNotFoundException,
    DatabaseException,
    InvalidInputException,
    StateConflictException,
)
from fastapi import status as http_status


router = APIRouter(prefix="/knowledge_base", tags=["knowledge_base"])


# Declared BEFORE the parametric /{llm_id}/status so "embedding-model" is not
# captured as an llm_id (which would 422). The KB needs the e5 embedding model;
# these gate its on-demand download so a fresh/offline install doesn't fail its
# first KB use silently. #146.
@router.get("/embedding-model/status")
def get_embedding_model_status():
    """On-disk presence of the KB embedding model + any in-flight download."""
    return download_state()


@router.post("/embedding-model/download")
def post_embedding_model_download():
    """Kick off the embedding-model download in the background (idempotent)."""
    return start_download()


@router.get("/{llm_id}/status")
def get_kb_job_status(llm_id: int, db: Session = Depends(get_db)):
    """Poll Knowledge Base job status with automatic cleanup for failed jobs.

    Args:
        llm_id: Specialized LLM ID created during KB creation.
        db: Database session.

    Returns:
        dict: status, status_updated_at, error_message

    Raises:
        KnowledgeBaseNotFoundException: If KB job not found.
        DatabaseException: If error fetching status.
    """
    service = KB_Service()

    try:
        status_data = service.get_kb_job_status(db, llm_id)
        return status_data

    except ValueError:
        # Polled by the UI: an absent job is a 404 the handler logs at INFO.
        raise KnowledgeBaseNotFoundException(llm_id)

    except Exception as e:
        raise DatabaseException("Error fetching KB job status", trace=str(e))


@router.post("/create", response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    payload: KnowledgeBaseCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Create new Knowledge Base assistant or update existing one.

    Decision tree:
    - If base LLM has NO KB attached: Create new specialized LLM + KB
    - If base LLM HAS KB attached: Update existing KB with new documents

    A background task then ingests the documents asynchronously (extraction →
    chunking → embeddings → vector store); the polled status endpoint reports
    progress and errors.

    Args:
        payload: Request body with selectedModel, modelName, description, paths.
        background_tasks: FastAPI background task queue.
        db: Database session.

    Returns:
        KnowledgeBaseResponse: msg and model_id

    Raises:
        InvalidInputException: If validation fails.
        KnowledgeBaseNotFoundException: If base LLM not found.
        DatabaseException: If KB creation fails.
    """
    # Validate payload
    if not payload.paths or not isinstance(payload.paths, list):
        raise InvalidInputException("paths (must be non-empty list)")

    if not payload.selectedModel:
        raise InvalidInputException("selectedModel")

    if not payload.modelName:
        raise InvalidInputException("modelName")

    logger.info(
        f"KB creation request: base_llm={payload.selectedModel}, "
        f"name={payload.modelName}, files={len(payload.paths)}"
    )

    service = KB_Service()

    try:
        # Check if base LLM has existing KB
        from src.domains.knowledge_base.repository import KB_Repository

        repo = KB_Repository()
        base_llm = repo.get_local_llm_by_id(db, payload.selectedModel)

        if not base_llm:
            raise KnowledgeBaseNotFoundException(payload.selectedModel)

        if base_llm.is_attached_to_kb:
            # Update existing KB
            logger.info(f"Updating existing KB for LLM {base_llm.id}")

            llm_id, kb_job_id = service.update_existing_kb(
                db=db, base_llm_id=base_llm.id, file_paths=payload.paths
            )

            # Queue background task for update
            background_tasks.add_task(
                _run_kb_update_task, kb_job_id=kb_job_id, file_paths=payload.paths
            )

            return KnowledgeBaseResponse(
                msg="Knowledge Base is being updated with new documents.", model_id=llm_id
            )

        else:
            # Create new KB assistant. Refuse a name already carried by any
            # installed model (#317): two identical names are indistinguishable
            # in every picker. The update path above never reads the name, so
            # updating an assistant with its own (existing) name stays valid.
            duplicate = repo.get_local_llm_by_name(db, payload.modelName)
            if duplicate:
                raise StateConflictException(
                    f"A local model named '{payload.modelName.strip()}' already "
                    "exists. Choose a different assistant name.",
                    status_code=http_status.HTTP_409_CONFLICT,
                )

            logger.info(f"Creating new KB assistant from base LLM {base_llm.id}")

            llm_id, kb_job_id = service.create_kb_assistant(
                db=db,
                base_llm_id=base_llm.id,
                model_name=payload.modelName,
                description=payload.description or "",
                file_paths=payload.paths,
            )

            # Queue background task for creation
            background_tasks.add_task(
                _run_kb_creation_task, kb_job_id=kb_job_id, file_paths=payload.paths
            )

            return KnowledgeBaseResponse(
                msg="Knowledge Base Assistant is being created.", model_id=llm_id
            )

    except AppBaseException:
        raise

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise KnowledgeBaseNotFoundException(payload.selectedModel)

    except Exception as e:
        raise DatabaseException("Error creating Knowledge Base Assistant", trace=str(e))


def _run_kb_creation_task(kb_job_id: int, file_paths: List[str]) -> None:
    """Background task: ingest documents and index them into the vector store.

    Args:
        kb_job_id: KBJob ID to track progress.
        file_paths: List of file paths to process.
    """
    db = SessionLocal()
    service = KB_Service()

    try:
        service.process_and_index_documents(
            db=db, kb_job_id=kb_job_id, file_paths=file_paths, is_update=False
        )
    except Exception as e:
        # The task boundary: nobody awaits a BackgroundTask, so this record
        # (with the traceback) is the only trace of the failure in the log.
        # The job row already says "failed" to the polling UI.
        logger.error(f"KB creation task failed for job {kb_job_id}: {e}", exc_info=True)
    finally:
        db.close()


def _run_kb_update_task(kb_job_id: int, file_paths: List[str]) -> None:
    """Background task: ingest new documents into the existing KB.

    Args:
        kb_job_id: KBJob ID to track progress.
        file_paths: List of new file paths to add.
    """
    db = SessionLocal()
    service = KB_Service()

    try:
        service.process_and_index_documents(
            db=db, kb_job_id=kb_job_id, file_paths=file_paths, is_update=True
        )
    except Exception as e:
        logger.error(f"KB update task failed for job {kb_job_id}: {e}", exc_info=True)
    finally:
        db.close()
