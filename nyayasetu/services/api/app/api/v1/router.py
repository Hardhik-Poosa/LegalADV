"""API v1 routes for NyayaSetu guidance, ingestion, and health checks."""
from __future__ import annotations

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.ingestion.exceptions import IngestionError
from app.ingestion.loader import LegalDocumentLoader
from app.ingestion.section_parser import ActSectionParser
from app.llm.llm_client import LMStudioClient, LMStudioConfig
from app.rag.citation_verifier import CitationVerificationError
from app.rag.retriever import HybridRetriever
from app.schemas.guidance import GuidanceRequest, IngestRequest, IngestResponse
from app.services.answer_pipeline import AnswerPipeline, LegalAnswerResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["NyayaSetu API"])


def get_llm_client() -> LMStudioClient:
    """Dependency provider for LMStudioClient using application settings."""
    settings = get_settings()
    cfg = LMStudioConfig(
        base_url=settings.lm_studio_base_url,
        chat_model=settings.lm_studio_chat_model,
        embedding_model=settings.lm_studio_embedding_model,
        timeout_s=settings.lm_studio_timeout_s,
        temperature=settings.lm_studio_temperature,
        max_tokens=settings.lm_studio_max_tokens,
    )
    return LMStudioClient(cfg)


@router.get("/health", summary="Health Check")
async def health_check(
    db: AsyncSession = Depends(get_db),
    llm: LMStudioClient = Depends(get_llm_client),
) -> dict[str, Any]:
    """Check database connectivity and LM Studio service status."""
    db_ok = False
    try:
        res = await db.execute(text("SELECT 1"))
        db_ok = res.scalar() == 1
    except Exception as exc:
        logger.warning("Health check DB connection failed: %s", exc)

    llm_ok = llm.health()

    status_str = "healthy" if (db_ok and llm_ok) else "degraded"
    return {
        "status": status_str,
        "database": "connected" if db_ok else "disconnected",
        "llm_service": "connected" if llm_ok else "unreachable",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@router.post(
    "/guidance",
    response_model=LegalAnswerResponse,
    summary="Get Verified Legal Guidance & Risk Awareness",
)
async def get_legal_guidance(
    req: GuidanceRequest,
    db: AsyncSession = Depends(get_db),
    llm: LMStudioClient = Depends(get_llm_client),
) -> LegalAnswerResponse:
    """Execute legal decision support pipeline: extract facts -> hybrid retrieve -> synthesize answer -> verify citations."""
    retriever = HybridRetriever(session=db, embedder=llm)
    pipeline = AnswerPipeline(retriever=retriever, llm_client=llm)

    try:
        response = await pipeline.run(
            user_query=req.query,
            jurisdiction=req.jurisdiction,
            as_of_date=req.as_of_date,
        )
        return response
    except CitationVerificationError as exc:
        logger.error("Citation verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Generated answer failed citation verification: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error in guidance endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Guidance pipeline error: {exc}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Legal Act Text into Knowledge Base",
)
async def ingest_legal_act(
    req: IngestRequest,
    db: AsyncSession = Depends(get_db),
    llm: LMStudioClient = Depends(get_llm_client),
) -> IngestResponse:
    """Parse raw Act text, compute embeddings, and store in PostgreSQL vector store."""
    parser = ActSectionParser()
    try:
        sections = parser.parse(act_name=req.act_name, text=req.raw_text)
    except Exception as exc:
        logger.error("Parse error for %s: %s", req.act_name, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse Act section structure: {exc}",
        )

    effective_dt = req.effective_from or datetime.date(1872, 1, 1)
    loader = LegalDocumentLoader(session=db, embedder=llm)

    try:
        result = await loader.load(
            act_name=req.act_name,
            full_title=req.full_title or req.act_name,
            jurisdiction=req.jurisdiction,
            effective_from=effective_dt,
            sections=sections,
            version=req.version,
            raw_text_for_checksum=req.raw_text,
        )
        return IngestResponse(
            document_id=result.document_id,
            act_name=req.act_name,
            jurisdiction=req.jurisdiction,
            version=req.version,
            sections_loaded=result.sections_loaded,
            chunks_created=result.chunks_created,
            checksum=result.checksum,
        )
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ingestion error: {exc}",
        )
