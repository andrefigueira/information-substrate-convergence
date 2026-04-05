"""
ISC Reasoning API Server

A FastAPI server exposing ISC reasoning capabilities for LLM integration.

Run with:
    uvicorn isc.api_server:app --reload --port 8000

Or:
    python -m isc.api_server
"""

import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .reasoning_api import ISCReasoner, ReasoningType, ReasoningResult, KnowledgeEntry


# Pydantic models for API
class ReasoningRequest(BaseModel):
    """Request body for reasoning endpoint"""
    query: str = Field(..., description="The question or reasoning task")
    context: Optional[List[str]] = Field(default=None, description="Additional context/premises")
    reasoning_type: str = Field(default="auto", description="Type of reasoning: auto, deductive, inductive, abductive, causal, analogical")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "If all mammals are warm-blooded and dogs are mammals, are dogs warm-blooded?",
                "context": ["All mammals are warm-blooded", "Dogs are mammals"],
                "reasoning_type": "deductive"
            }
        }


class KnowledgeRequest(BaseModel):
    """Request body for adding knowledge"""
    entries: List[Dict[str, Any]] = Field(..., description="Knowledge entries to add")

    class Config:
        json_schema_extra = {
            "example": {
                "entries": [
                    {"content": "Paris is the capital of France", "confidence": 1.0},
                    {"content": "France is in Europe", "confidence": 1.0}
                ]
            }
        }


class ReasoningResponse(BaseModel):
    """Response from reasoning endpoint"""
    answer: str
    confidence: float
    reasoning_chain: List[str]
    reasoning_type: str
    phi_scores: Dict[str, float]
    latency_ms: float
    emergent_concepts: List[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    substrate_active: bool


class StatsResponse(BaseModel):
    """Statistics response"""
    knowledge_entries: int
    total_queries: int
    avg_latency_ms: float
    reasoning_type_distribution: Dict[str, int]


# Initialize FastAPI app
app = FastAPI(
    title="ISC Reasoning API",
    description="""
    Information Substrate Convergence (ISC) Reasoning API

    A cognitive reasoning substrate for LLM augmentation. Provides structured
    reasoning with phi-calibrated confidence scores.

    ## Features

    - **Multi-type reasoning**: deductive, inductive, abductive, causal, analogical
    - **Phi-based confidence**: Confidence scores derived from information integration metrics
    - **Reasoning chains**: Full explanation of reasoning steps
    - **Emergent concepts**: Detection of new concepts formed during reasoning
    - **Knowledge persistence**: Add and accumulate knowledge over time

    ## Use Cases

    - Augment LLM responses with structured reasoning
    - Multi-hop inference for complex queries
    - Causal reasoning and what-if analysis
    - Confidence calibration for AI outputs
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reasoner instance (singleton pattern)
_reasoner: Optional[ISCReasoner] = None
_stats = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "reasoning_types": {}
}


def get_reasoner() -> ISCReasoner:
    """Get or create the global reasoner instance"""
    global _reasoner
    if _reasoner is None:
        _reasoner = ISCReasoner(verbose=False)
    return _reasoner


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check and API info"""
    reasoner = get_reasoner()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        substrate_active=reasoner.substrate is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    reasoner = get_reasoner()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        substrate_active=reasoner.substrate is not None
    )


@app.post("/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest):
    """
    Perform reasoning over a query with optional context.

    Returns structured reasoning result with:
    - answer: The reasoned answer
    - confidence: Phi-calibrated confidence score (0-1)
    - reasoning_chain: Step-by-step reasoning explanation
    - phi_scores: Detailed phi metrics
    - emergent_concepts: New concepts formed during reasoning
    """
    global _stats

    reasoner = get_reasoner()

    # Map reasoning type string to enum
    try:
        rt = ReasoningType(request.reasoning_type) if request.reasoning_type != "auto" else ReasoningType.AUTO
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reasoning_type. Must be one of: auto, deductive, inductive, abductive, causal, analogical"
        )

    # Perform reasoning
    result = reasoner.reason(
        query=request.query,
        context=request.context,
        reasoning_type=rt
    )

    # Update stats
    _stats["total_queries"] += 1
    _stats["total_latency_ms"] += result.latency_ms
    rt_name = result.reasoning_type
    _stats["reasoning_types"][rt_name] = _stats["reasoning_types"].get(rt_name, 0) + 1

    return ReasoningResponse(
        answer=result.answer,
        confidence=result.confidence,
        reasoning_chain=result.reasoning_chain,
        reasoning_type=result.reasoning_type,
        phi_scores=result.phi_scores,
        latency_ms=result.latency_ms,
        emergent_concepts=result.emergent_concepts
    )


@app.post("/knowledge")
async def add_knowledge(request: KnowledgeRequest):
    """
    Add knowledge entries to the reasoning substrate.

    Knowledge persists across reasoning queries and enriches future reasoning.
    """
    reasoner = get_reasoner()

    entries = []
    for e in request.entries:
        entry = KnowledgeEntry(
            content=e.get("content", ""),
            source=e.get("source"),
            confidence=e.get("confidence", 1.0),
            relationships=e.get("relationships", [])
        )
        entries.append(entry)

    added = reasoner.add_knowledge(entries)

    return {
        "status": "success",
        "entries_added": added,
        "total_knowledge": len(reasoner.knowledge_base)
    }


@app.get("/knowledge")
async def get_knowledge():
    """Get current knowledge base contents"""
    reasoner = get_reasoner()

    return {
        "total_entries": len(reasoner.knowledge_base),
        "entries": [
            {
                "content": e.content,
                "source": e.source,
                "confidence": e.confidence
            }
            for e in reasoner.knowledge_base
        ]
    }


@app.delete("/knowledge")
async def clear_knowledge():
    """Clear all knowledge from the substrate"""
    reasoner = get_reasoner()
    count = len(reasoner.knowledge_base)
    reasoner.knowledge_base = []

    return {
        "status": "success",
        "entries_cleared": count
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics"""
    reasoner = get_reasoner()

    avg_latency = 0.0
    if _stats["total_queries"] > 0:
        avg_latency = _stats["total_latency_ms"] / _stats["total_queries"]

    return StatsResponse(
        knowledge_entries=len(reasoner.knowledge_base),
        total_queries=_stats["total_queries"],
        avg_latency_ms=avg_latency,
        reasoning_type_distribution=_stats["reasoning_types"]
    )


@app.get("/reasoning-types")
async def get_reasoning_types():
    """Get available reasoning types with descriptions"""
    return {
        "types": [
            {
                "name": "auto",
                "description": "Automatically detect the best reasoning type from the query"
            },
            {
                "name": "deductive",
                "description": "Given premises, what must logically follow? (e.g., syllogisms)"
            },
            {
                "name": "inductive",
                "description": "Given examples, what pattern emerges? (e.g., generalizations)"
            },
            {
                "name": "abductive",
                "description": "Given observations, what best explains them? (e.g., diagnosis)"
            },
            {
                "name": "causal",
                "description": "If X happens, what effect on Y? (e.g., what-if analysis)"
            },
            {
                "name": "analogical",
                "description": "How is A like B? (e.g., metaphorical reasoning)"
            }
        ]
    }


# Batch reasoning endpoint for efficiency
class BatchReasoningRequest(BaseModel):
    """Request body for batch reasoning"""
    queries: List[ReasoningRequest]


@app.post("/reason/batch")
async def reason_batch(request: BatchReasoningRequest):
    """
    Perform reasoning on multiple queries in a single request.

    More efficient than individual calls for processing multiple questions.
    """
    results = []
    for query_request in request.queries:
        try:
            result = await reason(query_request)
            results.append({"status": "success", "result": result})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "total": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


def main():
    """Run the API server"""
    uvicorn.run(
        "isc.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
