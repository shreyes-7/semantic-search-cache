from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query_endpoint(req: QueryRequest):

    result = service.process_query(req.query)

    return result


@router.get("/cache/stats")
def cache_stats():

    return cache.stats()


@router.delete("/cache")
def flush_cache():

    cache.clear()

    return {"status":"cache cleared"}