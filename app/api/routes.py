from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

service = None
cache = None


def set_service(s, c):
    global service
    global cache
    service = s
    cache = c


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_endpoint(req: QueryRequest):

    return service.process_query(req.query)


@router.get("/cache/stats")
def cache_stats():

    return cache.stats()


@router.delete("/cache")
def flush_cache():

    cache.clear()

    return {"status": "cache cleared"}