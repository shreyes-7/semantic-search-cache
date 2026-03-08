# Semantic Search with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **production-style semantic search system**
built with modern machine learning and backend engineering practices.\
The system combines **sentence embeddings, vector search, fuzzy
clustering, and semantic caching** to efficiently answer user queries.

The project exposes a **FastAPI service** that allows clients to query a
large text corpus using semantic similarity rather than keyword
matching.

The system architecture is designed to be:

-   Scalable
-   Production-ready
-   Efficient in query retrieval
-   Optimized using clustering-aware caching

This project was built as part of an **AI/ML Engineering assignment**.

------------------------------------------------------------------------

# System Architecture

The system pipeline works as follows:

User Query\
↓\
Sentence Embedding Model\
↓\
Fuzzy Cluster Identification\
↓\
Cluster-aware Semantic Cache\
↓\
Vector Search (FAISS) if cache miss\
↓\
Return Ranked Results

### Architecture Diagram

    User Query
        │
        ▼
    Embedding Model (Sentence Transformer)
        │
        ▼
    Cluster Detection (Gaussian Mixture Model)
        │
        ▼
    Semantic Cache (Cluster Aware)
       ├── Cache Hit → Return Cached Result
       └── Cache Miss
               │
               ▼
           FAISS Vector Search
               │
               ▼
         Store Result in Cache
               │
               ▼
            Return Result

------------------------------------------------------------------------

# Key Components

## 1. Sentence Embeddings

The system uses the **Sentence Transformers model**:

    sentence-transformers/all-MiniLM-L6-v2

Reasons for selecting this model:

-   Lightweight (384 dimensional embeddings)
-   Fast inference
-   Strong semantic similarity performance
-   Suitable for CPU environments

Embeddings allow semantically similar sentences to have similar vector
representations.

Example:

  Query                    Similar Meaning
  ------------------------ ----------------------------
  "space shuttle launch"   "rocket launch mission"
  "NASA satellite"         "space research satellite"

------------------------------------------------------------------------

## 2. Vector Database (FAISS)

To efficiently search through thousands of documents, the project uses
**Facebook AI Similarity Search (FAISS)**.

FAISS enables:

-   Fast nearest neighbor search
-   Efficient high dimensional vector indexing
-   Low latency semantic search

Documents are stored as:

    384 dimensional embedding vectors

During search:

1.  Query is embedded
2.  FAISS returns nearest vectors
3.  Corresponding documents are returned

------------------------------------------------------------------------

## 3. Fuzzy Clustering

Instead of assigning documents to a single cluster, this project uses
**soft clustering** via:

    Gaussian Mixture Model (GMM)

Why fuzzy clustering?

Documents may belong to multiple topics.

Example:

    Document: "NASA satellite launch using new computer systems"

Possible clusters:

    Cluster 3 → Space
    Cluster 8 → Technology

GMM provides **probability membership** across clusters.

------------------------------------------------------------------------

## 4. Semantic Cache

The semantic cache stores previously answered queries.

Instead of exact matching, the cache performs **cosine similarity search
between embeddings**.

Example:

Query 1:

    space shuttle launch

Query 2:

    rocket launch mission

Even though the queries differ lexically, they are semantically similar.

Cache Hit occurs if:

    cosine_similarity(query1, query2) > threshold

Current threshold:

    0.85

------------------------------------------------------------------------

## 5. Cluster-aware Cache Optimization

To improve performance when the cache grows large, the cache is
organized **per cluster**.

Structure:

    cache = {
        cluster_1 : [entries],
        cluster_2 : [entries]
    }

This reduces cache lookup time from:

    O(N)

to

    O(N / number_of_clusters)

This design improves scalability.

------------------------------------------------------------------------

# Project Structure

    semantic-search-cache
    │
    ├── app
    │   ├── api
    │   │   └── routes.py
    │   │
    │   ├── cache
    │   │   └── semantic_cache.py
    │   │
    │   ├── clustering
    │   │   └── fuzzy_cluster.py
    │   │
    │   ├── embeddings
    │   │   └── embedder.py
    │   │
    │   ├── services
    │   │   └── query_service.py
    │   │
    │   ├── vectorstore
    │   │   └── faiss_store.py
    │   │
    │   └── main.py
    │
    ├── scripts
    │   └── build_index.py
    │
    ├── data
    │   └── processed
    │       └── corpus.csv
    │
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md

------------------------------------------------------------------------

# Installation Guide

## Step 1 --- Clone the repository

    git clone <repository-url>
    cd semantic-search-cache

------------------------------------------------------------------------

## Step 2 --- Create Python Virtual Environment

    python -m venv venv

Activate environment:

Windows:

    venv\Scripts\activate

Linux/Mac:

    source venv/bin/activate

------------------------------------------------------------------------

## Step 3 --- Install Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

# Running the Service

The service is designed to start with a single command.

    uvicorn app.main:app

### Automatic Index Building

If the vector index does not exist, the system automatically runs:

    scripts/build_index.py

This generates:

    models/
     ├ embeddings.npy
     ├ faiss.index
     ├ cluster_model.pkl
     └ documents.pkl

Once built, startup becomes instant.

------------------------------------------------------------------------

# API Endpoints

## Query Endpoint

    POST /query

Example request:

    {
     "query": "NASA space launch"
    }

Example response:

    {
     "query": "NASA space launch",
     "cache_hit": false,
     "similarity_score": 0.67,
     "result": "...",
     "dominant_cluster": 9
    }

------------------------------------------------------------------------

## Cache Statistics

    GET /cache/stats

Example:

    {
     "total_entries": 42,
     "hit_count": 17,
     "miss_count": 25,
     "hit_rate": 0.40
    }

------------------------------------------------------------------------

## Clear Cache

    DELETE /cache

------------------------------------------------------------------------

# Docker Deployment

Build image:

    docker build -t semantic-search .

Run container:

    docker run -p 8000:8000 semantic-search

Access API:

    http://localhost:8000/docs

------------------------------------------------------------------------

# Performance Optimizations

This system includes several optimizations:

-   Persistent vector index
-   Cluster-aware cache lookup
-   Lightweight embedding model
-   Fast vector search using FAISS

------------------------------------------------------------------------

# Future Improvements

Potential enhancements:

-   distributed vector database
-   GPU embedding inference
-   hybrid search (BM25 + embeddings)
-   Redis semantic cache
-   streaming responses

------------------------------------------------------------------------

# Conclusion

This project demonstrates a **production-grade semantic search system**
integrating:

-   modern NLP embeddings
-   vector similarity search
-   fuzzy clustering
-   semantic caching
-   REST API deployment

The architecture is designed for scalability and real-world ML systems.
