import typing as t
from typing import List

import more_itertools as mit
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models.models import QueryResponse

from screening_rag.custom_types import Crime
from screening_rag.db import Settings

settings = Settings()


def reset_and_create_cnn_news_qdrant_data_storage():
    client = QdrantClient(url=settings.QDRANT_DOMAIN)
    client.delete_collection(collection_name="cnn_news_chunk_vectors")
    client.create_collection(
        collection_name="cnn_news_chunk_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )


def reset_and_create_crime_qdrant_data_storage():
    client = QdrantClient(url=settings.QDRANT_DOMAIN)
    client.delete_collection(collection_name="crime_cnn_news_vectors")
    client.create_collection(
        collection_name="crime_cnn_news_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )


def process_and_insert_cnn_news_chunks_to_qdrant(chunk, article_id, chunk_id):
    unique_id = 0
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    text_openai_vectors = embeddings.embed_documents([chunk[0]])
    text_openai_vectors: t.List[List[float]]

    for each_text_openai_vectors in text_openai_vectors:
        client = QdrantClient(url=settings.QDRANT_DOMAIN)
        client.upsert(
            collection_name="cnn_news_chunk_vectors",
            points=[
                models.PointStruct(
                    id=chunk_id,
                    payload={
                        "unique_id": chunk_id,
                        "text": chunk,
                        "article_id": article_id,
                    },
                    vector=each_text_openai_vectors,
                ),
            ],
        )
        unique_id += 1


def process_and_insert_crime_to_qdrant(crime: Crime):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    client = QdrantClient(url=settings.QDRANT_DOMAIN)
    crime_openai_vectors = embeddings.embed_documents([str(crime.summary)])
    crime_openai_vectors: t.List[t.List[float]]
    crime_openai_vector = crime_openai_vectors[0]
    client.upsert(
        collection_name="crime_cnn_news_vectors",
        points=[
            models.PointStruct(
                id=crime.id,
                payload={
                    "id": crime.id,
                    "time": crime.time,
                    "subjects": crime.subjects,
                    "summary": crime.summary,
                    "adverse_info_type": crime.adverse_info_type,
                    "violated_laws": crime.violated_laws,
                    "enforcement_action": crime.enforcement_action,
                },
                vector=crime_openai_vector,
            ),
        ],
    )


def get_points_similar_to_embedding(
    query: str,
    collection_name: str,
    limit: int,
    embedding_model: t.Optional[str] = "text-embedding-3-large",
    dimentions: t.Optional[int] = 3072,
    score_threshold: t.Optional[float] = None,
    extra_qdrant_conditions: qdrant_client.conversions.common_types.Filter = None,
) -> QueryResponse:
    embedder = OpenAIEmbeddings(model=embedding_model, dimensions=dimentions)
    client = QdrantClient(url=settings.QDRANT_DOMAIN)
    embedding = mit.one(embedder.embed_documents([query]))
    embedding: t.List[List[float]]
    return client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=extra_qdrant_conditions,
    )
