import typing as t
from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")


question = "Putin money laundering"

#embed
question_openai_vectors = embeddings.embed_documents(question)
question_openai_vectors: t.List[List[float]]
# print(question_openai_vectors)
# print(type(question_openai_vectors))

#search
for question_vector in question_openai_vectors:
    results = client.query_points(
        collection_name="cnn_news_chunk_vectors",
        query=question_vector,
        # query_filter=models.Filter(
        # must=[
        #     models.FieldCondition(
        #         key="text",
        #         match=models.MatchValue(
        #             value="Putin",
        #         ),
        #     )
        # ]
        # ),
        limit=10
    )

print(results)