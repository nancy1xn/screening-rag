import typing as t
from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
import json
from qdrant_client.models import ScoredPoint

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)

from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")


subject = "Binance"
question = {
            "Q1 Subject Background": [
                f"q1_1 When was the company {subject} founded?",
                f"q1_2 Which country is the company {subject} headquartered in?",
                "q1_3 Is the company {subject} publicly listed?",
                "q1_4 What is the {subject}'s stock ticker?",
                "q1_5 What type of business does the company {subject} provide?"
            ],
            "Q2 Adverse Information Report Headline (ordered by timeline)": [
                "q2_1 Has the company {subject} committed any crimes?",
                "q2_2 When did the company {subject} commit a crime (the specific date, month, year)?",
                "q2_3 What type of crime did the company {subject} commit?"
            ]
}

saved_chunks = []

for question_key, question_value in question.items():
    question_openai_vectors = embeddings.embed_documents(question_value)
    question_openai_vectors: t.List[List[float]]
    
# print(question_openai_vectors)
# print(type(question_openai_vectors))

# search
    for question_index, question_vector in enumerate(question_openai_vectors):
        search_results = client.query_points(
            collection_name="cnn_news_chunk_vectors",
            query=question_vector,
            limit=1
        )
        # print("Search Results:", search_results)
        for result in search_results:
            saved_chunks.append({
                "question": question_value[question_index],
                "score": ..., #invoke structure output -> relevance.score
                "text": result.payload["text"]
                # or result["payload"]["text"]
            })
            ...



#search
# for question_vector in question_openai_vectors:
#     results = client.query_points(
#         collection_name="cnn_news_chunk_vectors",
#         query=question_vector,
        # query_filter=models.Filter(
        # must=[
        #     models.FieldCondition(
        #         key="text",
        #         match=models.MatchValue(
        #             value="Binance",
        #         ),
        #     )
        # ]
        # ),
    #     limit=5
    # )

# print(results)
