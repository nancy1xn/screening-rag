import json
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import numpy as np
import time

with open ("src/article_filter2.json", "r") as article_file:
    news_article_collection = json.load(article_file)

def chunk_text(text:str) ->list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=50,
    )
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter()
    chunks = []
    for section in text_split:
        chunks.extend(token_splitter.split_text(section))
    return chunks

news_article_collection_result_string = "\n".join(json.dumps(d) for d in news_article_collection)
chunks_example = chunk_text(news_article_collection_result_string)
#print(chunks_example)

#Langchain OpenAI Embedding 

# import os
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

start = time.time()
openai_vectors = embeddings.embed_documents(chunks_example)


end = time.time()
print(openai_vectors)
print(f"Embedding Time:{end - start}")
print(f"Embedding Dimension: {len(openai_vectors[0])}")

#create collections
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name="cnn_news",
#     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
# )

# client.create_collection(
#     collection_name="testing",
#     vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
# )

# assert client.collection_exists(collection_name="cnn_news")
# collection_info = client.get_collection("cnn_news")
# print(collection_info)

#insert in qdrant with dense vectors

# ids = []
# for idx, vector in enumerate(openai_vectors):
    # for j in i:
    #  print(i)
    #  print(type(i))
    #  print(type(j))

client.upsert(
    collection_name="cnn_news",
        points=[
            models.PointStruct(
                id=1,
                payload={
                "data": "news",
                },
                vector=openai_vectors[0]
            ),
        ],
)
    # ids.append(idx)

# test =[-0.01984037458896637, 0.01448314543813467, -0.019018713384866714]
# client.upsert(
#     collection_name="testing",
#         points=[
#             models.PointStruct(
#                 id=1,
#                 payload={
#                 "color": "red",
#                 },
#                 vector= test,
#             ),
#         ],
# )


#search
results = client.search(
    collection_name="cnn_news",
    query_vector=openai_vectors[0],
    limit=10,
    with_vectors=True  # 設置為 True 以返回向量
)
print(results)

#retrieve
retrieved_point = client.retrieve(
    collection_name="cnn_news",
    ids=[1],
)
print(retrieved_point)

# #get collection
# collection_info = client.get_collection("testing")
# print(collection_info)

# results, next_page = client.scroll(
#     collection_name="testing",
#     limit=100,                # 每次返回的資料點數量
#     offset=None,              # 可選的起始偏移量
#     with_vectors=True,        # 設為 True 來檢索向量
#     with_payload=True,        # 設為 True 來檢索 payload（附加資料)
# )
# print(results, next_page)

