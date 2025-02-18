import typing as t
from typing import List
import numpy as np
import MySQLdb
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)

# create collections
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="cnn_news_chunk_vectors",
    vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
)

# assert client.collection_exists(collection_name="cnn_news_chunk_vectors")
# collection_info = client.get_collection("cnn_news_chunk_vectors")
# print(collection_info)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
cur.execute("select PK_CHUNK, text, Article_PK from my_database.CHUNK_CNN_NEWS")
#只embed text, 其他插入payload
unique_id = 0
for row in cur.fetchall():
    row: t.Tuple
    pk_chunk, text, article_pk = row
    print(f'one of the row {row}')
    text_openai_vectors = embeddings.embed_documents([text])
    text_openai_vectors: t.List[List[float]]
    for each_text_openai_vectors in text_openai_vectors:
        print(f'unique_id:{unique_id}')
        client.upsert(
            collection_name="cnn_news_chunk_vectors",
                points=[
                    models.PointStruct(
                        id=unique_id,
                        payload={
                        "pk_chunk": pk_chunk,
                        "text": text,
                        "article_pk": article_pk
                        },
                        vector=each_text_openai_vectors
                    ),
                ],
        )
        unique_id+=1

    # print(openai_vectors)
    # print(type(openai_vectors))
    # d = np.array(openai_vectors)
    # print(d.shape)
    # raise ValueError



#search
# results = client.search(
#     collection_name="cnn_news",
#     query_vector=openai_vectors[0],
#     limit=10,
#     with_vectors=True  # 設置為 True 以返回向量
# )
# print(results)
#retrieve
# retrieved_point = client.retrieve(
#     collection_name="cnn_news_chunk_vectors",
#     ids=[1],
# )
# print(retrieved_point)


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

#CURL -L -X GET 'http://localhost:6333/collections/cnn_news_chunk_vectors/points/1'
#curl -X DELETE "http://localhost:6333/collections/cnn_news_chunk_vectors"